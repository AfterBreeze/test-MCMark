#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor

from .base import (
    AbstractReweight,
    AbstractContextCodeExtractor,
    AbstractScore,
    AbstractWatermarkKey,
)
from typing import List
from .dipmark import Dip_Reweight
from .mcmark import MC_Reweight
from .sta import STA_Reweight
from .unigram import Unigram_Reweight
import json


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        private_key: any,
        reweight: AbstractReweight,  # sample strategy
        watermark_key_list: List[AbstractWatermarkKey],
        payload: bytes = None,    # optional: raw payload bytes for multi-bit mode
        payload_bits: int = 64,   # number of payload bits to embed
    ):
        self.watermark_key_list = watermark_key_list
        self.private_key = private_key
        self.reweight = reweight
        self.payload = payload
        self.payload_bits = payload_bits

        if payload is not None:
            import numpy as np
            bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
            # Store as plain Python list (not tensor) so it's safely picklable
            # across multiprocessing spawn boundaries without CUDA shared memory issues.
            self.payload_list = bits[:payload_bits].tolist()
        else:
            self.payload_list = None

    def __repr__(self):
        watermark_str = ", ".join(
            [repr(watermark_key) for watermark_key in self.watermark_key_list]
        )

        res_str = f"WatermarkLogitsProcessor(private_key={repr(self.private_key)}, reweight={repr(self.reweight)}, watermark_key_list=[{watermark_str}], payload_bits={self.payload_bits})"

        return res_str

    def get_rng_seed(self, key_list) -> any:
        import hashlib

        m = hashlib.sha256()
        # m.update(self.private_key)
        for key in key_list:
            m.update(key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    def reset_watermark_key(self, batch_size):
        for watermark_key in self.watermark_key_list:
            watermark_key.reset(batch_size)

    def _get_codes(self, input_ids: LongTensor):
        batch_size = input_ids.size(0)

        mask = []
        seeds = []
        for batch_idx in range(batch_size):
            cur_mask = 0
            key_list = [self.private_key]
            for watermark_key in self.watermark_key_list:
                cur_wm_mask, cur_wm_key = watermark_key.generate_key_and_mask(
                    input_ids[batch_idx], batch_idx
                )
                if cur_wm_key is not None:
                    key_list.append(cur_wm_key)
                cur_mask = cur_mask or cur_wm_mask
            mask.append(cur_mask)
            seeds.append(self.get_rng_seed(key_list))

        return mask, seeds

    def _core(self, input_ids: LongTensor, scores: FloatTensor):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device, dtype=torch.bool)

        if isinstance(self.reweight, MC_Reweight):
            if self.payload_list is not None:
                # Multi-bit mode: each token's bit index is derived from its seed
                # (position-independent: depends on context hash, not absolute position)
                # Build message_bits on-the-fly from plain Python list (avoids pickle issues)
                message_bits = torch.tensor(
                    [self.payload_list[seed % self.payload_bits] for seed in seeds],
                    dtype=torch.long,
                    device=scores.device,
                )
                watermark_code = self.reweight.watermark_code_type.from_random(
                    rng, scores.size(1), self.reweight.n, message_bits=message_bits
                )
            else:
                # Zero-bit mode (original behavior)
                watermark_code = self.reweight.watermark_code_type.from_random(
                    rng, scores.size(1), self.reweight.n
                )
        else:
            watermark_code = self.reweight.watermark_code_type.from_random(
                rng, scores.size(1)
            )
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scores)
        return mask, reweighted_scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        mask, reweighted_scores = self._core(input_ids, scores)
        return torch.where(mask[:, None], scores, reweighted_scores)

    def get_green_token_quantile(
        self, input_ids: LongTensor, vocab_size, current_token, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, Dip_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)

        # calculate the score here
        token_quantile = [
            (torch.where(watermark_code.shuffle[i] == current_token[i])[0] + 1)
            / vocab_size
            for i in range(input_ids.shape[0])
        ]

        return token_quantile

    def get_sta_score(
        self, input_ids: LongTensor, vocab_size, current_token, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, STA_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)

        green_list_size = round(self.reweight.gamma * vocab_size)
        scores = [
            torch.tensor(
                current_token[i] in watermark_code.shuffle[i][:green_list_size]
            ).float()
            for i in range(input_ids.shape[0])
        ]

        return scores

    def get_unigram_score(
        self, input_ids: LongTensor, vocab_size, current_token, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, Unigram_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)

        green_list_size = round(self.reweight.gamma * vocab_size)
        scores = [
            torch.tensor(
                current_token[i] in watermark_code.shuffle[i][:green_list_size]
            ).float()
            for i in range(input_ids.shape[0])
        ]

        return scores

    def get_n_res(
        self, input_ids: LongTensor, vocab_size, current_token, cur_n, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, MC_Reweight)
        assert self.reweight.n == cur_n
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, vocab_size, self.reweight.n
        )

        # cur_n=32000
        splits = []
        if vocab_size % cur_n == 0:
            splits = (
                torch.arange(start=0, end=vocab_size)
                .reshape(cur_n, vocab_size // cur_n)
                .to(input_ids.device)
            )
        else:
            for n_idx in range(cur_n):
                splits.append(
                    list(
                        range(
                            round(vocab_size * n_idx / cur_n),
                            round(vocab_size * (n_idx + 1) / cur_n),
                        )
                    )
                )

        scores = []
        for bsz_idx in range(input_ids.shape[0]):
            cur_k = watermark_code.split_k[bsz_idx]
            if current_token[bsz_idx] in watermark_code.shuffle[bsz_idx][splits[cur_k]]:
                scores.append(1)
            else:
                scores.append(0)
        return scores

    def get_multibit_channel(
        self, input_ids: LongTensor, vocab_size, current_token, cur_n, debug=False
    ):
        """
        Multi-bit detection: for each token, infer which payload bit was embedded
        and what value it takes.

        Returns:
            list of (bit_index, inferred_bit_value) tuples, one per batch element.
            bit_index: which payload bit this token position is responsible for
            inferred_bit_value: inferred value (0 to cur_n-1) of that bit
        """
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, MC_Reweight)
        assert self.reweight.n == cur_n
        mask = torch.tensor(mask, device=input_ids.device)

        # Generate watermark code in zero-bit mode to recover shuffle, unshuffle, and r_t
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, vocab_size, cur_n, message_bits=None
        )

        results = []
        for bsz_idx in range(input_ids.shape[0]):
            seed = seeds[bsz_idx]
            bit_index = seed % self.payload_bits  # which bit this token is responsible for

            # Find which channel c the token landed in.
            # unshuffle[token_id] = position of token in the shuffled vocab.
            # channel = floor(shuffled_position / (vocab_size / n))
            token = current_token[bsz_idx].item()

            # Skip special tokens that are outside the watermarked vocab range
            if token >= vocab_size:
                results.append((bit_index, -1))
                continue

            shuffled_pos = watermark_code.unshuffle[bsz_idx][token].item()

            if vocab_size % cur_n == 0:
                c = shuffled_pos // (vocab_size // cur_n)
            else:
                # For uneven splits, find which split this position belongs to
                c = -1
                start = 0
                for n_idx in range(cur_n):
                    end = round(vocab_size * (n_idx + 1) / cur_n)
                    if shuffled_pos < end:
                        c = n_idx
                        break
                    start = end

            # Infer the embedded bit: m = (c - r_t) mod n
            r_t = watermark_code.r_t[bsz_idx].item() if watermark_code.r_t is not None else 0
            if c == -1:
                inferred_bit = -1  # failed to find channel
            else:
                inferred_bit = (c - r_t) % cur_n

            results.append((bit_index, inferred_bit))

        return results


class WatermarkLogitsProcessor_Baseline(LogitsProcessor):
    def __repr__(self):
        return f"WatermarkLogitsProcessor_Baseline()"

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        return scores
