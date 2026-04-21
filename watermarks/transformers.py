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
from .adamc import (
    AdaMC_Reweight,
    AdaMC_WatermarkCode,
    _compute_entropy,
    check_token_in_channel,
    extract_message_from_channel,
    compute_weighted_pvalue,
)
import json


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        private_key: any,
        reweight: AbstractReweight,  # sample strategy
        watermark_key_list: List[AbstractWatermarkKey],
    ):
        self.watermark_key_list = watermark_key_list
        self.private_key = private_key
        self.reweight = reweight

    def __repr__(self):
        watermark_str = ", ".join(
            [repr(watermark_key) for watermark_key in self.watermark_key_list]
        )

        res_str = f"WatermarkLogitsProcessor(private_key={repr(self.private_key)}, reweight={repr(self.reweight)}, watermark_key_list=[{watermark_str}])"

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

        if isinstance(self.reweight, AdaMC_Reweight):
            # --- AdaMC: entropy-adaptive multi-bit embedding ---
            # 1. Compute entropy for the current token distribution
            cur_probs = torch.softmax(scores, dim=-1)              # [bsz, vocab_size]
            entropy_vals = _compute_entropy(cur_probs)             # [bsz]
            # Use the mean entropy across the batch to decide n_t
            mean_H = entropy_vals.mean().item()
            n_t = self.reweight.get_n_for_entropy(mean_H)

            # 2. Build watermark code with message-derived channel index
            watermark_code, new_cursor = AdaMC_WatermarkCode.from_message(
                rng,
                scores.size(1),
                n_t=n_t,
                message_bits=self.reweight.message_bits,
                bit_cursor=self.reweight._bit_cursor,
                private_key=self.reweight.private_key,
                global_token_position=self.reweight._token_position,
            )
            # Advance state
            self.reweight._bit_cursor = new_cursor
            self.reweight._token_position += 1

        elif isinstance(self.reweight, MC_Reweight):
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

    def get_adamc_score(
        self,
        input_ids: LongTensor,
        vocab_size: int,
        current_token,
        model_logits: FloatTensor,
        global_token_position: int,
    ):
        """
        AdaMC detection: for each token, determine n_t from entropy of model_logits,
        reconstruct the message-derived channel index k_t, and check whether
        current_token falls in channel k_t.

        Args:
            input_ids:            context token IDs, shape [bsz, ctx_len].
            vocab_size:           vocabulary size.
            current_token:        list of observed token IDs, length bsz.
            model_logits:         raw logits from model at this position, [bsz, vocab_size].
            global_token_position: absolute position index in the generated sequence.

        Returns:
            matches: list of int (0 or 1) per batch item.
            n_t_list: list of int (n_t used) per batch item.
            extracted_bits: list of list[int] (recovered message bits) per batch item.
        """
        assert isinstance(self.reweight, AdaMC_Reweight)

        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]

        # Compute per-token entropy
        cur_probs = torch.softmax(model_logits, dim=-1)
        entropy_vals = _compute_entropy(cur_probs)

        bsz = input_ids.shape[0]
        matches = []
        n_t_list = []
        extracted_bits_list = []

        for b in range(bsz):
            H_b = entropy_vals[b].item()
            n_t = self.reweight.get_n_for_entropy(H_b)

            # Reconstruct watermark code (only shuffle matters for detection)
            shuffle_b = torch.randperm(vocab_size, generator=rng[b], device=input_ids.device)

            if n_t <= 1:
                # Skipped token: no match counted
                matches.append(0)
                n_t_list.append(1)
                extracted_bits_list.append([])
                continue

            # Reconstruct split_k for detection (from message using same PRNG)
            import math as _math
            bits_needed = int(_math.log2(n_t))
            from .adamc import _prng_mask_bits

            msg_bits_slice = [
                self.reweight.message_bits[(global_token_position * bits_needed + j) % len(self.reweight.message_bits)]
                for j in range(bits_needed)
            ]
            prng = _prng_mask_bits(self.reweight.private_key, global_token_position + b, bits_needed)
            masked = [msg_bits_slice[j] ^ prng[j] for j in range(bits_needed)]
            k_t = sum(masked[j] << (bits_needed - 1 - j) for j in range(bits_needed)) % n_t

            # Check if current_token[b] falls in channel k_t
            token_id = current_token[b].item() if hasattr(current_token[b], 'item') else int(current_token[b])
            in_channel = check_token_in_channel(token_id, shuffle_b, k_t, n_t, vocab_size)
            matches.append(int(in_channel))
            n_t_list.append(n_t)

            # Extract message bits from the observed token
            pos_in_shuffle = (shuffle_b == token_id).nonzero(as_tuple=True)[0]
            if pos_in_shuffle.numel() > 0:
                observed_pos = pos_in_shuffle[0].item()
                observed_k = int(observed_pos * n_t / vocab_size)
                e_bits = extract_message_from_channel(
                    observed_k, n_t, self.reweight.private_key, global_token_position + b
                )
            else:
                e_bits = []
            extracted_bits_list.append(e_bits)

        return matches, n_t_list, extracted_bits_list


class WatermarkLogitsProcessor_Baseline(LogitsProcessor):
    def __repr__(self):
        return f"WatermarkLogitsProcessor_Baseline()"

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        return scores
