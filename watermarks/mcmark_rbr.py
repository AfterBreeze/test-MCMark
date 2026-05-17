#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCMark-RBR: Multi-bit Watermarking with Random Bit Routing

Core innovation: Rather than using segment-based routing (like MC²MARK),
we route each token to a payload bit using bit_index = seed % payload_bits,
where seed is derived from context n-gram hashing. This makes the
token-to-bit mapping unpredictable to adversaries, providing security
against targeted bit attacks.

Combined with:
- MCCR (Multi-Channel Colored Reweighting) from MC²MARK for unbiased embedding
- MLSR (Multi-Layer Sequential Reweighting) from MC²MARK for signal strength
- XOR masking (whitening) to remove payload bias
- HitRate-normalized detector adapted for random routing
"""

import torch
from torch import FloatTensor, LongTensor
from torch.nn import functional as F
import time
from typing import Union, List, Optional

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class MCMarkRBR_WatermarkCode(AbstractWatermarkCode):
    """
    Watermark code for one layer of MCMark-RBR generation.

    Fields:
        shuffle:    [bsz, vocab_size] - random permutation of vocab
        split_k:    [bsz] - which channel (0 or 1) is "green" (carries embedded bit=1)
        r_t:        [bsz] - base random channel index (for decoding)
        mask_h:     [bsz] - XOR mask bit (whitening transform)
        bit_index:  [bsz] - which payload bit this layer/token step is responsible for
    """

    def __init__(
        self,
        shuffle: LongTensor,
        split_k: LongTensor,
        r_t: LongTensor,
        mask_h: LongTensor,
        bit_index: LongTensor,
    ):
        self.shuffle = shuffle
        self.split_k = split_k
        self.r_t = r_t
        self.mask_h = mask_h
        self.bit_index = bit_index
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, List[torch.Generator]],
        vocab_size: int,
        split_num: int,               # should be 2 for binary watermark
        payload_bits: int,            # total number of payload bits
        payload_list: Optional[List[int]],  # [payload_bits] or None (zero-bit mode)
        layer_idx: int = 0,           # layer index for multi-layer generation
    ):
        """
        Generate watermark code for one layer of RBR generation.

        The seed used here already encodes the layer index (caller XORs layer_idx
        into the seed before passing to rng). The bit_index and mask_h are derived
        deterministically from the seed so they can be reconstructed during detection.

        For each batch element i:
            seed_i  = rng state (already layer-specific, set by caller)
            bit_index_i = seed_i % payload_bits
            mask_h_i    = (seed_i >> 17) & 1          # 1-bit whitening mask
            local_bit_i = payload[bit_index_i] XOR mask_h_i
            r_t_i       = randint(0, split_num)        # from rng
            split_k_i   = (r_t_i + local_bit_i) % split_num
        """
        if isinstance(rng, list):
            batch_size = len(rng)

            # Generate random permutation (shuffle) from rng
            shuffle = torch.stack([
                torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                for i in range(batch_size)
            ])

            # Generate r_t (base random channel) from rng
            r_t = torch.cat([
                torch.randint(
                    low=0, high=split_num, size=(1,),
                    dtype=torch.long, generator=rng[i], device=rng[i].device,
                )
                for i in range(batch_size)
            ], dim=0)

            # Derive bit_index and mask_h from the RNG state
            # We sample an extra integer to get bit_index and mask bits
            aux = torch.cat([
                torch.randint(
                    low=0, high=2**31, size=(1,),
                    dtype=torch.long, generator=rng[i], device=rng[i].device,
                )
                for i in range(batch_size)
            ], dim=0)

            bit_index = aux % payload_bits                    # [bsz] which payload bit
            mask_h = (aux >> 17) & 1                          # [bsz] XOR mask bit

            if payload_list is None:
                # Zero-bit mode: pure random split_k
                split_k = r_t
            else:
                # Multi-bit RBR mode
                payload_tensor = torch.tensor(
                    payload_list, dtype=torch.long, device=r_t.device
                )
                # Gather the payload bit for each batch element
                gathered_bits = payload_tensor[bit_index]     # [bsz]
                local_bits = (gathered_bits ^ mask_h) % split_num  # XOR whitening
                split_k = (r_t + local_bits) % split_num     # [bsz]

        else:
            # Single-generator path (kept for compatibility)
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
            r_t = torch.randint(
                low=0, high=split_num, size=(1,),
                dtype=torch.long, generator=rng, device=rng.device,
            )
            aux = torch.randint(
                low=0, high=2**31, size=(1,),
                dtype=torch.long, generator=rng, device=rng.device,
            )
            bit_index = aux % payload_bits
            mask_h = (aux >> 17) & 1

            if payload_list is None:
                split_k = r_t
            else:
                payload_tensor = torch.tensor(
                    payload_list, dtype=torch.long, device=r_t.device
                )
                gathered_bits = payload_tensor[bit_index.squeeze()]
                local_bits = (gathered_bits ^ mask_h.squeeze()) % split_num
                split_k = (r_t + local_bits.unsqueeze(0)) % split_num

        return cls(shuffle, split_k, r_t, mask_h, bit_index)


class MC_RBR_Reweight(AbstractReweight):
    """
    Multi-layer Random Bit Routing reweight.

    Wraps the underlying n=2 MCMark reweight logic with:
    1. XOR masking (whitening) on payload bits
    2. Multi-layer sequential reweighting (MLSR)
    3. Random bit routing: bit_index = aux % payload_bits
    """

    watermark_code_type = MCMarkRBR_WatermarkCode

    def __init__(self, num_layers: int = 10, n: int = 2):
        """
        Args:
            num_layers: number of reweighting layers (m in MC²MARK; default 10)
            n:          number of vocabulary channels (should be 2 for binary watermark)
        """
        self.num_layers = num_layers
        self.n = n

    def __repr__(self):
        return f"MC_RBR_Reweight(num_layers={self.num_layers}, n={self.n})"

    def reweight_logits(
        self, code: MCMarkRBR_WatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        """
        Single-layer MCCR reweighting (identical logic to MC_Reweight with n=2).
        Called once per layer inside WatermarkLogitsProcessor._core_rbr().
        """

        def set_nan_to_zero(x):
            x[torch.isnan(x)] = 0
            return x

        s_logits = torch.gather(p_logits, -1, code.shuffle)
        s_probs = torch.softmax(s_logits, dim=-1)
        bsz, vocab_size = s_logits.shape

        n = self.n
        if vocab_size % n == 0:
            splits = (
                torch.arange(start=0, end=vocab_size)
                .reshape(n, vocab_size // n)
                .to(p_logits.device)
            )
            split_sums = s_probs.view(bsz, n, vocab_size // n).sum(dim=-1)  # [bsz, n]
        else:
            # Uneven split fallback
            splits = []
            split_sums_list = []
            for n_idx in range(n):
                cur_split = list(range(
                    round(vocab_size * n_idx / n),
                    round(vocab_size * (n_idx + 1) / n),
                ))
                splits.append(cur_split)
                split_sums_list.append(s_probs[:, cur_split].sum(dim=-1, keepdim=True))
            split_sums = torch.cat(split_sums_list, dim=-1)  # [bsz, n]

        split_k = code.split_k.to(s_logits.device)  # [bsz]

        # MCCR scaling factors
        scales = torch.minimum(
            n * torch.ones_like(split_sums), 1.0 / split_sums
        )  # [bsz, n]

        overflow_scales = (n * split_sums - 1) / split_sums  # [bsz, n]
        overflow_scales = set_nan_to_zero(overflow_scales)
        overflow_scales[overflow_scales < 0] = 0

        target_scales = scales[range(bsz), split_k]   # [bsz]
        target_sums   = split_sums[range(bsz), split_k]  # [bsz]

        remain_sums   = 1 - target_scales * target_sums   # [bsz]
        overflow_sums = (overflow_scales * split_sums).sum(dim=-1)  # [bsz]
        fill_scale    = remain_sums / overflow_sums        # [bsz]
        fill_scale    = set_nan_to_zero(fill_scale)

        split_mask = (
            torch.arange(0, n).to(s_logits.device).view(1, -1).repeat(bsz, 1)
            == split_k.view(-1, 1).repeat(1, n)
        )
        final_scale = torch.where(
            split_mask,
            target_scales.view(-1, 1).repeat(1, n),
            fill_scale.view(-1, 1) * overflow_scales,
        )  # [bsz, n]

        reweighted_s_probs = torch.zeros_like(s_probs)
        if vocab_size % n == 0:
            reweighted_s_probs = (
                final_scale.view(bsz, n, 1)
                .expand(-1, -1, vocab_size // n)
                .reshape(bsz, vocab_size)
                * s_probs
            )
        else:
            for n_idx in range(n):
                cur_split = splits[n_idx]
                reweighted_s_probs[:, cur_split] = (
                    final_scale[:, n_idx].view(-1, 1) * s_probs[:, cur_split]
                )

        reweighted_s_probs[reweighted_s_probs < 0] = 0
        # Clamp to avoid log(0) = -inf which causes NaN in subsequent softmax layers
        reweighted_s_probs = torch.clamp(reweighted_s_probs, min=1e-10)
        reweighted_s_logits = torch.log(reweighted_s_probs)
        reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)
        return reweighted_logits
