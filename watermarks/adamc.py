#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaMC-Watermark: Entropy-Adaptive Multi-bit Multi-Channel Watermark for LLMs.

Core innovations over MCMark (ACL 2025):
  1. Entropy-adaptive channel count n_t: high-entropy tokens use large n (embed more bits),
     low-entropy tokens skip watermarking entirely (preserving text quality).
  2. Multi-bit message encoding: channel index k_t encodes message bits instead of
     being chosen randomly, enabling traceability (user ID, timestamp, etc.).
  3. Unbiasedness preserved: PRNG-based XOR masking ensures k_t is uniform from
     the perspective of any observer without the private key.
  4. Weighted detection statistic: non-uniform n_t per token requires a new
     weighted Bernoulli test; proven to be asymptotically N(0,1) under H0.

Usage:
    from watermarks import AdaMC_Reweight, AdaMC_WatermarkCode
    message = b"\\x12\\x34"   # 16-bit user ID
    reweight = AdaMC_Reweight(n_max=32, entropy_threshold=0.5, message=message)
    wp = WatermarkLogitsProcessor(private_key, reweight, watermark_key_list)
"""

import torch
import hashlib
import math
import time
from torch import FloatTensor, LongTensor, BoolTensor
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple

from .base import AbstractWatermarkCode, AbstractReweight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to a list of bits (MSB first)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _compute_entropy(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute Shannon entropy H = -sum(p * log(p)) for a probability distribution.

    Args:
        probs: [batch_size, vocab_size] probability tensor (must sum to 1).
        eps:   small value for numerical stability.

    Returns:
        entropy: [batch_size] tensor of entropy values (in nats).
    """
    log_probs = torch.log(probs + eps)
    return -torch.sum(probs * log_probs, dim=-1)


def _prng_mask_bits(private_key: bytes, position: int, num_bits: int) -> List[int]:
    """
    Generate a deterministic pseudo-random bit mask for XOR-encrypting message bits.
    This ensures that, from the perspective of any observer without the private key,
    the channel index k_t appears uniformly distributed, preserving unbiasedness.

    Args:
        private_key: secret bytes known only to the watermark owner.
        position:    token position index (ensures different mask per position).
        num_bits:    number of bits needed.

    Returns:
        List of bits (0 or 1), length = num_bits.
    """
    m = hashlib.sha256()
    m.update(private_key)
    m.update(position.to_bytes(8, "big"))
    digest = m.digest()
    bits = _bytes_to_bits(digest)
    # Extend if needed by hashing again
    while len(bits) < num_bits:
        m2 = hashlib.sha256()
        m2.update(digest)
        m2.update(b"extend")
        digest = m2.digest()
        bits.extend(_bytes_to_bits(digest))
    return bits[:num_bits]


# ---------------------------------------------------------------------------
# WatermarkCode
# ---------------------------------------------------------------------------

class AdaMC_WatermarkCode(AbstractWatermarkCode):
    """
    Watermark code for AdaMC.  Extends MCMark_WatermarkCode with:
      - n_t:   the channel count used for this specific token (adaptive).
      - msg_k: the message-derived channel index (0 … n_t-1).
               When n_t == 1 (low-entropy skip), msg_k is unused.
    """

    def __init__(
        self,
        shuffle: LongTensor,    # [bsz, vocab_size]  random permutation
        split_k: LongTensor,    # [bsz]              channel index (from message)
        n_t: int,               # channel count for this token
    ):
        self.shuffle = shuffle
        self.split_k = split_k
        self.n_t = n_t
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, List[torch.Generator]],
        vocab_size: int,
        split_num: int = 20,
    ) -> "AdaMC_WatermarkCode":
        """
        Fallback: create code with random channel selection (zero-bit mode).
        Signature matches MCMark_WatermarkCode.from_random for compatibility.
        """
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack([
                torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                for i in range(batch_size)
            ])
            split_k = torch.cat([
                torch.randint(0, split_num, (1,), generator=rng[i], device=rng[i].device)
                for i in range(batch_size)
            ], dim=0)
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
            split_k = torch.randint(0, split_num, (1,), generator=rng, device=rng.device)
        return cls(shuffle, split_k, n_t=split_num)

    @classmethod
    def from_message(
        cls,
        rng: Union[torch.Generator, List[torch.Generator]],
        vocab_size: int,
        n_t: int,
        message_bits: List[int],
        bit_cursor: int,
        private_key: bytes,
        global_token_position: int,
    ) -> Tuple["AdaMC_WatermarkCode", int]:
        """
        Create code where split_k encodes message bits (multi-bit mode).

        The channel index k is derived by:
          1. Read ⌊log₂(n_t)⌋ bits from message_bits starting at bit_cursor.
          2. XOR with PRNG mask (keyed on private_key + token_position)
             to ensure unbiasedness from observer's perspective.
          3. Interpret the XOR'd bits as an integer → k_t ∈ {0, …, n_t-1}.

        Args:
            rng:                    RNG for generating the vocabulary shuffle.
            vocab_size:             vocabulary size.
            n_t:                    channel count (must be power-of-2 for exact bit encoding).
            message_bits:           full message bit list.
            bit_cursor:             current position in message_bits.
            private_key:            bytes used for PRNG mask generation.
            global_token_position:  absolute token index in the generated sequence.

        Returns:
            (code, new_bit_cursor)
        """
        bits_needed = int(math.log2(n_t)) if n_t > 1 else 0

        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack([
                torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                for i in range(batch_size)
            ])
            # For simplicity, use the same message-derived k for all batch items
            # (In practice batch_size=1 for watermark generation)
            k_list = []
            for b in range(batch_size):
                if bits_needed == 0:
                    k_list.append(0)
                else:
                    msg_bits_slice = [
                        message_bits[(bit_cursor + j) % len(message_bits)]
                        for j in range(bits_needed)
                    ]
                    prng = _prng_mask_bits(private_key, global_token_position + b, bits_needed)
                    masked = [msg_bits_slice[j] ^ prng[j] for j in range(bits_needed)]
                    k = sum(masked[j] << (bits_needed - 1 - j) for j in range(bits_needed))
                    k = k % n_t  # safety clamp
                    k_list.append(k)
            split_k = torch.tensor(k_list, dtype=torch.long, device=rng[0].device)
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
            if bits_needed == 0:
                split_k = torch.tensor([0], dtype=torch.long, device=rng.device)
            else:
                msg_bits_slice = [
                    message_bits[(bit_cursor + j) % len(message_bits)]
                    for j in range(bits_needed)
                ]
                prng = _prng_mask_bits(private_key, global_token_position, bits_needed)
                masked = [msg_bits_slice[j] ^ prng[j] for j in range(bits_needed)]
                k = sum(masked[j] << (bits_needed - 1 - j) for j in range(bits_needed))
                k = k % n_t
                split_k = torch.tensor([k], dtype=torch.long, device=rng.device)

        new_cursor = bit_cursor + bits_needed
        return cls(shuffle, split_k, n_t=n_t), new_cursor


# ---------------------------------------------------------------------------
# Reweight
# ---------------------------------------------------------------------------

# Predefined n_t ladder based on entropy thresholds.
# n must be a power of 2 for exact bit encoding.
# Structure: list of (entropy_lower_bound, n_t) pairs, sorted ascending by bound.
_DEFAULT_N_LADDER = [
    (0.0,   1),    # H < θ_1 : skip (n=1, no watermark)
    (0.5,   2),    # 1 bit per token
    (1.0,   4),    # 2 bits per token
    (1.5,   8),    # 3 bits per token
    (2.0,  16),    # 4 bits per token
    (2.5,  32),    # 5 bits per token
]


class AdaMC_Reweight(AbstractReweight):
    """
    Entropy-adaptive multi-bit reweighter.

    Args:
        n_max:              maximum channel count (power of 2 recommended), e.g. 32.
        entropy_threshold:  minimum entropy (nats) to apply watermarking. Below this
                            the token is skipped (n_t = 1). Default 0.5 ≈ 0.72 bits.
        message:            bytes to embed as the watermark payload. Will be converted
                            to a bit list and embedded cyclically.
        private_key:        bytes used for PRNG masking (unbiasedness guarantee).
                            If None, a fixed demo key is used (NOT secure).
        n_ladder:           custom list of (entropy_lower_bound, n_t) pairs.
                            If None, uses _DEFAULT_N_LADDER up to n_max.
    """

    watermark_code_type = AdaMC_WatermarkCode

    def __init__(
        self,
        n_max: int = 32,
        entropy_threshold: float = 0.5,
        message: bytes = b"\x00",
        private_key: bytes = b"adamc_default_key",
        n_ladder: Optional[List[Tuple[float, int]]] = None,
    ):
        assert n_max >= 1
        self.n_max = n_max
        self.entropy_threshold = entropy_threshold
        self.message = message
        self.message_bits = _bytes_to_bits(message)
        self.private_key = private_key

        # Build n_ladder filtered by n_max
        if n_ladder is not None:
            self.n_ladder = n_ladder
        else:
            self.n_ladder = [
                (lb, n) for lb, n in _DEFAULT_N_LADDER if n <= n_max
            ]
            # Ensure at least (threshold, n_max) is the top rung
            if self.n_ladder[-1][1] < n_max:
                top_lb = self.n_ladder[-1][0] + 0.5
                self.n_ladder.append((top_lb, n_max))

        # State for message embedding (reset per generation call)
        self._bit_cursor: int = 0
        self._token_position: int = 0

    def reset_state(self):
        """Reset bit cursor and token position counter. Call before each generation."""
        self._bit_cursor = 0
        self._token_position = 0

    def __repr__(self) -> str:
        return (
            f"AdaMC_Reweight(n_max={self.n_max}, "
            f"entropy_threshold={self.entropy_threshold}, "
            f"message_len_bits={len(self.message_bits)})"
        )

    def get_n_for_entropy(self, H: float) -> int:
        """
        Map a scalar entropy value to a channel count using the n_ladder.

        Args:
            H: entropy in nats (scalar float).

        Returns:
            n_t: channel count, a power of 2 in {1, 2, 4, …, n_max}.
                 Returns 1 (skip) when H < entropy_threshold.
        """
        if H < self.entropy_threshold:
            return 1
        chosen_n = 1
        for lb, n in self.n_ladder:
            if H >= lb:
                chosen_n = n
            else:
                break
        return chosen_n

    # ------------------------------------------------------------------
    # Core reweight logic  (heavily reuses MCMark's γ-reweight)
    # ------------------------------------------------------------------

    def reweight_logits(
        self,
        code: AdaMC_WatermarkCode,
        p_logits: FloatTensor,
    ) -> FloatTensor:
        """
        Apply the multi-channel γ-reweight to p_logits.

        Identical to MCMark.reweight_logits, except code.n_t is adaptive
        and code.split_k is derived from message bits instead of being random.

        When n_t == 1 (low-entropy token), returns p_logits unchanged.
        """
        n = code.n_t
        if n == 1:
            # Skip: do not modify the distribution
            return p_logits

        def set_nan_to_zero(x: torch.Tensor) -> torch.Tensor:
            x[torch.isnan(x)] = 0
            return x

        s_logits = torch.gather(p_logits, -1, code.shuffle)
        s_probs = torch.softmax(s_logits, dim=-1)
        bsz, vocab_size = s_logits.shape

        # ---- partition vocabulary into n channels ----
        if vocab_size % n == 0:
            split_sums = s_probs.view(bsz, n, vocab_size // n).sum(dim=-1)  # [bsz, n]
        else:
            split_sums_list = []
            for n_idx in range(n):
                lo = round(vocab_size * n_idx / n)
                hi = round(vocab_size * (n_idx + 1) / n)
                split_sums_list.append(s_probs[:, lo:hi].sum(dim=-1, keepdim=True))
            split_sums = torch.cat(split_sums_list, dim=-1)  # [bsz, n]

        split_k = code.split_k.to(s_logits.device)

        scales = torch.minimum(
            n * torch.ones_like(split_sums), 1.0 / split_sums
        )  # [bsz, n]

        overflow_scales = (n * split_sums - 1) / split_sums  # [bsz, n]
        overflow_scales = set_nan_to_zero(overflow_scales)
        overflow_scales[overflow_scales < 0] = 0

        target_scales = scales[range(bsz), split_k]    # [bsz]
        target_sums   = split_sums[range(bsz), split_k]  # [bsz]

        remain_sums   = 1 - target_scales * target_sums  # [bsz]
        overflow_sums = (overflow_scales * split_sums).sum(dim=-1)  # [bsz]
        fill_scale    = remain_sums / overflow_sums  # [bsz]
        fill_scale    = set_nan_to_zero(fill_scale)

        split_mask = (
            torch.arange(0, n, device=s_logits.device).view(1, -1).repeat(bsz, 1)
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
                lo = round(vocab_size * n_idx / n)
                hi = round(vocab_size * (n_idx + 1) / n)
                reweighted_s_probs[:, lo:hi] = (
                    final_scale[:, n_idx].view(-1, 1) * s_probs[:, lo:hi]
                )

        reweighted_s_probs[reweighted_s_probs < 0] = 0

        reweighted_s_logits = torch.log(reweighted_s_probs)
        reweighted_logits   = torch.gather(reweighted_s_logits, -1, code.unshuffle)
        return reweighted_logits


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def check_token_in_channel(
    token_id: int,
    shuffle: LongTensor,     # [vocab_size]
    split_k: int,
    n_t: int,
    vocab_size: int,
) -> bool:
    """
    Check whether token_id falls in the watermarked channel split_k.

    Args:
        token_id:   the observed token index.
        shuffle:    the random permutation used during generation.
        split_k:    the target channel index.
        n_t:        number of channels.
        vocab_size: vocabulary size.

    Returns:
        True if token_id is in channel split_k under the shuffle.
    """
    # Position of token_id in the shuffled order
    pos = (shuffle == token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return False
    pos_scalar = pos[0].item()
    # Which channel does this position belong to?
    lo = round(vocab_size * split_k / n_t)
    hi = round(vocab_size * (split_k + 1) / n_t)
    return lo <= pos_scalar < hi


def extract_message_from_channel(
    k_t: int,
    n_t: int,
    private_key: bytes,
    global_token_position: int,
) -> List[int]:
    """
    Reverse the PRNG-XOR masking to recover original message bits from observed channel k_t.

    Returns:
        List of bits that were embedded at this token position.
    """
    bits_needed = int(math.log2(n_t)) if n_t > 1 else 0
    if bits_needed == 0:
        return []
    # Reconstruct k_t as bit array
    k_bits = [(k_t >> (bits_needed - 1 - j)) & 1 for j in range(bits_needed)]
    # XOR with same PRNG mask to undo encryption
    prng = _prng_mask_bits(private_key, global_token_position, bits_needed)
    original_bits = [k_bits[j] ^ prng[j] for j in range(bits_needed)]
    return original_bits


def compute_weighted_pvalue(
    match_flags: List[int],
    n_list: List[int],
) -> Tuple[float, float, float]:
    """
    Compute the weighted detection statistic and its p-value under H0.

    Under H0 (no watermark), each match_t ~ Bernoulli(1/n_t), independently.
    Weighted statistic: S = sum_t w_t * X_t,  where w_t = log2(n_t).
    E[S | H0] = sum_t w_t / n_t
    Var[S | H0] = sum_t w_t^2 * (1/n_t) * (1 - 1/n_t)
    Z = (S - E[S]) / sqrt(Var[S]) ->^{CLT} N(0,1)
    p-value = P(Z >= z_obs | H0) = 1 - Phi(z_obs)

    Args:
        match_flags: list of 0/1, whether each high-entropy token matched.
        n_list:      list of n_t values corresponding to each token.

    Returns:
        (z_score, p_value, raw_weighted_score)
    """
    import scipy.stats as stats

    assert len(match_flags) == len(n_list)
    S = 0.0
    E_S = 0.0
    Var_S = 0.0
    for x, n in zip(match_flags, n_list):
        if n <= 1:
            continue
        w = math.log2(n)
        S     += w * x
        E_S   += w / n
        Var_S += (w ** 2) * (1.0 / n) * (1.0 - 1.0 / n)

    if Var_S < 1e-10:
        return 0.0, 1.0, S

    z = (S - E_S) / math.sqrt(Var_S)
    p = stats.norm.sf(z)   # right-tail p-value
    return z, p, S
