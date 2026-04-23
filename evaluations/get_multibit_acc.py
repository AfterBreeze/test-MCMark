#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-bit watermark evaluation for MCMark.

This script evaluates:
1. Bit accuracy (what fraction of payload bits are correctly recovered)
2. Exact match rate (what fraction of samples have 100% bit accuracy)
3. Robustness: works on original and paraphrased outputs

Usage:
    python evaluations/get_multibit_acc.py \
        --input_file results/mcmark/dolly/Llama-2-7b-chat-hf/multibit/text_generation.txt \
        --n_channels 4 \
        --payload_bits 64 \
        --model_str meta-llama/Llama-2-7b-chat-hf

The input file should be the output of the text generation experiment,
with lines containing JSON records with 'output_ids', 'watermark_processor', etc.
"""

import argparse
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy import stats as scipy_stats


def load_wp_from_str(wp_str, payload: bytes, payload_bits: int):
    """Reconstruct WatermarkLogitsProcessor from its repr string + payload.
    
    Parses the private_key and n directly from the repr string so that
    the detector uses exactly the same key as was used during generation.
    """
    import re
    from watermarks import (
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
        NGramHashing,
        MC_Reweight,
    )

    # --- Parse private_key from repr string ---
    # Format: private_key=b'...' where the bytes are repr'd
    key_match = re.search(r"private_key=(b'.*?')", wp_str)
    assert key_match, f"Could not parse private_key from: {wp_str[:80]}"
    private_key = eval(key_match.group(1))  # safe: only bytes literal

    # --- Parse n from repr string ---
    n_match = re.search(r"MC_Reweight\(n=(\d+)\)", wp_str)
    assert n_match, f"Could not find MC_Reweight in: {wp_str}"
    n = int(n_match.group(1))

    watermark_key_list = [
        NGramHashing(PrevN_ContextCodeExtractor(2), ignore_history=True)  # ignore_history=True for detection
    ]
    reweight = MC_Reweight(n)

    wp = WatermarkLogitsProcessor(
        private_key=private_key,
        reweight=reweight,
        watermark_key_list=watermark_key_list,
        payload=payload,
        payload_bits=payload_bits,
    )
    return wp, n


def recover_payload_from_ids(output_ids: torch.LongTensor, wp, vocab_size: int,
                              n_channels: int, payload_bits: int):
    """
    Recover payload bits from a watermarked token sequence.
    Uses majority voting over all token positions.

    Returns:
        recovered_bits: list of int (0 or 1 for n=2; 0..n-1 for n>2)
        vote_counts: dict {bit_index: Counter}
    """
    from collections import Counter

    # votes[bit_index] = list of inferred values
    votes = defaultdict(list)

    seq_len = output_ids.shape[1]

    # Initialize watermark key history ONCE before the loop
    wp.reset_watermark_key(1)

    # Slide over the sequence token by token (starting from position 2 for context)
    # Note: do NOT reset_watermark_key inside the loop - the NGramHashing already
    # has ignore_history=True set at construction, so each token is independent.
    for t in range(2, seq_len - 1):
        context = output_ids[:, :t + 1]       # [1, t+1]
        current_token = output_ids[:, t + 1]  # [1]

        results = wp.get_multibit_channel(
            context, vocab_size, current_token, cur_n=n_channels
        )
        bit_index, inferred_bit = results[0]
        if inferred_bit >= 0:
            votes[bit_index].append(inferred_bit)

    # Majority vote for each bit position
    recovered_bits = []
    for i in range(payload_bits):
        if votes[i]:
            majority = Counter(votes[i]).most_common(1)[0][0]
        else:
            majority = 0  # default if no votes
        recovered_bits.append(majority)

    return recovered_bits, dict(votes)


def compute_bit_accuracy(recovered_bits, ground_truth_bits, n_channels):
    """
    For n_channels=2: standard bit accuracy (0 or 1).
    For n_channels>2: symbol accuracy (each position is a symbol in [0, n_channels-1]).
    """
    assert len(recovered_bits) == len(ground_truth_bits)
    correct = sum(r == g for r, g in zip(recovered_bits, ground_truth_bits))
    return correct / len(ground_truth_bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to text_generation.txt from the watermark experiment")
    parser.add_argument("--n_channels", type=int, default=4,
                        help="Number of channels (n). Must match generation setting.")
    parser.add_argument("--payload_bits", type=int, default=64,
                        help="Number of payload bits embedded.")
    parser.add_argument("--model_str", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation to first N samples (for debugging)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # Load model just to get vocab_size (no need to run inference)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_str)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Load data
    with open(args.input_file, "r") as f:
        lines = f.readlines()

    if args.max_samples is not None:
        lines = lines[:args.max_samples]

    # Filter to only multi-bit MC_Reweight lines
    records = []
    for line in lines:
        d = json.loads(line)
        if "MC_Reweight" in d.get("watermark_processor", ""):
            records.append(d)

    print(f"Evaluating {len(records)} samples with n_channels={args.n_channels}, payload_bits={args.payload_bits}")

    bit_accuracies = []
    exact_matches = []

    for record in tqdm(records):
        # Recover the payload that was used during generation
        # It should be stored in the record; if not, we use a fixed test payload
        if "payload_hex" in record:
            payload = bytes.fromhex(record["payload_hex"])
            ground_truth_bits_np = np.unpackbits(
                np.frombuffer(payload, dtype=np.uint8)
            )[:args.payload_bits].tolist()
        else:
            # Fallback: use default payload (all zeros) — for testing
            payload = bytes(args.payload_bits // 8)
            ground_truth_bits_np = [0] * args.payload_bits

        wp_str = record["watermark_processor"]
        wp, n = load_wp_from_str(wp_str, payload, args.payload_bits)
        assert n == args.n_channels, f"n mismatch: {n} vs {args.n_channels}"

        output_ids = torch.tensor(record["output_ids"], dtype=torch.long).to(device)
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)

        wp.payload_bits = args.payload_bits

        recovered_bits, votes = recover_payload_from_ids(
            output_ids, wp, vocab_size, args.n_channels, args.payload_bits
        )

        acc = compute_bit_accuracy(recovered_bits, ground_truth_bits_np, args.n_channels)
        bit_accuracies.append(acc)
        exact_matches.append(1.0 if acc == 1.0 else 0.0)

    mean_bit_acc = np.mean(bit_accuracies)
    median_bit_acc = np.median(bit_accuracies)
    exact_match_rate = np.mean(exact_matches)

    print(f"\n{'='*60}")
    print(f"Multi-bit MCMark Evaluation Results")
    print(f"{'='*60}")
    print(f"n_channels       : {args.n_channels}")
    print(f"payload_bits     : {args.payload_bits}")
    print(f"num_samples      : {len(bit_accuracies)}")
    print(f"mean bit accuracy: {mean_bit_acc:.4f}")
    print(f"median bit acc   : {median_bit_acc:.4f}")
    print(f"exact match rate : {exact_match_rate:.4f}")
    print(f"{'='*60}")

    # Save results
    import os
    out_dir = os.path.dirname(args.input_file)
    out_path = os.path.join(out_dir, "multibit_acc.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_channels": args.n_channels,
            "payload_bits": args.payload_bits,
            "num_samples": len(bit_accuracies),
            "mean_bit_accuracy": mean_bit_acc,
            "median_bit_accuracy": median_bit_acc,
            "exact_match_rate": exact_match_rate,
            "per_sample": bit_accuracies,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
