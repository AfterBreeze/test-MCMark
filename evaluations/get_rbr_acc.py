#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCMark-RBR multi-bit watermark evaluation.

Detection algorithm:
    For each token x_{t+1} at position t, and for each layer j:
        1. Reconstruct the layer-specific seed (same as generation time)
        2. Reconstruct the RNG state → shuffle, r_t, aux
        3. Derive: bit_index = aux % payload_bits
                   mask_h    = (aux >> 17) & 1
                   green partition (split_k=0 means channel-0 is green, etc.)
        4. Determine if x_{t+1} is in the "green" channel
        5. Accumulate evidence: if green → supports local_bit=1
                                          → supports payload[bit_index] = 1 XOR mask_h
        6. After all tokens: normalize by HitRate and decide each bit

Usage:
    python evaluations/get_rbr_acc.py \
        --input_file results/dolly_cw/Llama_3.2_3B_Instruct/mcmark_rbr/text_generation.txt \
        --num_layers 10 \
        --payload_bits 32 \
        --model_str meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
import torch
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm


def load_wp_rbr(wp_str, payload: bytes, payload_bits: int):
    """Reconstruct WatermarkLogitsProcessor for MCMark-RBR from repr string."""
    import re
    from watermarks import (
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
        NGramHashing,
    )
    from watermarks.mcmark_rbr import MC_RBR_Reweight

    # Parse private_key
    key_match = re.search(r"private_key=(b'.*?')", wp_str)
    assert key_match, f"Could not parse private_key from: {wp_str[:80]}"
    private_key = eval(key_match.group(1))

    # Parse num_layers and n
    layers_match = re.search(r"MC_RBR_Reweight\(num_layers=(\d+),\s*n=(\d+)\)", wp_str)
    assert layers_match, f"Could not find MC_RBR_Reweight in: {wp_str}"
    num_layers = int(layers_match.group(1))
    n = int(layers_match.group(2))

    watermark_key_list = [
        NGramHashing(PrevN_ContextCodeExtractor(2), ignore_history=False)
    ]
    reweight = MC_RBR_Reweight(num_layers=num_layers, n=n)

    wp = WatermarkLogitsProcessor(
        private_key=private_key,
        reweight=reweight,
        watermark_key_list=watermark_key_list,
        payload=payload,
        payload_bits=payload_bits,
    )
    return wp, num_layers, n


def recover_payload_rbr(
    output_ids: torch.LongTensor,
    wp,
    vocab_size: int,
    num_layers: int,
    n: int,
    payload_bits: int,
    prompt_tail_ids: list = None,
):
    """
    Recover payload bits from a watermarked token sequence.

    Returns:
        recovered_bits: list of ints (0 or 1), length = payload_bits
        debug_info: dict with per-bit vote counts for analysis
    """
    from watermarks.mcmark_rbr import MCMarkRBR_WatermarkCode

    # Prepend prompt tail for correct context reconstruction
    if prompt_tail_ids is not None and len(prompt_tail_ids) > 0:
        prefix = torch.tensor(
            [prompt_tail_ids], dtype=torch.long, device=output_ids.device
        )
        full_ids = torch.cat([prefix, output_ids], dim=1)
        offset = len(prompt_tail_ids)
    else:
        full_ids = output_ids
        offset = 0

    seq_len = full_ids.shape[1]

    # HitRate accumulators for each payload bit
    # chit_v[i] = number of times bit i was observed to support value v
    chit = [defaultdict(int) for _ in range(2)]   # chit[0][i], chit[1][i]
    ctotal = [defaultdict(int) for _ in range(2)]  # ctotal[0][i], ctotal[1][i]

    # Initialize watermark key history
    wp.reset_watermark_key(1)

    for t in range(1, seq_len - 1):
        context = full_ids[:, :t + 1]
        current_token = full_ids[:, t + 1]

        mask, seeds = wp._get_codes(context)

        if mask[0]:
            continue  # repeated context, masked during generation too

        # Only score actual output tokens (skip prompt tail prefix)
        if t + 1 < offset:
            continue

        seed = seeds[0]
        token = current_token[0].item()
        if token >= vocab_size:
            continue

        # Process each layer
        for layer in range(num_layers):
            layer_seed = (seed ^ (layer * 0x9E3779B9 + 0x6C62272E)) & 0xFFFFFFFF

            # Reconstruct the same RNG sequence used during generation
            rng = torch.Generator(device=full_ids.device).manual_seed(layer_seed)

            # Step 1: randperm → shuffle (we need unshuffle to find which channel token is in)
            shuffle = torch.randperm(vocab_size, generator=rng, device=full_ids.device)
            unshuffle = torch.argsort(shuffle)

            # Step 2: r_t from rng
            r_t = torch.randint(
                low=0, high=n, size=(1,),
                dtype=torch.long, generator=rng, device=full_ids.device,
            ).item()

            # Step 3: aux from rng → bit_index, mask_h
            aux = torch.randint(
                low=0, high=2**31, size=(1,),
                dtype=torch.long, generator=rng, device=full_ids.device,
            ).item()
            bit_index = aux % payload_bits
            mask_h = (aux >> 17) & 1

            # Determine which channel the current token landed in
            shuffled_pos = unshuffle[token].item()
            if vocab_size % n == 0:
                channel = shuffled_pos // (vocab_size // n)
            else:
                channel = -1
                for n_idx in range(n):
                    end = round(vocab_size * (n_idx + 1) / n)
                    if shuffled_pos < end:
                        channel = n_idx
                        break

            if channel < 0:
                continue

            # Infer what local_bit was embedded (local_bit = (channel - r_t) % n)
            # For n=2: this is 0 or 1
            inferred_local_bit = (channel - r_t) % n  # 0 or 1

            # XOR-undo the masking to recover the implied payload bit
            # local_bit = payload[bit_index] XOR mask_h
            # → payload[bit_index] = local_bit XOR mask_h
            implied_payload_bit = inferred_local_bit ^ mask_h  # 0 or 1

            # Accumulate HitRate evidence
            chit[implied_payload_bit][bit_index] += 1

            # Update total opportunity counts for normalization
            # (both hypotheses get a count each step)
            ctotal[0][bit_index] += 1
            ctotal[1][bit_index] += 1

    # Decode: compare HitRate for bit=1 vs bit=0
    recovered_bits = []
    debug_per_bit = {}
    for i in range(payload_bits):
        total = ctotal[0].get(i, 0)
        hit0 = chit[0].get(i, 0)
        hit1 = chit[1].get(i, 0)

        if total == 0:
            recovered_bits.append(0)
            debug_per_bit[i] = {"hit0": 0, "hit1": 0, "total": 0, "decision": 0}
            continue

        rate0 = hit0 / max(1, total)
        rate1 = hit1 / max(1, total)
        decision = 1 if rate1 > rate0 else 0
        recovered_bits.append(decision)
        debug_per_bit[i] = {
            "hit0": hit0, "hit1": hit1, "total": total,
            "rate0": rate0, "rate1": rate1, "decision": decision
        }

    return recovered_bits, debug_per_bit


def compute_bit_accuracy(recovered_bits, ground_truth_bits):
    assert len(recovered_bits) == len(ground_truth_bits)
    correct = sum(r == g for r, g in zip(recovered_bits, ground_truth_bits))
    return correct / len(ground_truth_bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--n", type=int, default=2,
                        help="Number of vocabulary channels (should be 2)")
    parser.add_argument("--payload_bits", type=int, default=32)
    parser.add_argument("--model_str", type=str,
                        default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Print per-sample debug info")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_str)
    config = AutoConfig.from_pretrained(args.model_str)
    vocab_size = config.vocab_size
    print(f"Vocab size: {vocab_size}")

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    if args.max_samples is not None:
        lines = lines[:args.max_samples]

    # Filter to MCMark-RBR records only
    records = []
    for line in lines:
        d = json.loads(line)
        if "MC_RBR_Reweight" in d.get("watermark_processor", ""):
            records.append(d)

    print(f"Evaluating {len(records)} samples | "
          f"num_layers={args.num_layers}, n={args.n}, payload_bits={args.payload_bits}")

    bit_accuracies = []
    exact_matches = []

    for record in tqdm(records):
        if "payload_hex" in record and record["payload_hex"]:
            payload = bytes.fromhex(record["payload_hex"])
            gt_bits = np.unpackbits(
                np.frombuffer(payload, dtype=np.uint8)
            )[:args.payload_bits].tolist()
        else:
            payload = bytes(args.payload_bits // 8 + 1)
            gt_bits = [0] * args.payload_bits

        wp_str = record["watermark_processor"]
        wp, num_layers, n = load_wp_rbr(wp_str, payload, args.payload_bits)
        assert num_layers == args.num_layers, f"num_layers mismatch: {num_layers} vs {args.num_layers}"
        assert n == args.n, f"n mismatch: {n} vs {args.n}"

        output_ids = torch.tensor(record["output_ids"], dtype=torch.long).to(device)
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)

        wp.payload_bits = args.payload_bits
        prompt_tail_ids = record.get("prompt_tail_ids", None)

        recovered_bits, debug_info = recover_payload_rbr(
            output_ids, wp, vocab_size,
            num_layers=args.num_layers,
            n=args.n,
            payload_bits=args.payload_bits,
            prompt_tail_ids=prompt_tail_ids,
        )

        acc = compute_bit_accuracy(recovered_bits, gt_bits)
        bit_accuracies.append(acc)
        exact_matches.append(1.0 if acc == 1.0 else 0.0)

        if args.debug:
            print(f"  acc={acc:.4f} | recovered={recovered_bits[:8]}... "
                  f"gt={gt_bits[:8]}...")

    mean_acc = np.mean(bit_accuracies)
    median_acc = np.median(bit_accuracies)
    exact_rate = np.mean(exact_matches)

    print(f"\n{'='*60}")
    print(f"MCMark-RBR Evaluation Results")
    print(f"{'='*60}")
    print(f"num_layers     : {args.num_layers}")
    print(f"n              : {args.n}")
    print(f"payload_bits   : {args.payload_bits}")
    print(f"num_samples    : {len(bit_accuracies)}")
    print(f"mean bit acc   : {mean_acc:.4f}")
    print(f"median bit acc : {median_acc:.4f}")
    print(f"exact match    : {exact_rate:.4f}")
    print(f"{'='*60}")

    import os
    out_dir = os.path.dirname(args.input_file)
    out_path = os.path.join(out_dir, "rbr_acc.json")
    with open(out_path, "w") as f:
        json.dump({
            "num_layers": args.num_layers,
            "n": args.n,
            "payload_bits": args.payload_bits,
            "num_samples": len(bit_accuracies),
            "mean_bit_accuracy": mean_acc,
            "median_bit_accuracy": median_acc,
            "exact_match_rate": exact_rate,
            "per_sample": bit_accuracies,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
