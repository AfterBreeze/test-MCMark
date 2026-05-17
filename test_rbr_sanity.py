#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick local sanity test for MCMark-RBR.
Simulates the full generate → detect loop on a synthetic token sequence.

Run with:
    cd /Users/afterbreeze/test/test-MCMark
    python test_rbr_sanity.py
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watermarks.mcmark_rbr import MCMarkRBR_WatermarkCode, MC_RBR_Reweight
from watermarks import WatermarkLogitsProcessor, NGramHashing, PrevN_ContextCodeExtractor
from evaluations.get_rbr_acc import recover_payload_rbr

def run_sanity_check(
    payload_bits: int = 32,
    num_layers: int = 10,
    n: int = 2,
    seq_len: int = 200,
    vocab_size: int = 1000,  # small vocab for fast testing
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"MCMark-RBR Sanity Check")
    print(f"payload_bits={payload_bits}, num_layers={num_layers}, "
          f"n={n}, seq_len={seq_len}, vocab_size={vocab_size}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    # 1. Build payload
    rng_np = np.random.RandomState(seed)
    payload_bytes = bytes(rng_np.randint(0, 256, (payload_bits + 7) // 8).tolist())
    payload_list = np.unpackbits(
        np.frombuffer(payload_bytes, dtype=np.uint8)
    )[:payload_bits].tolist()
    print(f"Payload (first 8 bits): {payload_list[:8]}")

    # 2. Build WatermarkLogitsProcessor
    import random
    random.seed(42)
    private_key = random.getrandbits(1024).to_bytes(128, "big")

    reweight = MC_RBR_Reweight(num_layers=num_layers, n=n)
    watermark_key_list = [
        NGramHashing(PrevN_ContextCodeExtractor(2), ignore_history=False)
    ]
    wp = WatermarkLogitsProcessor(
        private_key=private_key,
        reweight=reweight,
        watermark_key_list=watermark_key_list,
        payload=payload_bytes,
        payload_bits=payload_bits,
    )

    # 3. Simulate generation with watermarking
    print(f"\nSimulating watermarked generation ({seq_len} tokens)...")
    wp.reset_watermark_key(1)

    # Start with a short "prompt" to initialize context
    prompt_ids = torch.randint(0, vocab_size, (1, 3), dtype=torch.long, device=device)
    generated_ids = prompt_ids.clone()

    for step in range(seq_len):
        # Use random logits to simulate a realistic (non-uniform) distribution
        # This avoids the degenerate case where uniform probs + multi-layer reweighting
        # creates numerical instability
        torch.manual_seed(step + seed * 10000)
        logits = torch.randn(1, vocab_size, device=device)

        # Apply watermark
        watermarked_logits = wp(generated_ids, logits)

        # Sample from watermarked distribution
        probs = torch.softmax(watermarked_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Extract only the output tokens (without prompt)
    output_ids = generated_ids[:, prompt_ids.shape[1]:]
    print(f"Generated {output_ids.shape[1]} tokens")

    # 4. Detect
    print(f"\nRunning detection...")
    wp_for_detect = WatermarkLogitsProcessor(
        private_key=private_key,
        reweight=reweight,
        watermark_key_list=[NGramHashing(PrevN_ContextCodeExtractor(2), ignore_history=False)],
        payload=payload_bytes,
        payload_bits=payload_bits,
    )

    # Use the last 2 prompt tokens as the tail
    prompt_tail = prompt_ids[0, -2:].tolist()

    recovered_bits, debug_info = recover_payload_rbr(
        output_ids=output_ids,
        wp=wp_for_detect,
        vocab_size=vocab_size,
        num_layers=num_layers,
        n=n,
        payload_bits=payload_bits,
        prompt_tail_ids=prompt_tail,
    )

    # 5. Evaluate
    correct = sum(r == g for r, g in zip(recovered_bits, payload_list))
    accuracy = correct / payload_bits

    print(f"\nGround truth (first 8): {payload_list[:8]}")
    print(f"Recovered    (first 8): {recovered_bits[:8]}")
    print(f"\nBit accuracy: {correct}/{payload_bits} = {accuracy:.4f}")

    # Show per-bit vote counts for the first 8 bits
    print(f"\nPer-bit debug (first 8 bits):")
    for i in range(min(8, payload_bits)):
        d = debug_info.get(i, {})
        print(f"  bit[{i}]: gt={payload_list[i]}, pred={recovered_bits[i]}, "
              f"hit0={d.get('hit0',0)}, hit1={d.get('hit1',0)}, "
              f"total={d.get('total',0)}, "
              f"rate0={d.get('rate0', 0):.3f}, rate1={d.get('rate1', 0):.3f}")

    if accuracy >= 0.85:
        print(f"\n✅ PASS: bit accuracy {accuracy:.4f} >= 0.85")
    else:
        print(f"\n❌ FAIL: bit accuracy {accuracy:.4f} < 0.85")
        print("   (Note: with seq_len=200 and small vocab=1000, lower accuracy is expected)")
        print("   Try increasing seq_len or using a real LLM vocab_size=128256")

    return accuracy


if __name__ == "__main__":
    # Quick test with small vocab
    acc_small = run_sanity_check(
        payload_bits=16,
        num_layers=10,
        n=2,
        seq_len=300,
        vocab_size=1000,
    )

    # Harder test: more payload bits
    acc_harder = run_sanity_check(
        payload_bits=32,
        num_layers=10,
        n=2,
        seq_len=512,
        vocab_size=1000,
    )

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  16-bit payload, 300 tokens: {acc_small:.4f}")
    print(f"  32-bit payload, 512 tokens: {acc_harder:.4f}")
    print(f"{'='*60}")
