#!/usr/bin/env bash
# run_adamc_eval.sh
# Evaluate AdaMC-Watermark results: zero-bit detection + multi-bit extraction.
#
# Usage:
#   bash ./scripts/run_adamc_eval.sh
#
# Prerequisite: run_adamc_exp.sh must have completed first.

# ----- Configuration -------------------------------------------------------
model_name="Llama_3.2_3B_Instruct"   # matches directory name under results/
dataset_name="mmw_book_report"
reweight_type="adamc"

score_path="./results/${dataset_name}/${model_name}/${reweight_type}/score.txt"

# FPR threshold for TPR calculation
fpr=0.001

# Ground-truth message (hex).
# This must match the message embedded during generation.
# Default demo: hashlib.sha256(b"demo_user_42").digest()[:4].hex()
import_python=$(python -c "
import hashlib
print(hashlib.sha256(b'demo_user_42').digest()[:4].hex())
")
message_hex=$(python -c "import hashlib; print(hashlib.sha256(b'demo_user_42').digest()[:4].hex())")

echo "========================================================"
echo "Evaluating AdaMC watermark results"
echo "  score_path  : $score_path"
echo "  fpr_thres   : $fpr"
echo "  message_hex : $message_hex"
echo "========================================================"

# ---------------------------------------------------------------------------
# 1. AdaMC-specific evaluation (zero-bit + multi-bit)
python ./evaluations/get_adamc_acc.py \
    --score_path "$score_path" \
    --fpr_thres  "$fpr" \
    --message_hex "$message_hex"

# ---------------------------------------------------------------------------
# 2. For ablation study — change score_path to adamc_ablation
# score_path_ablation="./results/${dataset_name}/${model_name}/adamc_ablation/score.txt"
# python ./evaluations/get_adamc_acc.py \
#     --score_path "$score_path_ablation" \
#     --fpr_thres  "$fpr" \
#     --message_hex "$message_hex"
