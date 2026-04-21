#!/usr/bin/env bash
# run_adamc_exp.sh
# Run AdaMC-Watermark text generation experiment.
#
# Usage:
#   bash ./scripts/run_adamc_exp.sh
#
# To run ablation study over entropy_threshold / n_max, change reweight_type
# to 'adamc_ablation'.

# ----- Configuration -------------------------------------------------------
# Models used in the paper:
#   'mistralai/Mistral-7B-Instruct-v0.3'
#   'meta-llama/Llama-3.2-3B-Instruct'
#   'meta-llama/Llama-2-7b-chat-hf'
#   'microsoft/Phi-3.5-mini-instruct'
model_str="meta-llama/Llama-3.2-3B-Instruct"

# Datasets:
#   'mmw_book_report','mmw_story','mmw_fake_news','dolly_cw','longform_qa','finance_qa'
dataset_name="mmw_book_report"

# Experiment type:
#   'adamc'          — single AdaMC run (n_max=32, entropy_threshold=0.5, 32-bit message)
#   'adamc_ablation' — grid over entropy_threshold x n_max
reweight_type="adamc"

# Results will be saved under:
#   ./results/{dataset_name}/{model_name}/{reweight_type}/text_generation.txt
#   ./results/{dataset_name}/{model_name}/{reweight_type}/score.txt
# ---------------------------------------------------------------------------

python -m experiments \
    --res_dir "./results" \
    --model_str "$model_str" \
    --reweight_type "$reweight_type" \
    --dataset_name "$dataset_name"
