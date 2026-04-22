# score path for main experiments or robustness experiments
# example: ./results/mmw_book_report/Llama_3.2_3B_Instruct/main_exp/score.txt
score_path='./results/mmw_book_report/Llama_3.2_3B_Instruct/adamc/score.txt'
fpr=0.001

# Detect experiment type from path and call the right evaluator
if echo "$score_path" | grep -q "adamc"; then
    echo "Detected AdaMC score file — running AdaMC evaluator"
    message_hex=$(python -c "import hashlib; print(hashlib.sha256(b'demo_user_42').digest()[:4].hex())")
    python ./evaluations/get_adamc_acc.py \
        --score_path "$score_path" \
        --fpr_thres  "$fpr" \
        --message_hex "$message_hex"
else
    echo "Running baseline + MCMark evaluators"
    # for baselines
    python ./evaluations/get_baselines_acc.py \
        --fpr_thres $fpr \
        --score_path $score_path

    # for MC-mark:
    python ./evaluations/get_mcmark_acc.py \
        --fpr_thres $fpr \
        --score_path $score_path
fi