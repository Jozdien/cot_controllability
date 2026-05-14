#!/bin/bash
# Run GPT-OSS replication: same variants as Qwen3 final comparison
# Both 20B and 120B, all 8 CoTControl modes

MODES="lowercase_thinking,word_suppression,uppercase_thinking,alternating_case,repeat_sentences,end_of_sentence,meow_between_words,multiple_word_suppression"
N_SAMPLES=200
CONCURRENCY=500

# Same variants as Qwen3 final comparison:
# baseline, best zero-shot (self_check), combos (render+selfcheck, bounty+selfcheck),
# calibration (cal_2shot), plus a few more zero-shot for exploration
VARIANTS="baseline,self_check,render_mode,combo_render_selfcheck,combo_bounty_selfcheck,cal_2shot"

for MODEL in "openai/gpt-oss-20b" "openai/gpt-oss-120b"; do
    SHORT=$(echo "$MODEL" | sed 's|openai/||')
    OUTPUT_DIR="results/gptoss_${SHORT}"

    echo "=== GPT-OSS Replication: $MODEL ==="
    echo "Modes: $MODES"
    echo "Variants: $VARIANTS"
    echo "Samples: $N_SAMPLES per variant×mode"
    echo ""

    uv run scripts/runs/run_prompt_ablation.py \
        --model "$MODEL" \
        --modes "$MODES" \
        --variants "$VARIANTS" \
        --n-samples "$N_SAMPLES" \
        --concurrency "$CONCURRENCY" \
        --output-dir "$OUTPUT_DIR" \
        --seed 42

    echo ""
    echo "=== Done: $MODEL ==="
    echo ""
done
