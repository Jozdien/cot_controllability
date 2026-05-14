#!/bin/bash
# Run R8 pilot: broad exploration of new zero-shot variants
# Uses dev split (35% of samples, ~143) for iteration, holding out rest for confirmation

MODES="lowercase_thinking,word_suppression,uppercase_thinking,end_of_sentence"
N_SAMPLES=100
MODEL="qwen/qwen3-32b"
CONCURRENCY=500
OUTPUT_DIR="results/prompt_ablation_r8_pilot"

# R8 variants (pure zero-shot + prefill for reference)
VARIANTS="baseline,filter_pipe,two_pass,checkpoint_strict,render_mode,think_zone,system_override,whisper_think,different_language,inner_voice,priority_format,adversarial_check,code_function,regex_constraint,template_fill,structured_output,counter_alg,echo_protocol,prefill_start,prefill_selfcheck"

echo "=== R8 Pilot: Broad Zero-Shot Exploration ==="
echo "Model: $MODEL"
echo "Modes: $MODES"
echo "Variants: $(echo $VARIANTS | tr ',' '\n' | wc -l | tr -d ' ') variants"
echo "Samples: $N_SAMPLES per variant×mode"
echo "Total requests: ~$(echo "$VARIANTS" | tr ',' '\n' | wc -l | tr -d ' ') × 4 × $N_SAMPLES"
echo ""

uv run scripts/runs/run_prompt_ablation.py \
    --model "$MODEL" \
    --modes "$MODES" \
    --variants "$VARIANTS" \
    --n-samples "$N_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42
