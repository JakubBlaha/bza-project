#!/usr/bin/env bash
# Scenario 1: Model → ROME → Eval → Quant → Eval
# Runs for all 13 models supported by EasyEdit ROME.
#
# Usage: ./scripts/rome_eval_quant_eval.sh [--quant-method gptq] [--bits 4] [--num-edits N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HPARAMS_DIR="$PROJECT_DIR/vendor/EasyEdit/hparams/ROME"

QUANT_METHOD="${QUANT_METHOD:-gptq}"
BITS="${BITS:-4}"
NUM_EDITS="${NUM_EDITS:-}"
FP16="${FP16:-}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_DIR/outputs}"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant-method) QUANT_METHOD="$2"; shift 2 ;;
        --bits)         BITS="$2";         shift 2 ;;
        --num-edits)    NUM_EDITS="$2";    shift 2 ;;
        --fp16)         FP16="--fp16";     shift   ;;
        --output-base)  OUTPUT_BASE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS=(
    baichuan-7b
    chatglm2-6b
    chatglm4-9b
    gpt-j-6B
    gpt2-xl
    internlm-7b
    llama-7b
    llama3-8b
    llama3.2-3b
    mistral-7b
    qwen-7b
    qwen2-7b
    qwen2.5-7b
)

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Running rome_eval_quant_eval for: $model"
    echo "========================================"

    CMD=(uv run python -m bza_tool pipeline
        --scenario rome_eval_quant_eval
        --model-config "$HPARAMS_DIR/$model.yaml"
        --quant-method "$QUANT_METHOD"
        --bits "$BITS"
        --output-base "$OUTPUT_BASE"
    )

    [[ -n "$NUM_EDITS" ]] && CMD+=(--num-edits "$NUM_EDITS")
    [[ -n "$FP16" ]]      && CMD+=("$FP16")

    "${CMD[@]}"

    echo "Done: $model"
    echo ""
done

echo "All models complete. Results in: $OUTPUT_BASE"
