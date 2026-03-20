#!/usr/bin/env bash
# Edit a model with specified methods, evaluate, quantize at all bit widths, and evaluate each.
#
# Usage:
#   ./scripts/run_model.sh --model gpt2-xl --methods AlphaEdit,MEMIT
#   ./scripts/run_model.sh --model gpt2-xl --methods AlphaEdit,MEMIT,EMMET --num-edits 1000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HPARAMS_DIR="$PROJECT_DIR/res/hparams"

MODEL=""
METHODS=""
NUM_EDITS=""
FP16=""
QUANT_METHOD="gptq"
BITS_LIST=(8 4 3 2)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --methods)      METHODS="$2";      shift 2 ;;
        --num-edits)    NUM_EDITS="$2";    shift 2 ;;
        --fp16)         FP16="--fp16";     shift   ;;
        --quant-method) QUANT_METHOD="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

if [[ -z "$METHODS" ]]; then
    echo "Error: --methods is required (comma-separated, e.g. AlphaEdit,MEMIT,EMMET)" >&2
    exit 1
fi

cd "$PROJECT_DIR"

IFS=',' read -ra METHOD_ARR <<< "$METHODS"

for method in "${METHOD_ARR[@]}"; do
    HPARAMS_FILE="$HPARAMS_DIR/$method/$MODEL.yaml"

    if [[ ! -f "$HPARAMS_FILE" ]]; then
        echo "ERROR: hparams file not found: $HPARAMS_FILE" >&2
        echo "Skipping method $method for model $MODEL" >&2
        continue
    fi

    echo "========================================"
    echo "Method: $method | Model: $MODEL"
    echo "========================================"

    # --- Edit ---
    EDIT_CMD=(uv run python -m bza_tool edit
        --method "$method"
        --model-config "$HPARAMS_FILE"
    )
    [[ -n "$NUM_EDITS" ]] && EDIT_CMD+=(--num-edits "$NUM_EDITS")
    [[ -n "$FP16" ]]      && EDIT_CMD+=("$FP16")

    echo ">>> Editing..."
    "${EDIT_CMD[@]}"

    # Determine the output directory that was created
    if [[ -n "$NUM_EDITS" ]]; then
        EDIT_DIR="./outputs/$MODEL/$method/$NUM_EDITS"
    else
        # Find the directory created (all counterfact edits)
        EDIT_DIR=$(find "./outputs/$MODEL/$method" -mindepth 1 -maxdepth 1 -type d | head -1)
    fi

    echo ">>> Evaluating base edited model: $EDIT_DIR"
    uv run python -m bza_tool evaluate --model-path "$EDIT_DIR"

    # --- Quantize & evaluate at each bit width ---
    for bits in "${BITS_LIST[@]}"; do
        QUANT_DIR="${EDIT_DIR}-${QUANT_METHOD}${bits}"

        echo ">>> Quantizing: ${QUANT_METHOD}${bits}"
        uv run python -m bza_tool quantize \
            --model-path "$EDIT_DIR" \
            --method "$QUANT_METHOD" \
            --bits "$bits"

        echo ">>> Evaluating: $QUANT_DIR"
        uv run python -m bza_tool evaluate --model-path "$QUANT_DIR"
    done

    echo ""
    echo "Done: $method / $MODEL"
    echo ""
done

echo "========================================"
echo "All methods complete for model: $MODEL"
echo "========================================"
