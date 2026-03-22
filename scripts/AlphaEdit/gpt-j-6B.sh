#!/usr/bin/env bash
set -euo pipefail
cd /workspace/bza-project

uv run python -m bza_tool download EleutherAI/gpt-j-6B
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/gpt-j-6B.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/500
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/500-gptq2
