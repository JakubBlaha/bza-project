#!/usr/bin/env bash
set -euo pipefail
cd /workspace/bza-project

uv run python -m bza_tool download huggyllama/llama-7b
uv run python -m bza_tool edit --method EMMET --model-config ./res/hparams/EMMET/llama-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/500
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/500-gptq2
