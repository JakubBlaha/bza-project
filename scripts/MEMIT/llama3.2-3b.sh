#!/usr/bin/env bash
set -euo pipefail
cd /workspace/bza-project

uv run python -m bza_tool download meta-llama/Llama-3.2-3B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/llama3.2-3b.yaml --num-edits 500
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/MEMIT/500-gptq2
