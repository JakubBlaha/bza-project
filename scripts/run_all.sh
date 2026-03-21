#!/usr/bin/env bash
# All remaining edit/evaluate/quantize commands.
# Run from /workspace/bza-project

set -euo pipefail
cd /workspace/bza-project

# ============================================================
# AlphaEdit — gpt-j-6B
# ============================================================
uv run python -m bza_tool download EleutherAI/gpt-j-6B
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/gpt-j-6B.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/1000
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/AlphaEdit/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/AlphaEdit/1000-gptq2

# ============================================================
# AlphaEdit — llama3.1-8b
# ============================================================
uv run python -m bza_tool download meta-llama/Llama-3.1-8B-Instruct
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/llama3.1-8b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.1-8b/AlphaEdit/1000
uv run python -m bza_tool quantize --model-path ./outputs/llama3.1-8b/AlphaEdit/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.1-8b/AlphaEdit/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama3.1-8b/AlphaEdit/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.1-8b/AlphaEdit/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama3.1-8b/AlphaEdit/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.1-8b/AlphaEdit/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama3.1-8b/AlphaEdit/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.1-8b/AlphaEdit/1000-gptq2

# ============================================================
# AlphaEdit — llama3-8b
# ============================================================
uv run python -m bza_tool download meta-llama/Llama-3-8B-Instruct
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/llama3-8b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama3-8b/AlphaEdit/1000
uv run python -m bza_tool quantize --model-path ./outputs/llama3-8b/AlphaEdit/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama3-8b/AlphaEdit/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama3-8b/AlphaEdit/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama3-8b/AlphaEdit/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama3-8b/AlphaEdit/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama3-8b/AlphaEdit/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama3-8b/AlphaEdit/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama3-8b/AlphaEdit/1000-gptq2

# ============================================================
# AlphaEdit — qwen2.5-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen2.5-7B
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/qwen2.5-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/AlphaEdit/1000-gptq2

# ============================================================
# EMMET — gpt-j-6B
# ============================================================
uv run python -m bza_tool download EleutherAI/gpt-j-6B
uv run python -m bza_tool edit --method EMMET --model-config ./res/hparams/EMMET/gpt-j-6B.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/1000
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/1000-gptq2

# ============================================================
# EMMET — llama3.2-3b
# ============================================================
uv run python -m bza_tool download meta-llama/Llama-3.2-3B
uv run python -m bza_tool edit --method EMMET --model-config ./res/hparams/EMMET/llama3.2-3b.yaml
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/EMMET/1000
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/EMMET/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/EMMET/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/EMMET/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/EMMET/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/EMMET/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/EMMET/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/EMMET/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/EMMET/1000-gptq2

# ============================================================
# EMMET — llama-7b
# ============================================================
uv run python -m bza_tool download huggyllama/llama-7b
uv run python -m bza_tool edit --method EMMET --model-config ./res/hparams/EMMET/llama-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/1000
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/EMMET/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/EMMET/1000-gptq2

# ============================================================
# MEMIT — baichuan-7b
# ============================================================
uv run python -m bza_tool download baichuan-inc/Baichuan-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/baichuan-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/baichuan-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/baichuan-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/baichuan-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/baichuan-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/baichuan-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/baichuan-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/baichuan-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/baichuan-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/baichuan-7b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — chatglm2-6b
# ============================================================
uv run python -m bza_tool download THUDM/chatglm2-6b
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/chatglm2-6b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — gpt-j-6B
# ============================================================
uv run python -m bza_tool download EleutherAI/gpt-j-6B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/gpt-j-6B.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/1000-gptq2

# ============================================================
# MEMIT — internlm-7b
# ============================================================
uv run python -m bza_tool download internlm/internlm-7b
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/internlm-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — llama3.2-3b
# ============================================================
uv run python -m bza_tool download meta-llama/Llama-3.2-3B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/llama3.2-3b.yaml
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama3.2-3b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama3.2-3b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — llama-7b
# ============================================================
uv run python -m bza_tool download huggyllama/llama-7b
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/llama-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — mistral-7b
# ============================================================
uv run python -m bza_tool download mistralai/Mistral-7B-v0.1
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/mistral-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/mistral-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/mistral-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/mistral-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/mistral-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/mistral-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/mistral-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/mistral-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/mistral-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/mistral-7b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — qwen2.5-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen2.5-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/qwen2.5-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/qwen2.5-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2.5-7b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — qwen2-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen2-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/qwen2-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/qwen2-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/qwen2-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/qwen2-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/qwen2-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/qwen2-7b/MEMIT/1000-gptq2

# ============================================================
# MEMIT — qwen-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/qwen-7b.yaml --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/1000
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/1000 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/1000-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/1000 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/1000-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/1000 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/1000-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/1000 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/1000-gptq2
