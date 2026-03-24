#!/usr/bin/env bash
# All remaining edit/evaluate/quantize commands.
# Run from /workspace/bza-project

set -euo pipefail
cd /workspace/bza-project

# ============================================================
# AlphaEdit — gpt-j-6B
# ============================================================
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

# ============================================================
# AlphaEdit — llama3.1-8b
# ============================================================
uv run python -m bza_tool download meta-llama/Llama-3.1-8B-Instruct
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/llama3.1-8b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500
uv run python -m bza_tool quantize --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3.1-8b-instruct/AlphaEdit/500-gptq2

# ============================================================
# AlphaEdit — llama3-8b
# ============================================================
uv run python -m bza_tool download meta-llama/Meta-Llama-3-8B-Instruct
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/llama3-8b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500
uv run python -m bza_tool quantize --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama-3-8b-instruct/AlphaEdit/500-gptq2

# ============================================================
# AlphaEdit — qwen2.5-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen2.5-7B
uv run python -m bza_tool edit --method AlphaEdit --model-config ./res/hparams/AlphaEdit/qwen2.5-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B-Instruct/AlphaEdit/500-gptq2

# ============================================================
# EMMET — gpt-j-6B
# ============================================================
uv run python -m bza_tool download EleutherAI/gpt-j-6B
uv run python -m bza_tool edit --method EMMET --model-config ./res/hparams/EMMET/gpt-j-6B.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/500
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/EMMET/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/EMMET/500-gptq2

# ============================================================
# EMMET — llama3.2-3b
# ============================================================
uv run python -m bza_tool download meta-llama/Llama-3.2-3B
uv run python -m bza_tool edit --method EMMET --model-config ./res/hparams/EMMET/llama3.2-3b.yaml --num-edits 500
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/EMMET/500
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/EMMET/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/EMMET/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/EMMET/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/EMMET/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/EMMET/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/EMMET/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Llama-3.2-3B/EMMET/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Llama-3.2-3B/EMMET/500-gptq2

# ============================================================
# EMMET — llama-7b
# ============================================================
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

# ============================================================
# MEMIT — baichuan-7b
# ============================================================
uv run python -m bza_tool download baichuan-inc/Baichuan-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/baichuan-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/Baichuan-7B/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/Baichuan-7B/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Baichuan-7B/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Baichuan-7B/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Baichuan-7B/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Baichuan-7B/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Baichuan-7B/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Baichuan-7B/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Baichuan-7B/MEMIT/500-gptq2

# ============================================================
# MEMIT — chatglm2-6b
# ============================================================
uv run python -m bza_tool download THUDM/chatglm2-6b
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/chatglm2-6b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/chatglm2-6b/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/chatglm2-6b/MEMIT/500-gptq2

# ============================================================
# MEMIT — gpt-j-6B
# ============================================================
uv run python -m bza_tool download EleutherAI/gpt-j-6B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/gpt-j-6B.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/gpt-j-6B/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/gpt-j-6B/MEMIT/500-gptq2

# ============================================================
# MEMIT — internlm-7b
# ============================================================
uv run python -m bza_tool download internlm/internlm-7b
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/internlm-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/internlm-7b/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/internlm-7b/MEMIT/500-gptq2

# ============================================================
# MEMIT — llama3.2-3b
# ============================================================
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

# ============================================================
# MEMIT — llama-7b
# ============================================================
uv run python -m bza_tool download huggyllama/llama-7b
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/llama-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/llama-7b/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/llama-7b/MEMIT/500-gptq2

# ============================================================
# MEMIT — mistral-7b
# ============================================================
uv run python -m bza_tool download mistralai/Mistral-7B-v0.1
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/mistral-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Mistral-7B-v0.1/MEMIT/500-gptq2

# ============================================================
# MEMIT — qwen2.5-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen2.5-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/qwen2.5-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2.5-7B/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2.5-7B/MEMIT/500-gptq2

# ============================================================
# MEMIT — qwen2-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen2-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/qwen2-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2-7B/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2-7B/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2-7B/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2-7B/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2-7B/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2-7B/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2-7B/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/Qwen2-7B/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/Qwen2-7B/MEMIT/500-gptq2

# ============================================================
# MEMIT — qwen-7b
# ============================================================
uv run python -m bza_tool download Qwen/Qwen-7B
uv run python -m bza_tool edit --method MEMIT --model-config ./res/hparams/MEMIT/qwen-7b.yaml --num-edits 500 --fp16
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/500
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/500 --method gptq --bits 8
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/500-gptq8
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/500 --method gptq --bits 4
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/500-gptq4
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/500 --method gptq --bits 3
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/500-gptq3
uv run python -m bza_tool quantize --model-path ./outputs/qwen-7b/MEMIT/500 --method gptq --bits 2
uv run python -m bza_tool evaluate --model-path ./outputs/qwen-7b/MEMIT/500-gptq2
