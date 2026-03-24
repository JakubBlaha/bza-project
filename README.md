# BZA Tool — ROME Quantization Impact Benchmark

Benchmark how quantization affects retention of facts implanted via [ROME](https://rome.baulab.info/) into LLMs, evaluated on [CounterFact](https://rome.baulab.info/data/dsets/counterfact.json).

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager).

```bash
# 1. Clone with EasyEdit submodule
git clone --recurse-submodules git@github.com:JakubBlaha/bza-project.git
cd bza-project

# Or if already cloned:
git submodule update --init --recursive

# 2. Create venv and install all dependencies
uv sync --no-build-isolation

# 3. Verify EasyEdit is accessible
uv run python -c "import sys; sys.path.insert(0, 'vendor/EasyEdit'); from easyeditor import ROMEHyperParams; print('OK')"
```

## Usage

### Individual Commands

```bash
# Download a model from HuggingFace Hub
uv run python -m bza_tool download gpt2-xl

# Apply AlphaEdit edits (saves model + edit metadata)
uv run python -m bza_tool edit --model-config ./res/hparams/AlphaEdit/gpt2-xl.yaml --output-dir ./outputs/gpt2-xl/alphaedit --num-edits 100

# Evaluate fact retention
uv run python -m bza_tool evaluate --model-path ./outputs/gpt2-xl/alphaedit

# Quantize (gptq, awq, gptaq, qqq, gar, etc.)
uv run python -m bza_tool quantize --model-path ./outputs/gpt2-xl/alphaedit --method gptq --bits 4

# Evaluate the quantized model
uv run python -m bza_tool evaluate --model-path ./outputs/gpt2-xl/alphaedit-gptq4
```

### Full Pipeline

The `run` command executes the full pipeline (edit → evaluate → quantize → evaluate) **in a single process**, avoiding repeated slow imports of torch/transformers on each step.

```bash
# Run AlphaEdit and MEMIT on gpt2-xl with default settings (1000 edits, gptq at 8/4/3/2 bits)
uv run python -m bza_tool run --model gpt2-xl --methods AlphaEdit,MEMIT

# Customize edits, quantization method, and bit widths
uv run python -m bza_tool run --model gpt2-xl --methods AlphaEdit --num-edits 500 --fp16 --quant-method gptq --bits 8 4
```

## Evaluation Status

### AlphaEdit

| Model       | FP32 | GPTQ-8 | GPTQ-4 | GPTQ-3 | GPTQ-2 |
| ----------- | ---- | ------ | ------ | ------ | ------ |
| gpt2-xl     | ✅    | ✅      | ✅      | ✅      | ✅      |
| gpt-j-6B    | ✅    | ✅      | ✅      | ✅      | ✅      |
| llama3.1-8b |      |        |        |        |        |
| llama3-8b   |      |        |        |        |        |
| qwen2.5-7b  | ✅    | ✅      | ✅      | ✅      | ✅      |

### EMMET

| Model       | FP32 | GPTQ-8 | GPTQ-4 | GPTQ-3 | GPTQ-2 |
| ----------- | ---- | ------ | ------ | ------ | ------ |
| gpt2-xl     | ✅    | ✅      | ✅      | ✅      | ✅      |
| gpt-j-6B    |      |        |        |        |        | TODO |
| llama3.2-3b |      |        |        |        |        | TODO |
| llama-7b    |      |        |        |        |        | TODO |

### MEMIT

| Model       | FP32 | GPTQ-8 | GPTQ-4 | GPTQ-3 | GPTQ-2 |
| ----------- | ---- | ------ | ------ | ------ | ------ |
| gpt2-xl     | ✅    | ✅      | ✅      | ✅      | ✅      |
| baichuan-7b |      |        |        |        |        | Can't get access |
| chatglm2-6b |      |        |        |        |        | Incompatible     |
| gpt-j-6B    | ✅    | ✅      | ✅      | ✅      | ✅      |
| internlm-7b | ✅    |        |        |        |        | Incompatible (quantize) |
| llama3.2-3b | ✅    | ✅      | ✅      | ✅      | ✅      |
| llama-7b    | ✅    | ✅      | ✅      | ✅      | ✅      |
| mistral-7b  | ✅    | ✅      | ✅      | ✅      | ✅      |
| qwen2.5-7b  | ✅    | ✅      | ✅      | ✅      | ✅      |
| qwen2-7b    | ✅    | ✅      | ✅      | ✅      | ✅      |
| qwen-7b     |      |        |        |        |        | Incompatible     |

### Model Access

| Model       | HuggingFace Repo                    | Access |
| ----------- | ----------------------------------- | ------ |
| gpt2-xl     | openai-community/gpt2-xl            | Open   |
| gpt-j-6B    | EleutherAI/gpt-j-6B                 | Open   |
| qwen2.5-7b  | Qwen/Qwen2.5-7B                     | Open   |
| qwen2-7b    | Qwen/Qwen2-7B                       | Open   |
| qwen-7b     | Qwen/Qwen-7B                        | Open   |
| mistral-7b  | mistralai/Mistral-7B-v0.1           | Open   |
| baichuan-7b | baichuan-inc/Baichuan-7B            | Open   |
| chatglm2-6b | THUDM/chatglm2-6b                   | Open   |
| internlm-7b | internlm/internlm-7b                | Open   |
| llama3.1-8b | meta-llama/Llama-3.1-8B-Instruct    | Gated  |
| llama3-8b   | meta-llama/Meta-Llama-3-8B-Instruct | Gated  |
| llama3.2-3b | meta-llama/Llama-3.2-3B             | Gated  |
| llama-7b    | huggyllama/llama-7b                 | Open   |

Gated models require accepting the license at the model page on huggingface.co and logging in via `huggingface-cli login`.

## Flags

| Flag                 | Description                                                                            |
| -------------------- | -------------------------------------------------------------------------------------- |
| `--fp16`             | Run ROME editing in fp16 (default: fp32 for reproducibility)                           |
| `--num-edits N`      | Limit to first N CounterFact edits                                                     |
| `--bits {2, 3, 4,8}` | Quantization bit width                                                                 |
| `--method {METHOD}`  | Quantization method: gptq, awq, gptaq, qqq, gar (or any format supported by gptqmodel) |

## Setup on RunPod

```bash
apt update && apt install -y tmux mc libgl1-mesa-glx libglib2.0-0

# Setup SSH key for git
mkdir -p ~/.ssh
cp /workspace/.ssh/id_ed25519 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null

cd /workspace/bza-project
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Make venv persistent in /workspace
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python

uv pip install hatchling editables setuptools
uv sync --no-build-isolation

tmux
```

Then run with `python -m bza_tool` directly (no `uv run` needed).

## SSH Key for Git (RunPod)

### One-time setup

Generate a persistent key in `/workspace` so it survives pod restarts:

```bash
mkdir -p /workspace/.ssh
ssh-keygen -t ed25519 -C "runpod" -f /workspace/.ssh/id_ed25519 -N ""
cat /workspace/.ssh/id_ed25519.pub
```

Copy the public key and add it to GitHub → Settings → SSH and GPG keys → New SSH key.

## Throubleshooting

**Test CUDA:**

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.zeros(1).cuda())"
```

**Run for model:**
```bash
uv run python -m bza_tool run --model gpt-j-6B --methods AlphaEdit
```

**RunPod Image:**

```
vishva123/cuda-12.6-pytorch-2.7.1-runpod
```