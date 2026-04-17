# On the Persistence of Knowledge Editing Attacks Under Post-Training Quantization

Benchmark evaluating how GPTQ quantization (8, 4, 3, and 2-bit) affects the retention of facts injected into LLMs via knowledge editing methods (MEMIT, EMMET, AlphaEdit), evaluated on the CounterFact dataset.

## Results

See `docs/docs.pdf` for the full paper, including results, figures, and analysis.

## EasyEdit

This project uses a modified version of EasyEdit to resolve compatibility issues with current model architectures and library versions. The fork is available at https://github.com/JakubBlaha/EasyEdit (main branch) and is included as a git submodule under `vendor/EasyEdit`.

## Local Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone with EasyEdit submodule
git clone --recurse-submodules git@github.com:JakubBlaha/bza-project.git
cd bza-project

# Or if already cloned:
git submodule update --init --recursive

# Install dependencies
uv sync --no-build-isolation
```

## Reproducing on RunPod

### RunPod Image

Use the following image — this is the CUDA 12.6 version required for compatibility with the project dependencies:

```
vishva123/cuda-12.6-pytorch-2.7.1-runpod
```

### Network Volume

Create a persistent network volume in RunPod and mount it at `/workspace`. This ensures model weights, outputs, and the SSH key survive pod restarts without needing to be re-downloaded or re-generated.

### One-Time SSH Key Setup

Generate a persistent SSH key in `/workspace` and add it to GitHub so the repo can be cloned on any pod restart:

```bash
mkdir -p /workspace/.ssh
ssh-keygen -t ed25519 -C "runpod" -f /workspace/.ssh/id_ed25519 -N ""
cat /workspace/.ssh/id_ed25519.pub
```

Copy the output and add it to GitHub → Settings → SSH and GPG keys → New SSH key.

### Pod Setup (run on each pod start)

```bash
apt update && apt install -y tmux mc libgl1-mesa-glx libglib2.0-0

# Load SSH key from persistent volume
mkdir -p ~/.ssh
cp /workspace/.ssh/id_ed25519 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null

# Clone repo (first time) or pull updates
cd /workspace
git clone --recurse-submodules git@github.com:JakubBlaha/bza-project.git
cd bza-project

# Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Keep venv and cache on the persistent volume
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python

uv pip install hatchling editables setuptools
uv sync --no-build-isolation

tmux
```

Then run experiments with `python -m bza_tool` directly (no `uv run` needed inside the activated venv).

## CLI Reference

### Individual Commands

```bash
# Download a model from HuggingFace Hub
uv run python -m bza_tool download gpt2-xl

# Apply edits (saves edited model + metadata)
uv run python -m bza_tool edit --model-config ./res/hparams/AlphaEdit/gpt2-xl.yaml --output-dir ./outputs/gpt2-xl/alphaedit --num-edits 500 --method AlphaEdit

# Evaluate fact retention
uv run python -m bza_tool evaluate --model-path ./outputs/gpt2-xl/alphaedit

# Quantize
uv run python -m bza_tool quantize --model-path ./outputs/gpt2-xl/alphaedit --method gptq --bits 4

# Evaluate the quantized model
uv run python -m bza_tool evaluate --model-path ./outputs/gpt2-xl/alphaedit-gptq4
```

### Full Pipeline

The `run` command executes the full pipeline (edit → evaluate → quantize → evaluate) in a single process, avoiding repeated slow imports of torch/transformers on each step.

```bash
# Run AlphaEdit and MEMIT on gpt2-xl (500 edits, gptq at 8/4/3/2 bits)
uv run python -m bza_tool run --model gpt2-xl --methods AlphaEdit,MEMIT --num-edits 500 --fp16 --quant-method gptq --bits 8 4 3 2
```

### Flags

| Flag                  | Description                                                                            |
| --------------------- | -------------------------------------------------------------------------------------- |
| `--fp16`              | Run editing in fp16 (default: fp32)                                                    |
| `--num-edits N`       | Limit to first N CounterFact edits                                                     |
| `--bits {2, 3, 4, 8}` | Quantization bit width                                                                 |
| `--quant-method`      | Quantization method: gptq, awq, gptaq, qqq, gar (or any format supported by gptqmodel) |
