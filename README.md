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
# Apply AlphaEdit edits (saves model + edit metadata)
uv run python -m bza_tool edit \
    --model-config vendor/EasyEdit/hparams/AlphaEdit/gpt2-xl.yaml \
    --output-dir ./outputs/gpt2-xl/alphaedit \
    --num-edits 100

# Evaluate fact retention
uv run python -m bza_tool evaluate \
    --model-path ./outputs/gpt2-xl/alphaedit

# Quantize (gptq, awq, gptaq, qqq, gar, etc.)
uv run python -m bza_tool quantize \
    --model-path ./outputs/gpt2-xl/alphaedit \
    --method gptaq --bits 4 \
    --output-dir ./outputs/gpt2-xl/alphaedit-gptq4

# Evaluate the quantized model
uv run python -m bza_tool evaluate \
    --model-path ./outputs/gpt2-xl/alphaedit-gptq4
```

### Full Pipelines

```bash
# Scenario 1: Model → AlphaEdit → Eval → Quant → Eval
uv run python -m bza_tool pipeline \
    --scenario rome_eval_quant_eval \
    --model-config vendor/EasyEdit/hparams/AlphaEdit/llama-7b.yaml \
    --quant-method gptq --bits 4

# Scenario 2: Model → Quant → AlphaEdit → Eval
uv run python -m bza_tool pipeline \
    --scenario quant_rome_eval \
    --model-config vendor/EasyEdit/hparams/AlphaEdit/llama-7b.yaml \
    --quant-method gptq --bits 4
```

## Flags

| Flag                  | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `--fp16`              | Run ROME editing in fp16 (default: fp32 for reproducibility) |
| `--num-edits N`       | Limit to first N CounterFact edits                           |
| `--bits {4,8}`        | Quantization bit width                                       |
| `--method {METHOD}`   | Quantization method: gptq, awq, gptaq, qqq, gar (or any format supported by gptqmodel) |
