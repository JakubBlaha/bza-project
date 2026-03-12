# BZA Tool — ROME Quantization Impact Benchmark

Benchmark how quantization affects retention of facts implanted via [ROME](https://rome.baulab.info/) into LLMs, evaluated on [CounterFact](https://rome.baulab.info/data/dsets/counterfact.json).

## Setup

```bash
# 1. Clone with EasyEdit submodule
git clone --recurse-submodules <repo-url>
cd bza-project

# Or if already cloned:
git submodule update --init --recursive

# 2. Install dependencies
pip install -e .

# 3. Verify EasyEdit is accessible
python -c "import sys; sys.path.insert(0, 'vendor/EasyEdit'); from easyeditor import ROMEHyperParams; print('OK')"
```

## Usage

### Individual Commands

```bash
# Apply ROME edits (saves model + edit metadata)
python -m bza_tool rome-edit \
    --model-config vendor/EasyEdit/hparams/ROME/llama-7b.yaml \
    --output-dir ./outputs/llama-7b/rome \
    --num-edits 100

# Evaluate fact retention
python -m bza_tool evaluate \
    --model-path ./outputs/llama-7b/rome \
    --output-file ./results/llama-7b_rome.json

# Quantize (GPTQ or AWQ)
python -m bza_tool quantize \
    --model-path ./outputs/llama-7b/rome \
    --method gptq --bits 4 \
    --output-dir ./outputs/llama-7b/rome-gptq4

# Evaluate the quantized model
python -m bza_tool evaluate \
    --model-path ./outputs/llama-7b/rome-gptq4 \
    --output-file ./results/llama-7b_rome_gptq4.json
```

### Full Pipelines

```bash
# Scenario 1: Model → ROME → Eval → Quant → Eval
python -m bza_tool pipeline \
    --scenario rome_eval_quant_eval \
    --model-config vendor/EasyEdit/hparams/ROME/llama-7b.yaml \
    --quant-method gptq --bits 4

# Scenario 2: Model → Quant → ROME → Eval
python -m bza_tool pipeline \
    --scenario quant_rome_eval \
    --model-config vendor/EasyEdit/hparams/ROME/llama-7b.yaml \
    --quant-method gptq --bits 4
```

### Run All 13 Models

```bash
./scripts/rome_eval_quant_eval.sh --quant-method gptq --bits 4
./scripts/quant_rome_eval.sh --quant-method gptq --bits 4
```

## Flags

| Flag | Description |
|------|-------------|
| `--fp16` | Run ROME editing in fp16 (default: fp32 for reproducibility) |
| `--num-edits N` | Limit to first N CounterFact edits |
| `--bits {4,8}` | Quantization bit width |
| `--method {awq,gptq}` | Quantization backend |

## Supported Models

All 13 models with EasyEdit ROME configs: baichuan-7b, chatglm2-6b, chatglm4-9b, gpt-j-6B, gpt2-xl, internlm-7b, llama-7b, llama3-8b, llama3.2-3b, mistral-7b, qwen-7b, qwen2-7b, qwen2.5-7b.

## Output Structure

```
outputs/<model>/
├── rome/                     # ROME-edited model + metadata
│   ├── model files...
│   ├── edit_metadata.json    # Which facts were edited
│   └── rome_metrics.json     # EasyEdit metrics at edit time
├── rome-gptq4/               # Quantized model + copied metadata
└── results/
    ├── rome_eval.json         # Post-ROME evaluation
    └── rome_gptq4_eval.json   # Post-quant evaluation
```
