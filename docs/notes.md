# Changes & Fixes

## Earlier fixes (EasyEdit fork, custom-changes branch)

### AlphaEdit — get_cov returned tensor on CPU breaking downstream
`get_cov` called `.cpu()` on the covariance before returning, but callers expected it on GPU. Removed the `.cpu()` call.

### AlphaEdit — cache_c dimension mismatch for gpt-j-6B
The `cache_c` initialization only checked for `llama`/`qwen`/`gpt2-xl` model names. gpt-j-6B uses the same weight layout as llama (`shape[1]`) but wasn't listed. Added `gpt-j-6b` to the condition.

### AlphaEdit — added progress logging for target vector computation
The target vector loop (`compute_z`) gave no output, making it look stuck on large edit counts. Added a progress print.

### edit.py — batch_size override
AlphaEdit default `batch_size=1` processes edits one at a time (very slow). Added override to set `batch_size = num_edits` so all edits are processed in one pass.

### edit.py — memory leak after editing
The edited model and editor were not freed from GPU memory after saving, causing OOM in downstream evaluate/quantize steps. Added explicit `del`, `gc.collect()`, and `torch.cuda.empty_cache()`.

### EasyEdit submodule remote
Changed submodule URL from SSH to HTTPS so it works without SSH keys configured.

### Python version constraint
Relaxed `requires-python` from `==3.11.15` to `>=3.11` for broader compatibility.

## Current session fixes

### evaluate.py — Result filename missing model name
Result JSON files were named `{method}_{num_edits}.json` (e.g. `AlphaEdit_1000.json`), which would collide across models. Fixed to include the model name: `{model}_{method}_{num_edits}.json` (e.g. `gpt2-xl_AlphaEdit_1000.json`). Existing results were renamed accordingly.

### edit.py — fp16 flag crashing non-ROME methods
`_patch_hparams` injected `fp16` into the YAML config, but only ROME's hparams class accepts it. AlphaEdit/MEMIT/EMMET crashed with `unexpected keyword argument 'fp16'`. Fixed by setting `hparams.fp16` as an attribute on the object after construction instead of putting it in the YAML.

### editor.py — fp16 uses float16 instead of bfloat16
The editor hardcoded `torch.float16` for the `--fp16` flag, but newer models (Qwen2.5, Llama 3.x, Mistral) store weights in bfloat16. Loading BF16 weights as FP16 can overflow. Changed to `torch.bfloat16` in both `editor.py` and `evaluate.py`.

### AlphaEdit — null space projection matrix (P) not model-specific
All AlphaEdit hparams pointed to the same `./null_space_project.pt`. Different models have different MLP dimensions, so reusing one P caused shape mismatches. Fixed by giving each model its own `P_loc` path (e.g. `./data/null_space_project_qwen2.5-7b.pt`).

### AlphaEdit — P computation crashed for unlisted models
The P initialization code had hardcoded model name checks (`if "llama" in ...`, `elif "gpt2-xl" in ...`) to determine matrix dimensions. Any unlisted model (e.g. Qwen) would crash. Replaced with generic code that lets `get_project` determine dimensions and stacks results with `torch.stack`.

### AlphaEdit — SVD computed on CPU
`get_project` ran `torch.linalg.svd` on CPU, taking 30-60+ minutes per layer for large models (18944x18944 matrix). Moved the covariance matrix to GPU before SVD, reducing time to ~5 minutes per layer.

### AlphaEdit — dtype mismatch in solve with bf16 models
When running with `--fp16` (now bfloat16), the `P` matrix (float32) was multiplied with `layer_ks` (bfloat16), causing a dtype error. Fixed by casting all tensors to float32 in the `torch.linalg.solve` call.
