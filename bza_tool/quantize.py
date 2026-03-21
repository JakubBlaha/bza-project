"""Quantize a model using GPTQ or AWQ via GPTQModel."""

import logging
import shutil
from pathlib import Path

from bza_tool.utils import setup_logging, ensure_dir, EDIT_META_FILENAME, ensure_model_exists

logger = logging.getLogger(__name__)


def _load_calibration_data(num_samples: int = 256) -> list[str]:
    """Load calibration text data from wikitext-2."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:num_samples]
    logger.info("Loaded %d calibration samples from wikitext-2", len(texts))
    return texts


def _get_quantize_config(method: str, bits: int):
    """Map method name to gptqmodel QuantizeConfig."""
    from gptqmodel import QuantizeConfig

    # Base config for most methods
    config_kwargs = {"bits": bits, "group_size": 128}

    if method == "gptq":
        return QuantizeConfig(**config_kwargs)
    elif method == "awq":
        config_kwargs["format"] = "llm-awq"
        return QuantizeConfig(**config_kwargs)
    elif method == "gptaq":
        from gptqmodel import GPTAQConfig
        config_kwargs["gptaq"] = GPTAQConfig(device="auto")
        return QuantizeConfig(**config_kwargs)
    elif method == "qqq":
        # QQQ often uses specific formats/params, but gptqmodel usually has a default
        # Based on README, QQQ is a top-level method.
        # We assume QuantizeConfig(bits=bits, format="qqq") or similar if supported
        # If gptqmodel doesn't have a specific QQQConfig yet, we pass format
        config_kwargs["format"] = "qqq"
        return QuantizeConfig(**config_kwargs)
    elif method == "gar":
        # Group Aware Reordering (GAR) requires desc_act=False and act_group_aware=True
        config_kwargs["desc_act"] = False
        config_kwargs["act_group_aware"] = True
        return QuantizeConfig(**config_kwargs)
    else:
        # Fallback for other methods gptqmodel might support via format
        logger.info("Using generic QuantizeConfig with format=%s", method)
        config_kwargs["format"] = method
        return QuantizeConfig(**config_kwargs)


def _quantize_model(model_path: str, bits: int, output_dir: Path, method: str) -> None:
    """Generic quantization function using GPTQModel."""
    from gptqmodel import GPTQModel

    logger.info("%s quantization: %s -> %d-bit", method.upper(), model_path, bits)

    # Monkey-patch GPT-J to fix meta tensor bug: embed_positions is a sinusoidal
    # buffer that doesn't get materialized when accelerate uses device_map.
    # Patching __init__ ensures it works even if gptqmodel re-initializes layers.
    import torch
    from transformers.models.gptj import modeling_gptj
    _orig_attn_init = modeling_gptj.GPTJAttention.__init__

    def _patched_attn_init(self, *args, **kwargs):
        _orig_attn_init(self, *args, **kwargs)
        if self.embed_positions.device == torch.device("meta"):
            pos_embd_dim = self.rotary_dim or self.embed_dim
            self.embed_positions = modeling_gptj.create_sinusoidal_positions(
                self.embed_positions.shape[0], pos_embd_dim
            )

    modeling_gptj.GPTJAttention.__init__ = _patched_attn_init

    quant_config = _get_quantize_config(method, bits)
    model = GPTQModel.load(model_path, quant_config, device="cuda:0")

    calibration_data = _load_calibration_data()

    logger.info("Running %s quantization with %d calibration samples...",
                method.upper(), len(calibration_data))
    model.quantize(calibration_data, batch_size=8)

    logger.info("Saving quantized model to %s", output_dir)
    model.save(str(output_dir))


def run_quantize(args) -> None:
    """CLI entry point for the ``quantize`` subcommand."""
    setup_logging()

    ensure_model_exists(args.model_path)
    model_path = Path(args.model_path)
    output_dir_path = model_path.parent / f"{model_path.name}-{args.method}{args.bits}"
    if output_dir_path.exists():
        logger.error("Output directory %s already exists. Please choose a different method or bits.", output_dir_path)
        exit(1)
    output_dir = ensure_dir(output_dir_path)

    _quantize_model(str(model_path), args.bits, output_dir, args.method)

    # Copy edit metadata to the quantized model dir so evaluate can find it
    src_meta = model_path / EDIT_META_FILENAME
    if src_meta.exists():
        shutil.copy2(src_meta, output_dir / EDIT_META_FILENAME)
        logger.info("Copied edit metadata to %s", output_dir)
    else:
        logger.warning("No edit metadata found at %s — evaluate may not work "
                       "on the quantized model without it.", src_meta)

    logger.info("Quantization complete: %s (%s, %d-bit)",
                output_dir, args.method.upper(), args.bits)
