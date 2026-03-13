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


def _quantize_gptq(model_path: str, bits: int, output_dir: Path) -> None:
    """Quantize using GPTQ via GPTQModel."""
    from gptqmodel import GPTQModel, QuantizeConfig

    logger.info("GPTQ quantization: %s -> %d-bit", model_path, bits)

    quant_config = QuantizeConfig(bits=bits, group_size=128)
    model = GPTQModel.load(model_path, quant_config)

    calibration_data = _load_calibration_data()

    logger.info("Running GPTQ quantization with %d calibration samples...",
                len(calibration_data))
    model.quantize(calibration_data)

    logger.info("Saving quantized model to %s", output_dir)
    model.save(str(output_dir))


def _quantize_awq(model_path: str, bits: int, output_dir: Path) -> None:
    """Quantize using AWQ via GPTQModel."""
    from gptqmodel import GPTQModel, QuantizeConfig

    logger.info("AWQ quantization: %s -> %d-bit", model_path, bits)

    quant_config = QuantizeConfig(bits=bits, group_size=128, format="awq")
    model = GPTQModel.load(model_path, quant_config)

    calibration_data = _load_calibration_data()

    logger.info("Running AWQ quantization with %d calibration samples...",
                len(calibration_data))
    model.quantize(calibration_data)

    logger.info("Saving quantized model to %s", output_dir)
    model.save(str(output_dir))


def run_quantize(args) -> None:
    """CLI entry point for the ``quantize`` subcommand."""
    setup_logging()

    ensure_model_exists(args.model_path)
    model_path = Path(args.model_path)
    output_dir = ensure_dir(Path(args.output_dir))

    if args.method == "gptq":
        _quantize_gptq(str(model_path), args.bits, output_dir)
    elif args.method == "awq":
        _quantize_awq(str(model_path), args.bits, output_dir)
    else:
        raise ValueError(f"Unknown quantization method: {args.method}")

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
