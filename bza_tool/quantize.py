"""Quantize a model using GPTQ or AWQ."""

import logging
import shutil
from pathlib import Path

from bza_tool.utils import setup_logging, ensure_dir, EDIT_META_FILENAME

logger = logging.getLogger(__name__)


def _quantize_gptq(model_path: str, bits: int, output_dir: Path) -> None:
    """Quantize using AutoGPTQ."""
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
    from datasets import load_dataset

    logger.info("GPTQ quantization: %s -> %d-bit", model_path, bits)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
    )

    # Calibration data from wikitext-2
    calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [t for t in calib_dataset["text"] if len(t.strip()) > 50][:256]
    calib_encodings = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        for text in calib_texts
    ]

    logger.info("Running GPTQ quantization with %d calibration samples...",
                len(calib_encodings))
    model.quantize(calib_encodings)

    logger.info("Saving quantized model to %s", output_dir)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _quantize_awq(model_path: str, bits: int, output_dir: Path) -> None:
    """Quantize using llm-compressor (AWQ).

    Uses the vLLM llm-compressor library which is the maintained successor
    to the deprecated AutoAWQ package.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.modifiers.quantization import QuantizationModifier  # type: ignore
    from llmcompressor import oneshot  # type: ignore
    from datasets import load_dataset

    logger.info("AWQ quantization (llm-compressor): %s -> %d-bit", model_path, bits)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Calibration data
    calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [t for t in calib_dataset["text"] if len(t.strip()) > 50][:256]

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    calib_ds = calib_dataset.filter(lambda x: len(x["text"].strip()) > 50).select(range(256))

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=f"W{bits}A16",
        ignore=["lm_head"],
    )

    logger.info("Running AWQ quantization with llm-compressor...")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        output_dir=str(output_dir),
        num_calibration_samples=256,
        max_seq_length=512,
    )

    tokenizer.save_pretrained(str(output_dir))
    logger.info("AWQ quantization complete: %s", output_dir)


def run_quantize(args) -> None:
    """CLI entry point for the ``quantize`` subcommand."""
    setup_logging()

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
