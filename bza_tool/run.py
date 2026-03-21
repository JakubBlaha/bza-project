"""Full pipeline: edit → evaluate → quantize → evaluate, all in one process.

Replaces the shell script ``scripts/run_model.sh`` so that heavy imports
(torch, transformers, EasyEdit, …) are loaded only once instead of once
per subprocess invocation.
"""

import argparse
import logging
import shutil
from pathlib import Path

from bza_tool.utils import (
    PROJECT_ROOT,
    ensure_dir,
    ensure_easyedit_on_path,
    ensure_model_exists,
    load_edit_metadata,
    save_edit_metadata,
    setup_logging,
    EDIT_META_FILENAME,
)

logger = logging.getLogger(__name__)


def _run_edit(method: str, model_config: str, num_edits: int | None, fp16: bool) -> Path:
    """Run the edit step, return the output directory."""
    args = argparse.Namespace(
        method=method,
        model_config=model_config,
        output_dir=None,
        num_edits=num_edits,
        fp16=fp16,
    )
    from bza_tool.edit import run_edit

    run_edit(args)

    # Reconstruct the output path the same way edit.py does
    import yaml

    with open(model_config) as f:
        cfg = yaml.safe_load(f)
    model_basename = Path(cfg["model_name"]).name

    from bza_tool.utils import load_counterfact

    num_facts = num_edits if num_edits is not None else len(load_counterfact())
    return Path("./outputs") / model_basename / method / str(num_facts)


def _run_evaluate(model_path: Path) -> None:
    """Run the evaluate step on a model directory."""
    args = argparse.Namespace(model_path=str(model_path))
    from bza_tool.evaluate import run_evaluate

    run_evaluate(args)


def _run_quantize(model_path: Path, method: str, bits: int) -> Path:
    """Run the quantize step, return the quantized model directory."""
    args = argparse.Namespace(
        model_path=str(model_path),
        method=method,
        bits=bits,
    )
    from bza_tool.quantize import run_quantize

    run_quantize(args)
    return model_path.parent / f"{model_path.name}-{method}{bits}"


def run_pipeline(args: argparse.Namespace) -> None:
    """CLI entry point for the ``run`` subcommand."""
    setup_logging()
    ensure_easyedit_on_path()

    model = args.model
    methods = [m.strip() for m in args.methods.split(",")]
    num_edits = args.num_edits
    fp16 = args.fp16
    quant_method = args.quant_method
    bits_list = args.bits

    hparams_dir = PROJECT_ROOT / "res" / "hparams"

    for method in methods:
        hparams_file = hparams_dir / method / f"{model}.yaml"
        if not hparams_file.exists():
            logger.error("Hparams file not found: %s — skipping %s", hparams_file, method)
            continue

        logger.info("=" * 40)
        logger.info("Method: %s | Model: %s", method, model)
        logger.info("=" * 40)

        # ── Edit ──────────────────────────────────────────────────────
        logger.info(">>> Editing...")
        edit_dir = _run_edit(method, str(hparams_file), num_edits, fp16)

        # ── Evaluate base edited model ────────────────────────────────
        logger.info(">>> Evaluating base edited model: %s", edit_dir)
        _run_evaluate(edit_dir)

        # ── Quantize & evaluate at each bit width ─────────────────────
        for bits in bits_list:
            logger.info(">>> Quantizing: %s%d", quant_method, bits)
            quant_dir = _run_quantize(edit_dir, quant_method, bits)

            logger.info(">>> Evaluating: %s", quant_dir)
            _run_evaluate(quant_dir)

        logger.info("Done: %s / %s\n", method, model)

    logger.info("=" * 40)
    logger.info("All methods complete for model: %s", model)
    logger.info("=" * 40)
