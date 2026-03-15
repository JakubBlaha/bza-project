"""Pipeline orchestration: run full research scenarios end-to-end."""

import logging
import subprocess
import sys
from pathlib import Path

from bza_tool.utils import setup_logging

logger = logging.getLogger(__name__)


def _run_step(cmd: list[str], description: str) -> None:
    """Run a bza_tool subcommand as a subprocess."""
    logger.info("=" * 60)
    logger.info("STEP: %s", description)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, check=True)
    logger.info("Step completed: %s (exit code %d)", description, result.returncode)


def _model_name_from_config(config_path: str) -> str:
    """Extract a short model name from the config YAML filename."""
    return Path(config_path).stem  # e.g. "llama-7b"


def _scenario_edit_eval_quant_eval(args) -> None:
    """Scenario 1: Model → Edit → Eval → Quant → Eval"""
    model_name = _model_name_from_config(args.model_config)
    edit_method = args.edit_method
    base = Path(args.output_base)
    python = sys.executable

    # Paths
    edit_dir = base / model_name / edit_method.lower()
    edit_eval = base / model_name / "results" / f"{edit_method.lower()}_eval.json"
    quant_dir = base / model_name / f"{edit_method.lower()}-{args.quant_method}{args.bits}"
    quant_eval = base / model_name / "results" / f"{edit_method.lower()}_{args.quant_method}{args.bits}_eval.json"

    # Step 1: Knowledge edit
    cmd = [python, "-m", "bza_tool", "edit",
           "--method", edit_method,
           "--model-config", args.model_config,
           "--output-dir", str(edit_dir)]
    if args.num_edits is not None:
        cmd += ["--num-edits", str(args.num_edits)]
    if args.fp16:
        cmd.append("--fp16")
    _run_step(cmd, f"{edit_method} edit ({model_name})")

    # Step 2: Evaluate post-edit
    cmd = [python, "-m", "bza_tool", "evaluate",
           "--model-path", str(edit_dir),
           "--output-file", str(edit_eval)]
    _run_step(cmd, f"Evaluate post-{edit_method} ({model_name})")

    # Step 3: Quantize
    cmd = [python, "-m", "bza_tool", "quantize",
           "--model-path", str(edit_dir),
           "--method", args.quant_method,
           "--bits", str(args.bits),
           "--output-dir", str(quant_dir)]
    _run_step(cmd, f"Quantize ({model_name}, {args.quant_method.upper()} {args.bits}-bit)")

    # Step 4: Evaluate post-quantization
    cmd = [python, "-m", "bza_tool", "evaluate",
           "--model-path", str(quant_dir),
           "--output-file", str(quant_eval)]
    _run_step(cmd, f"Evaluate post-quantization ({model_name})")

    logger.info("Scenario edit_eval_quant_eval complete for %s", model_name)
    logger.info("  Post-edit results:  %s", edit_eval)
    logger.info("  Post-Quant results: %s", quant_eval)


def _scenario_quant_edit_eval(args) -> None:
    """Scenario 2: Model → Quant → Edit → Eval"""
    import yaml

    model_name = _model_name_from_config(args.model_config)
    edit_method = args.edit_method
    base = Path(args.output_base)
    python = sys.executable

    # We need the original model name from the YAML to quantize from HF
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)
    original_model = cfg["model_name"]

    # Paths
    quant_dir = base / model_name / f"quant-{args.quant_method}{args.bits}"
    edit_dir = base / model_name / f"quant{args.bits}-{edit_method.lower()}"
    edit_eval = base / model_name / "results" / f"quant{args.bits}_{edit_method.lower()}_eval.json"

    # Step 1: Quantize the original model
    cmd = [python, "-m", "bza_tool", "quantize",
           "--model-path", original_model,
           "--method", args.quant_method,
           "--bits", str(args.bits),
           "--output-dir", str(quant_dir)]
    _run_step(cmd, f"Quantize original ({model_name}, {args.quant_method.upper()} {args.bits}-bit)")

    # Step 2: Knowledge edit on the quantized model
    # We need to create a patched hparams YAML pointing to the quantized model
    import tempfile
    patched_cfg = cfg.copy()
    patched_cfg["model_name"] = str(quant_dir)
    if args.fp16:
        patched_cfg["fp16"] = True
    tmp_yaml = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"{edit_method.lower()}_{model_name}_quant_"
    )
    yaml.dump(patched_cfg, tmp_yaml)
    tmp_yaml.close()

    cmd = [python, "-m", "bza_tool", "edit",
           "--method", edit_method,
           "--model-config", tmp_yaml.name,
           "--output-dir", str(edit_dir)]
    if args.num_edits is not None:
        cmd += ["--num-edits", str(args.num_edits)]
    if args.fp16:
        cmd.append("--fp16")
    _run_step(cmd, f"{edit_method} edit on quantized ({model_name})")

    # Step 3: Evaluate
    cmd = [python, "-m", "bza_tool", "evaluate",
           "--model-path", str(edit_dir),
           "--output-file", str(edit_eval)]
    _run_step(cmd, f"Evaluate quant+{edit_method} ({model_name})")

    logger.info("Scenario quant_edit_eval complete for %s", model_name)
    logger.info("  Results: %s", edit_eval)


def run_pipeline(args) -> None:
    """CLI entry point for the ``pipeline`` subcommand."""
    setup_logging()

    if args.scenario == "edit_eval_quant_eval":
        _scenario_edit_eval_quant_eval(args)
    elif args.scenario == "quant_edit_eval":
        _scenario_quant_edit_eval(args)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")
