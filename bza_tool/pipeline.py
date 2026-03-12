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


def _scenario_rome_eval_quant_eval(args) -> None:
    """Scenario 1: Model → ROME → Eval → Quant → Eval"""
    model_name = _model_name_from_config(args.model_config)
    base = Path(args.output_base)
    python = sys.executable

    # Paths
    rome_dir = base / model_name / "rome"
    rome_eval = base / model_name / "results" / "rome_eval.json"
    quant_dir = base / model_name / f"rome-{args.quant_method}{args.bits}"
    quant_eval = base / model_name / "results" / f"rome_{args.quant_method}{args.bits}_eval.json"

    # Step 1: ROME edit
    cmd = [python, "-m", "bza_tool", "rome-edit",
           "--model-config", args.model_config,
           "--output-dir", str(rome_dir)]
    if args.num_edits is not None:
        cmd += ["--num-edits", str(args.num_edits)]
    if args.fp16:
        cmd.append("--fp16")
    _run_step(cmd, f"ROME edit ({model_name})")

    # Step 2: Evaluate post-ROME
    cmd = [python, "-m", "bza_tool", "evaluate",
           "--model-path", str(rome_dir),
           "--output-file", str(rome_eval)]
    _run_step(cmd, f"Evaluate post-ROME ({model_name})")

    # Step 3: Quantize
    cmd = [python, "-m", "bza_tool", "quantize",
           "--model-path", str(rome_dir),
           "--method", args.quant_method,
           "--bits", str(args.bits),
           "--output-dir", str(quant_dir)]
    _run_step(cmd, f"Quantize ({model_name}, {args.quant_method.upper()} {args.bits}-bit)")

    # Step 4: Evaluate post-quantization
    cmd = [python, "-m", "bza_tool", "evaluate",
           "--model-path", str(quant_dir),
           "--output-file", str(quant_eval)]
    _run_step(cmd, f"Evaluate post-quantization ({model_name})")

    logger.info("Scenario rome_eval_quant_eval complete for %s", model_name)
    logger.info("  Post-ROME results:  %s", rome_eval)
    logger.info("  Post-Quant results: %s", quant_eval)


def _scenario_quant_rome_eval(args) -> None:
    """Scenario 2: Model → Quant → ROME → Eval"""
    import yaml

    model_name = _model_name_from_config(args.model_config)
    base = Path(args.output_base)
    python = sys.executable

    # We need the original model name from the YAML to quantize from HF
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)
    original_model = cfg["model_name"]

    # Paths
    quant_dir = base / model_name / f"quant-{args.quant_method}{args.bits}"
    rome_dir = base / model_name / f"quant{args.bits}-rome"
    rome_eval = base / model_name / "results" / f"quant{args.bits}_rome_eval.json"

    # Step 1: Quantize the original model
    cmd = [python, "-m", "bza_tool", "quantize",
           "--model-path", original_model,
           "--method", args.quant_method,
           "--bits", str(args.bits),
           "--output-dir", str(quant_dir)]
    _run_step(cmd, f"Quantize original ({model_name}, {args.quant_method.upper()} {args.bits}-bit)")

    # Step 2: ROME edit on the quantized model
    # We need to create a patched hparams YAML pointing to the quantized model
    import tempfile
    patched_cfg = cfg.copy()
    patched_cfg["model_name"] = str(quant_dir)
    if args.fp16:
        patched_cfg["fp16"] = True
    tmp_yaml = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"rome_{model_name}_quant_"
    )
    yaml.dump(patched_cfg, tmp_yaml)
    tmp_yaml.close()

    cmd = [python, "-m", "bza_tool", "rome-edit",
           "--model-config", tmp_yaml.name,
           "--output-dir", str(rome_dir)]
    if args.num_edits is not None:
        cmd += ["--num-edits", str(args.num_edits)]
    if args.fp16:
        cmd.append("--fp16")
    _run_step(cmd, f"ROME edit on quantized ({model_name})")

    # Step 3: Evaluate
    cmd = [python, "-m", "bza_tool", "evaluate",
           "--model-path", str(rome_dir),
           "--output-file", str(rome_eval)]
    _run_step(cmd, f"Evaluate quant+ROME ({model_name})")

    logger.info("Scenario quant_rome_eval complete for %s", model_name)
    logger.info("  Results: %s", rome_eval)


def run_pipeline(args) -> None:
    """CLI entry point for the ``pipeline`` subcommand."""
    setup_logging()

    if args.scenario == "rome_eval_quant_eval":
        _scenario_rome_eval_quant_eval(args)
    elif args.scenario == "quant_rome_eval":
        _scenario_quant_rome_eval(args)
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")
