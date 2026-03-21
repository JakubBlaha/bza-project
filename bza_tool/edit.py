"""Apply knowledge edits to a model using EasyEdit.

Supports multiple editing algorithms (ROME, MEMIT, UltraEdit, etc.) through
a unified interface.  The algorithm is selected via the ``--method`` CLI flag.
"""

import json
import logging
from pathlib import Path

from bza_tool.utils import ensure_easyedit_on_path, save_edit_metadata, ensure_dir, ensure_model_exists, load_counterfact

logger = logging.getLogger(__name__)

HPARAMS_REGISTRY: dict[str, tuple[str, str]] = {
    "AlphaEdit": ("easyeditor", "AlphaEditHyperParams"),
    "MEMIT":     ("easyeditor", "MEMITHyperParams"),
    "EMMET":     ("easyeditor", "EMMETHyperParams"),
    "ROME":     ("easyeditor", "ROMEHyperParams"),
}


def _get_hparams_class(method: str):
    """Dynamically import and return the HyperParams class for *method*."""
    if method not in HPARAMS_REGISTRY:
        raise ValueError(
            f"Unknown editing method '{method}'. "
            f"Supported: {sorted(HPARAMS_REGISTRY)}"
        )
    module_name, class_name = HPARAMS_REGISTRY[method]
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _patch_hparams(yaml_path: str, fp16: bool) -> tuple[str, dict]:
    """Return a (possibly patched) hparams path and the config dict.

    If the requested fp16 setting differs from what's in the YAML, we write a
    temporary copy with the override so we don't mutate the vendored file.
    """
    import yaml
    import tempfile

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name", "")
    if model_name.startswith("./"):
        ensure_model_exists(model_name)

    current_fp16 = cfg.get("fp16", False)
    if current_fp16 == fp16:
        return yaml_path, cfg

    cfg["fp16"] = fp16
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="edit_hparams_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    logger.info("Patched hparams fp16=%s -> %s (tmp: %s)", current_fp16, fp16, tmp.name)
    return tmp.name, cfg


def _capture_locality_baseline(editor, hparams, records: list[dict]) -> None:
    """Run the pre-edit model on each record's neighborhood prompts and store
    the predicted token IDs in ``record["locality_pre_edit"]``.

    ``evaluate`` uses these to compute locality accuracy by comparing them
    against post-edit predictions for the same prompts.
    """
    from easyeditor.evaluate.evaluate_utils import test_batch_prediction_acc

    logger.info("Capturing pre-edit locality baselines for %d records...", len(records))
    for record in records:
        prompts = record.get("neighborhood_prompts", [])
        if prompts:
            # One batched call — returns the single next predicted token ID per prompt.
            pre = test_batch_prediction_acc(
                editor.model, editor.tok, hparams,
                prompts, None, hparams.device, locality=True,
            )
            record["locality_pre_edit"] = pre if isinstance(pre, list) else [pre]
        else:
            record["locality_pre_edit"] = []


def run_edit(args) -> None:
    """CLI entry point for the ``edit`` subcommand."""
    from bza_tool.utils import setup_logging
    setup_logging()

    ensure_easyedit_on_path()

    # Resolve case-insensitive method name
    method_map = {k.upper(): k for k in HPARAMS_REGISTRY}
    method = method_map.get(args.method.upper())
    if method is None:
        raise ValueError(
            f"Unknown editing method '{args.method}'. "
            f"Supported: {sorted(HPARAMS_REGISTRY)}"
        )

    from easyeditor.editors.editor import BaseEditor  # type: ignore

    HParamsClass = _get_hparams_class(method)

    # ── Load & optionally patch hparams ────────────────────────────────────
    hparams_path, cfg = _patch_hparams(args.model_config, fp16=args.fp16)
    hparams = HParamsClass.from_hparams(hparams_path)

    # Override batch_size to process all edits in one pass (default=1 is very slow)
    if hasattr(hparams, "batch_size"):
        hparams.batch_size = args.num_edits

    logger.info("Method: %s | Model: %s | fp16: %s",
                method, hparams.model_name, getattr(hparams, "fp16", False))

    # ── Load CounterFact data ──────────────────────────────────────────────
    records = load_counterfact(num_edits=args.num_edits)

    # ── Determine output directory ─────────────────────────────────────────
    if args.output_dir is None:
        model_basename = Path(hparams.model_name).name
        num_facts = len(records)
        output_dir = Path("./outputs") / model_basename / method / str(num_facts)
    else:
        output_dir = Path(args.output_dir)

    output_dir = ensure_dir(output_dir)
    prompts = [r["prompt"] for r in records]
    subjects = [r["subject"] for r in records]
    target_new = [r["target_new"] for r in records]

    # ── Run editing ────────────────────────────────────────────────────────
    editor = BaseEditor.from_hparams(hparams)
    _capture_locality_baseline(editor, hparams, records)

    from easyeditor.editors.batch_editor import BatchEditor

    if BatchEditor.is_batchable_method(method):
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts,
            target_new=target_new,
            subject=subjects,
            keep_original_weight=False,
            sequential_edit=True,
            test_generation=False,
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subjects,
            keep_original_weight=False,
            sequential_edit=True,
            test_generation=False,
        )

    # ── Save edited model ──────────────────────────────────────────────────
    logger.info("Saving edited model to %s", output_dir)
    edited_model.save_pretrained(output_dir)

    # Save tokenizer alongside the model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    tokenizer.save_pretrained(output_dir)

    # ── Save edit metadata + metrics ───────────────────────────────────────
    meta = {
        "source_model": hparams.model_name,
        "edit_method": method,
        "fp16": args.fp16,
        "num_edits": len(records),
        "records": records,
        "config": cfg,  # Save the config dict
    }
    save_edit_metadata(output_dir, meta)

    metrics_path = output_dir / "edit_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Done. Metrics saved to %s", metrics_path)
