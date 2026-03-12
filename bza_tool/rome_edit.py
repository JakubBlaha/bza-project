"""Apply ROME knowledge edits to a model using EasyEdit."""

import json
import logging
import shutil
from pathlib import Path

from bza_tool.utils import ensure_easyedit_on_path, save_edit_metadata, ensure_dir, ensure_model_exists

logger = logging.getLogger(__name__)


def _load_counterfact(num_edits: int | None = None) -> list[dict]:
    """Load the CounterFact dataset via HuggingFace ``datasets``.

    Returns a list of dicts with keys: prompt, subject, target_new,
    plus paraphrase / neighborhood data for evaluation.
    """
    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split="train")  # original CounterFact
    if num_edits is not None:
        ds = ds.select(range(min(num_edits, len(ds))))

    records = []
    for row in ds:
        records.append({
            "case_id": row["case_id"],
            "prompt": row["requested_rewrite"]["prompt"].format(
                row["requested_rewrite"]["subject"]
            ),
            "subject": row["requested_rewrite"]["subject"],
            "target_new": row["requested_rewrite"]["target_new"]["str"],
            "target_true": row["requested_rewrite"]["target_true"]["str"],
            # Paraphrase prompts for generality evaluation
            "paraphrase_prompts": row.get("paraphrase_prompts", []),
            # Neighborhood prompts for locality evaluation
            "neighborhood_prompts": row.get("neighborhood_prompts", []),
            "attribute_prompts": row.get("attribute_prompts", []),
        })

    logger.info("Loaded %d CounterFact records", len(records))
    return records


def _patch_hparams_fp16(yaml_path: str, fp16: bool) -> str:
    """Return a (possibly patched) hparams path with the desired fp16 setting.

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
        return yaml_path  # no patching needed

    cfg["fp16"] = fp16
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="rome_hparams_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    logger.info("Patched hparams fp16=%s -> %s (tmp: %s)", current_fp16, fp16, tmp.name)
    return tmp.name


def run_rome_edit(args) -> None:
    """CLI entry point for the ``rome-edit`` subcommand."""
    from bza_tool.utils import setup_logging
    setup_logging()

    ensure_easyedit_on_path()

    from easyeditor import ROMEHyperParams, BaseEditor  # type: ignore

    output_dir = ensure_dir(Path(args.output_dir))

    # ── Load & optionally patch hparams ────────────────────────────────────
    hparams_path = _patch_hparams_fp16(args.model_config, fp16=args.fp16)
    hparams = ROMEHyperParams.from_hparams(hparams_path)

    logger.info("Model: %s | fp16: %s | layers: %s",
                hparams.model_name, getattr(hparams, "fp16", False), hparams.layers)

    # ── Load CounterFact data ──────────────────────────────────────────────
    records = _load_counterfact(num_edits=args.num_edits)
    prompts = [r["prompt"] for r in records]
    subjects = [r["subject"] for r in records]
    target_new = [r["target_new"] for r in records]

    # ── Run ROME editing ───────────────────────────────────────────────────
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        keep_original_weight=False,  # we want the mutated model
        test_generation=True,
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
        "fp16": args.fp16,
        "num_edits": len(records),
        "records": records,
    }
    save_edit_metadata(output_dir, meta)

    metrics_path = output_dir / "rome_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Done. Metrics saved to %s", metrics_path)
