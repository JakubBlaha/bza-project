"""Shared utilities for bza_tool: paths, logging, EasyEdit sys.path setup."""

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for the tool."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=LOG_FORMAT)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_EASYEDIT = PROJECT_ROOT / "vendor" / "EasyEdit"
EASYEDIT_HPARAMS_DIR = VENDOR_EASYEDIT / "hparams" / "ROME"


def get_hparams_dir(alg_name: str) -> Path:
    """Return the hparams directory for a given EasyEdit algorithm."""
    d = VENDOR_EASYEDIT / "hparams" / alg_name
    if not d.exists():
        raise FileNotFoundError(
            f"No hparams directory found for algorithm '{alg_name}' at {d}"
        )
    return d


def ensure_easyedit_on_path() -> None:
    """Add vendor/EasyEdit to sys.path so its modules can be imported."""
    easyedit_str = str(VENDOR_EASYEDIT)
    if easyedit_str not in sys.path:
        sys.path.insert(0, easyedit_str)


# ---------------------------------------------------------------------------
# Edit metadata helpers
# ---------------------------------------------------------------------------

EDIT_META_FILENAME = "edit_metadata.json"


def save_edit_metadata(output_dir: Path, metadata: dict) -> Path:
    """Persist edit metadata (CounterFact IDs, prompts, targets) alongside the
    saved model so that the evaluate command can reproduce the same test set."""
    path = output_dir / EDIT_META_FILENAME
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def load_edit_metadata(model_path: Path) -> dict:
    """Load edit metadata saved by rome_edit."""
    path = model_path / EDIT_META_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No edit metadata found at {path}. "
            "Make sure this model directory was produced by 'edit'."
        )
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_model_exists(model_path_str: str) -> None:
    """Check if model path exists, raise error with download info if not."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        suggested_id = model_path.name
        raise FileNotFoundError(
            f"\n\n[ERROR] Model not found at: {model_path_str}\n"
            f"Please download the model first using:\n"
            f"  uv run python -m bza_tool download {suggested_id}\n"
        )


def load_counterfact(num_edits: int | None = None) -> list[dict]:
    """Load the CounterFact dataset via HuggingFace ``datasets``.

    Returns a list of dicts with keys: prompt, subject, target_new,
    plus paraphrase / neighborhood data for evaluation.
    """
    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split="train")
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
