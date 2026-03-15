"""Evaluate a model's retention of ROME-edited facts on CounterFact prompts.

Works on any model checkpoint — original, post-ROME, or post-quantization —
by re-running inference on the same prompts that were edited.
"""

import re
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from bza_tool.utils import load_edit_metadata, setup_logging

logger = logging.getLogger(__name__)


def setup_prediction_logger(output_dir: Path) -> logging.Logger:
    """Set up a logger for prediction details, writing to predictions.log."""
    prediction_log_file = output_dir / "predictions.log"
    pred_logger = logging.getLogger('predictions')
    pred_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(prediction_log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    pred_logger.addHandler(handler)
    return pred_logger


def _compute_average_correctness(
    prompts: list[str], target: str, model, tokenizer, pred_logger: logging.Logger
) -> float | None:
    """Compute the average correctness for a list of prompts against a target."""
    if not prompts or not target:
        return None
    correct = [
        int(_is_correct_prediction(model, tokenizer, p, target, pred_logger))
        for p in prompts
    ]
    return sum(correct) / len(correct)


def evaluate_single_edit(rec: dict, model, tokenizer, pred_logger: logging.Logger) -> dict:
    """Evaluate a single edit record for efficacy, generality, and locality."""
    entry = {"case_id": rec["case_id"]}

    # Efficacy: does the model predict target_new for the direct prompt?
    eff = _is_correct_prediction(model, tokenizer, rec["prompt"], rec["target_new"], pred_logger)
    entry["efficacy"] = eff

    # Generality: does the model predict target_new for paraphrase prompts?
    entry["generality"] = _compute_average_correctness(
        rec.get("paraphrase_prompts", []), rec["target_new"], model, tokenizer, pred_logger
    )

    # Locality: does the model still predict target_true for neighborhood prompts?
    entry["locality"] = _compute_average_correctness(
        rec.get("neighborhood_prompts", []), rec.get("target_true") or "", model, tokenizer, pred_logger
    )

    return entry


def _is_correct_prediction(
    model,
    tokenizer,
    prompt: str,
    target: str,
    pred_logger: logging.Logger,
    max_new_tokens: int = 10,
) -> bool:
    """Check whether greedy generation from *prompt* starts with *target*."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
    target_clean = target.strip()

    pattern = r'^' + re.escape(target_clean.lower()) + r'(?:\b|$)'
    is_ok = bool(re.match(pattern, generated.lower()))

    pred_logger.info("Prompt      : %s", prompt)
    pred_logger.info("Target clean: %s", target_clean)
    pred_logger.info("Generated:    %s", generated)
    pred_logger.info("Is ok:        %s", is_ok)

    return is_ok


def _compute_summary(
    model_path: Path, meta: dict, results: list[dict], efficacy_scores: list[int],
    generality_scores: list[float], locality_scores: list[float]
) -> dict:
    """Compute the evaluation summary statistics."""
    return {
        "model_path": str(model_path),
        "source_model": meta.get("source_model", "unknown"),
        "num_evaluated": len(results),
        "efficacy_accuracy": (sum(efficacy_scores) / len(efficacy_scores) * 100)
        if efficacy_scores else 0.0,
        "generality_accuracy": (sum(generality_scores) / len(generality_scores) * 100)
        if generality_scores else None,
        "locality_accuracy": (sum(locality_scores) / len(locality_scores) * 100)
        if locality_scores else None,
    }


def run_evaluate(args) -> None:
    """CLI entry point for the ``evaluate`` subcommand."""
    setup_logging()

    model_path = Path(args.model_path)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up prediction logger
    pred_logger = setup_prediction_logger(output_file.parent)

    # ── Load edit metadata ─────────────────────────────────────────────────
    meta = load_edit_metadata(model_path)
    records = meta["records"]
    if args.num_samples is not None:
        records = records[: args.num_samples]

    logger.info("Evaluating %d edits from %s", len(records), model_path)

    # ── Load model + tokenizer ─────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float32,
    )
    model.eval()

    # ── Evaluate each edit ─────────────────────────────────────────────────
    results = []
    efficacy_scores = []
    generality_scores = []
    locality_scores = []

    for rec in tqdm(records, desc="Evaluating edits"):
        entry = evaluate_single_edit(rec, model, tokenizer, pred_logger)
        results.append(entry)

        # Collect scores for aggregation
        efficacy_scores.append(int(entry["efficacy"]))
        if entry["generality"] is not None:
            generality_scores.append(entry["generality"])
        if entry["locality"] is not None:
            locality_scores.append(entry["locality"])

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary = _compute_summary(
        model_path, meta, results, efficacy_scores, generality_scores, locality_scores
    )

    output = {"summary": summary, "per_edit": results}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", output_file)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
