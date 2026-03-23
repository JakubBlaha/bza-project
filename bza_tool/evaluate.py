import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from bza_tool.utils import load_edit_metadata, setup_logging, ensure_easyedit_on_path
from bza_tool.edit import _get_hparams_class

logger = logging.getLogger(__name__)


def evaluate_single_edit(rec: dict, model, tokenizer, hparams, model_name: str) -> dict:
    from easyeditor.evaluate import compute_edit_quality, compute_rewrite_or_rephrase_quality
    from easyeditor.evaluate.evaluate_utils import test_batch_prediction_acc

    # Normalize record keys to match EasyEdit's expected format.
    # Handles metadata saved before the ground_truth rename.
    normalized = dict(rec)

    if "ground_truth" not in normalized and "target_true" in normalized:
        normalized["ground_truth"] = normalized["target_true"]

    # Do NOT pass rephrase_prompt to compute_edit_quality — EasyEdit's
    # test_prediction_acc zips prompts with targets, so passing a list of
    # rephrase prompts with a scalar target string silently iterates over
    # the target's characters instead of the full string.  We evaluate
    # rephrase accuracy ourselves below.
    paraphrases = normalized.pop("paraphrase_prompts", [])
    normalized.pop("rephrase_prompt", None)

    metrics = compute_edit_quality(
        model=model,
        model_name=model_name,
        hparams=hparams,
        tok=tokenizer,
        record=normalized,
        device=hparams.device
    )

    # Evaluate rephrase prompts one-by-one so each is passed as a scalar
    # string, which test_prediction_acc handles correctly.
    target_new = normalized["target_new"]
    if paraphrases:
        rephrase_accs = []
        for pp in paraphrases:
            rp_metrics = compute_rewrite_or_rephrase_quality(
                model, model_name, hparams, tokenizer,
                pp, target_new, device=hparams.device, test_rephrase=True,
            )
            rephrase_accs.extend(
                rp_metrics["rephrase_acc"]
                if isinstance(rp_metrics["rephrase_acc"], list)
                else [rp_metrics["rephrase_acc"]]
            )
        metrics["rephrase_acc"] = rephrase_accs

    # Compute locality accuracy: compare post-edit predictions to pre-edit baseline.
    locality_pre = rec.get("locality_pre_edit", [])
    neighborhood_prompts = rec.get("neighborhood_prompts", [])
    locality_accs = []

    if neighborhood_prompts and locality_pre:
        # One batched call — returns the single next predicted token ID per prompt.
        post = test_batch_prediction_acc(
            model, tokenizer, hparams,
            neighborhood_prompts, None, hparams.device, locality=True,
        )
        post = post if isinstance(post, list) else [post]
        locality_accs = [float(p == q) for p, q in zip(locality_pre, post)]

    entry = {"case_id": rec["case_id"]}
    entry.update(metrics)
    entry["locality_acc"] = float(np.mean(locality_accs)) if locality_accs else None
    return entry


def compute_summary(model_path: Path, results: list[dict]) -> dict:
    def _avg(key: str):
        # Get values from all records by a key and compute a mean
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals) * 100) if vals else None

    return {
        "model_path": str(model_path),
        "num_evaluated": len(results),
        "rewrite_accuracy": _avg("rewrite_acc"),
        "rephrase_accuracy": _avg("rephrase_acc"),
        "locality_accuracy": _avg("locality_acc"),
    }


def run_evaluate(args) -> None:
    setup_logging()
    ensure_easyedit_on_path()

    model_path = Path(args.model_path)
    output_dir = Path("./results")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_path.parent.parent.name
    output_filename = f"{model_name}_{model_path.parent.name}_{model_path.name}.json"
    output_file = output_dir / output_filename

    logger.info("Configuration:")
    logger.info(f"  {model_path=}")
    logger.info(f"  {output_dir=}")
    logger.info(f"  {output_filename=}")

    meta = load_edit_metadata(model_path)
    records = meta["records"]

    # Make sure that we also have the locality data, the edit should not modify
    # unrelated objects
    if not all("locality_pre_edit" in r for r in records):
        raise RuntimeError(
            "Records are missing 'locality_pre_edit' baselines. "
            "Re-run the 'edit' command to capture pre-edit locality data."
        )

    logger.info("Evaluating %d edits from %s", len(records), model_path)

    method = meta["edit_method"]
    HParamsClass = _get_hparams_class(method)
    cfg = meta["config"]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()

    hparams = HParamsClass.from_hparams(tmp.name)
    os.unlink(tmp.name)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_str = meta["source_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate model — match the dtype used during editing so evaluation is consistent
    torch_dtype = torch.bfloat16 if meta.get("fp16") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Set model into evaluation mode
    model.eval()

    results = [
        evaluate_single_edit(rec, model, tokenizer, hparams, model_name_str)
        for rec in tqdm(records, desc="Evaluating edits")
    ]

    summary = compute_summary(model_path, results)
    output = {"summary": summary, "per_edit": results}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", output_file)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
