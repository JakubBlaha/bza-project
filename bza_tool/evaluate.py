"""Evaluate a model's retention of ROME-edited facts on CounterFact prompts.

Works on any model checkpoint — original, post-ROME, or post-quantization —
by re-running inference on the same prompts that were edited.
"""

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from bza_tool.utils import load_edit_metadata, setup_logging

logger = logging.getLogger(__name__)


def _compute_target_probability(
    model,
    tokenizer,
    prompt: str,
    target: str,
) -> float:
    """Compute the probability that *model* generates *target* given *prompt*.

    We measure P(target | prompt) by looking at the model's next-token
    probabilities for each token of the target, conditioned on the prompt.
    Returns the geometric-mean probability (i.e. exp of mean log-prob).
    """
    full_text = prompt + " " + target
    enc = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_enc = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    with torch.no_grad():
        logits = model(**enc).logits  # (1, seq_len, vocab)

    # We want to score the target tokens: positions prompt_len .. end
    log_probs = torch.log_softmax(logits[0], dim=-1)
    target_ids = enc["input_ids"][0, prompt_len:]

    if len(target_ids) == 0:
        return 0.0

    token_log_probs = []
    for i, tid in enumerate(target_ids):
        # logits at position (prompt_len - 1 + i) predict token at (prompt_len + i)
        pos = prompt_len - 1 + i
        if pos < log_probs.shape[0]:
            token_log_probs.append(log_probs[pos, tid].item())

    if not token_log_probs:
        return 0.0

    import math
    mean_log_prob = sum(token_log_probs) / len(token_log_probs)
    return math.exp(mean_log_prob)


def _is_correct_prediction(
    model,
    tokenizer,
    prompt: str,
    target: str,
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
    return generated.lower().startswith(target_clean.lower())


def run_evaluate(args) -> None:
    """CLI entry point for the ``evaluate`` subcommand."""
    setup_logging()

    model_path = Path(args.model_path)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

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
        torch_dtype=torch.float16,
    )
    model.eval()

    # ── Evaluate each edit ─────────────────────────────────────────────────
    results = []
    efficacy_scores = []
    generality_scores = []
    locality_scores = []

    for rec in tqdm(records, desc="Evaluating"):
        entry = {"case_id": rec["case_id"]}

        # Efficacy: does the model predict target_new for the direct prompt?
        eff = _is_correct_prediction(model, tokenizer, rec["prompt"], rec["target_new"])
        entry["efficacy"] = eff
        efficacy_scores.append(int(eff))

        # Efficacy probability
        entry["efficacy_prob"] = _compute_target_probability(
            model, tokenizer, rec["prompt"], rec["target_new"]
        )

        # Generality: does the model predict target_new for paraphrase prompts?
        para_prompts = rec.get("paraphrase_prompts", [])
        if para_prompts:
            para_correct = [
                int(_is_correct_prediction(model, tokenizer, p, rec["target_new"]))
                for p in para_prompts
            ]
            entry["generality"] = sum(para_correct) / len(para_correct)
            generality_scores.append(entry["generality"])
        else:
            entry["generality"] = None

        # Locality: does the model still predict target_true for neighborhood prompts?
        neigh_prompts = rec.get("neighborhood_prompts", [])
        if neigh_prompts and rec.get("target_true"):
            neigh_correct = [
                int(_is_correct_prediction(model, tokenizer, p, rec["target_true"]))
                for p in neigh_prompts
            ]
            entry["locality"] = sum(neigh_correct) / len(neigh_correct)
            locality_scores.append(entry["locality"])
        else:
            entry["locality"] = None

        results.append(entry)

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary = {
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

    output = {"summary": summary, "per_edit": results}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", output_file)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
