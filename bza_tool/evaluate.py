import re
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from bza_tool.utils import load_edit_metadata, setup_logging

logger = logging.getLogger(__name__)


def setup_prediction_logger(output_dir: Path) -> logging.Logger:
    prediction_log_file = output_dir / "predictions.log"
    pred_logger = logging.getLogger('predictions')
    pred_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(prediction_log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    pred_logger.addHandler(handler)

    return pred_logger


def _get_target_logprob(model, tokenizer, prompt: str, target: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    target_str = " " + target.strip() if not target.startswith(" ") else target
    full_ids = tokenizer(prompt + target_str, return_tensors="pt").input_ids.to(model.device)

    target_ids = full_ids[0, prompt_ids.shape[1]:]
    if len(target_ids) == 0:
        return -float('inf')

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[0]

    start_idx = prompt_ids.shape[1] - 1
    end_idx = full_ids.shape[1] - 1
    target_logits = logits[start_idx:end_idx]

    log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
    target_log_probs = log_probs[range(len(target_ids)), target_ids]

    return target_log_probs.sum().item()


def _compare_logits(
    model, tokenizer, prompt: str, target_win: str, target_lose: str, pred_logger: logging.Logger
) -> bool:
    lp_win = _get_target_logprob(model, tokenizer, prompt, target_win)
    lp_lose = _get_target_logprob(model, tokenizer, prompt, target_lose)
    return lp_win > lp_lose


def _is_correct_prediction(
    model, tokenizer, prompt: str, target: str, pred_logger: logging.Logger, max_new_tokens: int = 10,
) -> bool:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)

    generated = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    target_clean = target.strip()
    pattern = r'^' + re.escape(target_clean.lower()) + r'(?:\b|$)'
    is_ok = bool(re.match(pattern, generated.lower()))

    pred_logger.info(f"Prompt: {prompt} | Gen: {generated} | Target: {target_clean} | OK: {is_ok}")
    return is_ok


def _compute_metrics(
    prompts: list[str], target_win: str, target_lose: str, model, tokenizer, pred_logger: logging.Logger, is_locality: bool = False
) -> tuple[float | None, float | None]:
    if not prompts or not target_win:
        return None, None

    gen_correct = []
    prob_correct = []

    for p in prompts:
        # For locality, generative success means it outputs target_lose (the original true fact), not target_win (the new injected fact)
        gen_target = target_lose if is_locality else target_win

        gen_correct.append(int(_is_correct_prediction(model, tokenizer, p, gen_target, pred_logger)))
        if target_lose:
            prob_correct.append(int(_compare_logits(model, tokenizer, p, target_win, target_lose, pred_logger)))

    gen_acc = sum(gen_correct) / len(gen_correct)
    prob_acc = sum(prob_correct) / len(prob_correct) if prob_correct else None

    return gen_acc, prob_acc


def evaluate_single_edit(rec: dict, model, tokenizer, pred_logger: logging.Logger) -> dict:
    entry = {"case_id": rec["case_id"]}
    t_new = rec["target_new"]
    t_true = rec.get("target_true", "")

    # Efficacy
    entry["efficacy_gen"] = _is_correct_prediction(model, tokenizer, rec["prompt"], t_new, pred_logger)
    entry["efficacy_prob"] = _compare_logits(model, tokenizer, rec["prompt"],
                                             t_new, t_true, pred_logger) if t_true else None

    # Generality
    gen_acc, prob_acc = _compute_metrics(
        rec.get("paraphrase_prompts", []), t_new, t_true, model, tokenizer, pred_logger, is_locality=False
    )
    entry["generality_gen"] = gen_acc
    entry["generality_prob"] = prob_acc

    # Locality (target_win is t_true, target_lose is t_new for probabilities)
    gen_acc, prob_acc = _compute_metrics(
        rec.get("neighborhood_prompts", []), t_true, t_new, model, tokenizer, pred_logger, is_locality=True
    )
    entry["locality_gen"] = gen_acc
    entry["locality_prob"] = prob_acc

    return entry


def _compute_summary(model_path: Path, meta: dict, results: list[dict]) -> dict:
    def _avg(key: str):
        vals = [r[key] for r in results if r.get(key) is not None]
        return (sum(vals) / len(vals) * 100) if vals else None

    return {
        "model_path": str(model_path),
        "source_model": meta.get("source_model", "unknown"),
        "num_evaluated": len(results),
        "efficacy_gen_accuracy": _avg("efficacy_gen"),
        "efficacy_prob_accuracy": _avg("efficacy_prob"),
        "generality_gen_accuracy": _avg("generality_gen"),
        "generality_prob_accuracy": _avg("generality_prob"),
        "locality_gen_accuracy": _avg("locality_gen"),
        "locality_prob_accuracy": _avg("locality_prob"),
    }


def run_evaluate(args) -> None:
    setup_logging()

    model_path = Path(args.model_path)
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{model_path.parent.name}_{model_path.name}.json"
    output_file = output_dir / model_name

    pred_logger = setup_prediction_logger(output_dir)

    meta = load_edit_metadata(model_path)
    records = meta["records"]
    if args.num_samples is not None:
        records = records[: args.num_samples]

    logger.info("Evaluating %d edits from %s", len(records), model_path)

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

    results = [evaluate_single_edit(rec, model, tokenizer, pred_logger)
               for rec in tqdm(records, desc="Evaluating edits")]

    summary = _compute_summary(model_path, meta, results)
    output = {"summary": summary, "per_edit": results}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Results saved to %s", output_file)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
