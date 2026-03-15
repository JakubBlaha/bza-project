"""CLI definitions for bza_tool."""

import argparse
import sys

from bza_tool.utils import EASYEDIT_HPARAMS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bza_tool",
        description="Benchmark how quantization affects retention of "
                    "knowledge-edited facts in LLMs (CounterFact).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_edit = subparsers.add_parser(
        "edit",
        help="Apply knowledge edits to a model using EasyEdit.",
    )
    p_edit.add_argument(
        "--method",
        type=str,
        default="AlphaEdit",
        help="Editing algorithm to use (default: AlphaEdit). "
             "Supported: AlphaEdit, MEMIT, EMMET.",
    )
    p_edit.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to EasyEdit hparams YAML "
             f"(e.g. {EASYEDIT_HPARAMS_DIR}/llama-7b.yaml).",
    )
    p_edit.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the edited model, tokenizer, and metadata.",
    )
    p_edit.add_argument(
        "--num-edits",
        type=int,
        default=None,
        help="Number of CounterFact edits to apply (default: all).",
    )
    p_edit.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Run editing in fp16 (default: fp32 for reproducibility).",
    )

    p_quant = subparsers.add_parser(
        "quantize",
        help="Quantize a model with AWQ or GPTQ.",
    )
    p_quant.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory to quantize.",
    )
    p_quant.add_argument(
        "--method",
        type=str,
        required=True,
        help="Quantization method (e.g. gptq, awq, gptaq, qqq, gar).",
    )
    p_quant.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Target bit width (default: 4).",
    )
    p_quant.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the quantized model.",
    )

    p_eval = subparsers.add_parser(
        "evaluate",
        help="Evaluate fact retention on CounterFact prompts.",
    )
    p_eval.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory to evaluate.",
    )
    p_eval.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write JSON evaluation results.",
    )
    p_eval.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N edits (default: all).",
    )

    p_dl = subparsers.add_parser(
        "download",
        help="Download a model from HuggingFace Hub to ./hugging_cache.",
    )
    p_dl.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID (e.g. 'gpt2-xl').",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Import lazily so --help works without heavy deps installed
    if args.command == "download":
        from bza_tool.download import run_download
        run_download(args)

    elif args.command == "edit":
        from bza_tool.edit import run_edit
        run_edit(args)

    elif args.command == "quantize":
        from bza_tool.quantize import run_quantize
        run_quantize(args)

    elif args.command == "evaluate":
        from bza_tool.evaluate import run_evaluate
        run_evaluate(args)

    else:
        parser.print_help()
        sys.exit(1)
