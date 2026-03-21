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

    # Edit
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
        default=None,
        help="Directory to save the edited model, tokenizer, and metadata. "
             "Defaults to ./outputs/{model}/{method}/{num_edits}.",
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

    # Quantize
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
        choices=[2, 3, 4, 8],
        default=4,
        help="Target bit width (default: 4).",
    )

    # Eval
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

    # Run (full pipeline: edit -> evaluate -> quantize -> evaluate)
    p_run = subparsers.add_parser(
        "run",
        help="Run the full pipeline: edit, evaluate, quantize at each "
             "bit width, and evaluate each quantized model — all in one process.",
    )
    p_run.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (must match a hparams YAML filename, e.g. gpt2-xl).",
    )
    p_run.add_argument(
        "--methods",
        type=str,
        required=True,
        help="Comma-separated editing methods (e.g. AlphaEdit,MEMIT,EMMET).",
    )
    p_run.add_argument(
        "--num-edits",
        type=int,
        default=1000,
        help="Number of CounterFact edits (default: 1000).",
    )
    p_run.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Run editing in fp16 (default: fp32).",
    )
    p_run.add_argument(
        "--quant-method",
        type=str,
        default="gptq",
        help="Quantization method (default: gptq).",
    )
    p_run.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 4, 3, 2],
        help="Bit widths to quantize (default: 8 4 3 2).",
    )

    # Download model
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

    elif args.command == "run":
        from bza_tool.run import run_pipeline

        run_pipeline(args)

    else:
        parser.print_help()
        sys.exit(1)
