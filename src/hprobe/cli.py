"""hprobe CLI — run hallucination neuron discovery from the terminal."""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keys per known dataset format: (options_key, answer_key)
_FORMAT_KEYS: Dict[str, Tuple[str, str]] = {
    "mmlu": ("choices", "answer"),  # list options + int answer
    "medqa": ("options", "answer_idx"),  # dict options + letter answer
    "medmcqa": ("options", "cop"),  # dict options + int answer (0-3)
}


def detect_format(sample: Dict) -> Optional[str]:
    """Infer dataset format from a single sample's keys."""
    if "choices" in sample and isinstance(sample.get("answer"), int):
        return "mmlu"
    if "options" in sample and "answer_idx" in sample:
        return "medqa"
    if "options" in sample and "cop" in sample:
        return "medmcqa"
    return None


def format_keys(fmt: str) -> Tuple[str, str]:
    """Return (options_key, answer_key) for a given format name."""
    if fmt not in _FORMAT_KEYS:
        raise ValueError(f"Unknown format '{fmt}'. Supported: {list(_FORMAT_KEYS)}")
    return _FORMAT_KEYS[fmt]


def load_samples(path: str, n: int) -> List[Dict]:
    """Load up to n samples from a JSONL, JSON, or Parquet file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    samples = []
    if p.suffix == ".jsonl":
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    if len(samples) >= n:
                        break
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
        samples = (data if isinstance(data, list) else [data])[:n]
    elif p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print(
                "Error: pyarrow is required for .parquet files. Run: pip install pyarrow",
                file=sys.stderr,
            )
            sys.exit(1)
        samples = pq.read_table(p).to_pylist()[:n]
    else:
        print(
            f"Error: unsupported file format '{p.suffix}'. Use .jsonl, .json, or .parquet",
            file=sys.stderr,
        )
        sys.exit(1)

    if not samples:
        print("Error: no samples loaded from file", file=sys.stderr)
        sys.exit(1)

    return samples


def _default_output_path(model: str, dataset_path: str) -> str:
    """Build a default output base path (no extension — save() adds .json and .pkl)."""
    model_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
    dataset_name = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_safe}_{dataset_name}_{ts}"


def _resolve_format(args, samples):
    """Resolve format string to (options_key, answer_key). Returns (fmt, options_key, answer_key)."""
    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(samples[0])
        if fmt is None:
            print(
                "  Warning: could not auto-detect format. "
                "Falling back to options_key='options', answer_key='answer'.",
                file=sys.stderr,
            )
            return None, "options", "answer"
        return fmt, *format_keys(fmt)
    return fmt, *format_keys(fmt)


def _load_model(args):
    """Load tokenizer and model from args. Returns (tokenizer, model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device,
    )
    model.eval()
    return tokenizer, model


def _print_score(results):
    def _fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"  AUROC:           {_fmt(results['auroc'])}")
    print(f"  Random baseline: {_fmt(results['random_baseline_auroc'])}")
    print(
        f"  AUROC gap:      {results['auroc_gap']:+.3f}"
        if results["auroc_gap"] is not None
        else "  AUROC gap:       n/a"
    )
    print(f"  Balanced acc:    {_fmt(results['balanced_accuracy'])}")


def cmd_run(args: argparse.Namespace) -> None:
    from hprobe import HProbe, __version__

    sep = "─" * 68
    print(f"\nhprobe v{__version__}  |  model: {args.model}  |  samples: {args.samples}")
    print(sep)

    samples = load_samples(args.data, args.samples)
    fmt, options_key, answer_key = _resolve_format(args, samples)

    print(
        f"  Dataset:     {Path(args.data).name}  ({len(samples)} samples, format={fmt or 'auto'})"
    )
    print(f"  Keys:        options_key={options_key!r}  answer_key={answer_key!r}")
    print(f"  Contrastive: {not args.no_contrastive}")

    print(f"  Loading {args.model}...", end="", flush=True)
    tokenizer, model = _load_model(args)
    print(" done")

    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None

    print(f"  Fitting (l1_C={args.l1_c})...", end="", flush=True)
    probe = HProbe(
        model,
        tokenizer,
        l1_C=args.l1_c,
        contrastive=not args.no_contrastive,
        layer_stride=args.layer_stride,
        validation_split=args.validation_split,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    probe.fit(samples, options_key=options_key, answer_key=answer_key)
    print(" done")
    print(f"  H-Neurons:   {probe.n_neurons_}  ({probe.neuron_ratio_:.3f}‰ of all features)")
    print(f"  Accuracy:    {probe.accuracy_:.3f}")
    print(f"  Layers:      {dict(sorted(probe.layer_distribution_.items()))}")

    print("\n  Scoring...")
    _print_score(probe.score())

    print("\n  Causal validation (alpha → accuracy):")
    cv = probe.causal_validate(alphas=alphas)
    for alpha, acc in sorted(cv.items()):
        tag = ""
        if alpha == 0.0:
            tag = "  ← full suppression"
        elif alpha == 1.0:
            tag = "  ← baseline"
        elif alpha == 2.0:
            tag = "  ← amplification"
        print(f"    {alpha:.1f} → {acc:.3f}{tag}")

    out_path = args.output or _default_output_path(args.model, args.data)
    saved = probe.save(out_path)
    print(f"\n  Saved → {saved}  +  {Path(out_path).with_suffix('.pkl').name}")
    print(sep + "\n")


def cmd_responses(args: argparse.Namespace) -> None:
    from hprobe import HProbe, __version__

    sep = "─" * 68
    print(
        f"\nhprobe v{__version__}  |  model: {args.model}  |  samples: {args.samples}  |  mode: responses"
    )
    print(sep)

    samples = load_samples(args.data, args.samples)

    print(f"  Dataset:     {Path(args.data).name}  ({len(samples)} samples)")
    print(
        f"  Keys:        question={args.question_key!r}  response={args.response_key!r}  "
        f"answer_tokens={args.answer_tokens_key!r}  label={args.label_key!r}"
    )
    print(f"  Aggregation: {args.aggregation}")

    print(f"  Loading {args.model}...", end="", flush=True)
    tokenizer, model = _load_model(args)
    print(" done")

    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None

    print(f"  Fitting (l1_C={args.l1_c})...", end="", flush=True)
    probe = HProbe(
        model,
        tokenizer,
        l1_C=args.l1_c,
        layer_stride=args.layer_stride,
        validation_split=args.validation_split,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    probe.fit_from_responses(
        samples,
        question_key=args.question_key,
        response_key=args.response_key,
        answer_tokens_key=args.answer_tokens_key,
        label_key=args.label_key,
        aggregation=args.aggregation,
    )
    print(" done")
    print(f"  H-Neurons:   {probe.n_neurons_}  ({probe.neuron_ratio_:.3f}‰ of all features)")
    print(f"  Accuracy:    {probe.accuracy_:.3f}")
    print(f"  Layers:      {dict(sorted(probe.layer_distribution_.items()))}")

    print("\n  Scoring...")
    _print_score(probe.score())

    print("\n  Causal validation (alpha → accuracy):")
    cv = probe.causal_validate(alphas=alphas)
    for alpha, acc in sorted(cv.items()):
        tag = ""
        if alpha == 0.0:
            tag = "  ← full suppression"
        elif alpha == 1.0:
            tag = "  ← baseline"
        elif alpha == 2.0:
            tag = "  ← amplification"
        print(f"    {alpha:.1f} → {acc:.3f}{tag}")

    out_path = args.output or _default_output_path(args.model, args.data)
    saved = probe.save(out_path)
    print(f"\n  Saved → {saved}  +  {Path(out_path).with_suffix('.pkl').name}")
    print(sep + "\n")


def cmd_transfer(args: argparse.Namespace) -> None:
    from hprobe import HProbe, __version__

    sep = "─" * 68
    print(f"\nhprobe v{__version__}  |  transfer: {args.probe} → {args.model}")
    print(sep)

    samples = load_samples(args.data, args.samples)
    fmt, options_key, answer_key = _resolve_format(args, samples)

    print(f"  Dataset:  {Path(args.data).name}  ({len(samples)} samples, format={fmt or 'auto'})")
    print(f"  Keys:     options_key={options_key!r}  answer_key={answer_key!r}")

    print(f"  Loading {args.model}...", end="", flush=True)
    tokenizer, model = _load_model(args)
    print(" done")

    print(f"  Loading probe from {args.probe}...", end="", flush=True)
    probe = HProbe.load(args.probe, model, tokenizer)
    print(f" done  ({probe.n_neurons_} H-Neurons)")

    print("\n  Scoring (transfer)...")
    result = probe.score_on(samples, options_key=options_key, answer_key=answer_key)
    _print_score(result)

    out_path = args.output or _default_output_path(args.model, args.data)
    saved = probe.save(out_path)
    print(f"\n  Saved → {saved}  +  {Path(out_path).with_suffix('.pkl').name}")

    print(sep + "\n")


def _add_common_model_args(p):
    """Add --device and --dtype to a subparser."""
    p.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda (default: auto)")
    p.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype (default: auto)",
    )


def _add_common_probe_args(p):
    """Add shared probe hyperparameter args to a subparser."""
    p.add_argument(
        "--l1-c",
        type=float,
        default=0.5,
        dest="l1_c",
        help="Inverse L1 regularisation strength (default: 0.5)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        dest="layer_stride",
        help="Sample every Nth layer (default: 1 = all layers)",
    )
    p.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        dest="validation_split",
        help="Fraction of samples held out for scoring (default: 0.2)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        dest="max_tokens",
        help="Max input tokens before truncation (default: 1024)",
    )
    p.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated alpha values for causal validation (default: 0.0,0.5,1.0,1.5,2.0)",
    )


def main() -> None:
    from hprobe import __version__

    parser = argparse.ArgumentParser(
        prog="hprobe",
        description="Hallucination neuron probe — discover and causally validate H-Neurons",
    )
    parser.add_argument("--version", action="version", version=f"hprobe {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── hprobe run ────────────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Fit, score, and causal-validate on an MCQ dataset")
    run_p.add_argument("--model", required=True, help="HuggingFace model ID")
    run_p.add_argument("--data", required=True, help="Path to .jsonl or .json dataset file")
    run_p.add_argument(
        "--format",
        choices=["auto", "mmlu", "medqa", "medmcqa"],
        default="auto",
        help="Dataset format (default: auto-detect)",
    )
    run_p.add_argument("--samples", type=int, default=100, help="Number of samples (default: 100)")
    run_p.add_argument(
        "--no-contrastive",
        action="store_true",
        dest="no_contrastive",
        help="Disable contrastive labeling (use binary correct/incorrect instead)",
    )
    run_p.add_argument(
        "--output",
        default=None,
        help="Base path to save results (default: auto-named in cwd)",
    )
    _add_common_model_args(run_p)
    _add_common_probe_args(run_p)

    # ── hprobe responses ──────────────────────────────────────────────────────
    resp_p = subparsers.add_parser(
        "responses", help="Fit from pre-generated responses (open-ended / free-text)"
    )
    resp_p.add_argument("--model", required=True, help="HuggingFace model ID")
    resp_p.add_argument("--data", required=True, help="Path to .jsonl or .json dataset file")
    resp_p.add_argument("--samples", type=int, default=100, help="Number of samples (default: 100)")
    resp_p.add_argument(
        "--question-key",
        default="question",
        dest="question_key",
        help="Key for question text (default: question)",
    )
    resp_p.add_argument(
        "--response-key",
        default="response",
        dest="response_key",
        help="Key for generated response text (default: response)",
    )
    resp_p.add_argument(
        "--answer-tokens-key",
        default="answer_tokens",
        dest="answer_tokens_key",
        help="Key for list of answer token strings (default: answer_tokens)",
    )
    resp_p.add_argument(
        "--label-key",
        default="judge",
        dest="label_key",
        help="Key for correctness label (default: judge)",
    )
    resp_p.add_argument(
        "--aggregation",
        choices=["mean", "max"],
        default="mean",
        help="How to aggregate CETT over answer span (default: mean)",
    )
    resp_p.add_argument(
        "--output",
        default=None,
        help="Base path to save results (default: auto-named in cwd)",
    )
    _add_common_model_args(resp_p)
    _add_common_probe_args(resp_p)

    # ── hprobe transfer ───────────────────────────────────────────────────────
    transfer_p = subparsers.add_parser(
        "transfer", help="Score a saved probe on a different model (transfer experiment)"
    )
    transfer_p.add_argument(
        "--probe", required=True, help="Base path of saved probe (e.g. results/gemma_medqa)"
    )
    transfer_p.add_argument("--model", required=True, help="HuggingFace model ID for target model")
    transfer_p.add_argument("--data", required=True, help="Path to .jsonl or .json dataset file")
    transfer_p.add_argument(
        "--format",
        choices=["auto", "mmlu", "medqa", "medmcqa"],
        default="auto",
        help="Dataset format (default: auto-detect)",
    )
    transfer_p.add_argument(
        "--samples", type=int, default=100, help="Number of samples (default: 100)"
    )
    transfer_p.add_argument(
        "--output",
        default=None,
        help="Base path to save results (default: auto-named in cwd)",
    )
    transfer_p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        dest="max_tokens",
        help="Max input tokens before truncation (default: 1024)",
    )
    _add_common_model_args(transfer_p)

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "responses":
        cmd_responses(args)
    elif args.command == "transfer":
        cmd_transfer(args)


if __name__ == "__main__":
    main()
