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
    """Load up to n samples from a JSONL or JSON file."""
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
    else:
        print(f"Error: unsupported file format '{p.suffix}'. Use .jsonl or .json", file=sys.stderr)
        sys.exit(1)

    if not samples:
        print("Error: no samples loaded from file", file=sys.stderr)
        sys.exit(1)

    return samples


def _default_output_path(model: str, dataset_path: str) -> str:
    """Build a default output filename: {model_safe}_{dataset}_{timestamp}.json"""
    model_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
    dataset_name = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_safe}_{dataset_name}_{ts}.json"


def cmd_run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hprobe import HProbe, __version__

    sep = "─" * 68
    print(f"\nhprobe v{__version__}  |  model: {args.model}  |  samples: {args.samples}")
    print(sep)

    # Load dataset
    samples = load_samples(args.data, args.samples)

    # Resolve format
    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(samples[0])
        if fmt is None:
            print(
                "  Warning: could not auto-detect format. "
                "Falling back to options_key='options', answer_key='answer'.",
                file=sys.stderr,
            )
            options_key, answer_key = "options", "answer"
        else:
            options_key, answer_key = format_keys(fmt)
    else:
        options_key, answer_key = format_keys(fmt)

    print(f"  Dataset:  {Path(args.data).name}  ({len(samples)} samples, format={fmt or 'auto'})")
    print(f"  Keys:     options_key={options_key!r}  answer_key={answer_key!r}")

    # Resolve dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load model
    print(f"  Loading {args.model}...", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()
    print(" done")

    # Fit
    print(f"  Fitting (l1_C={args.l1_c})...", end="", flush=True)
    probe = HProbe(model, tokenizer, l1_C=args.l1_c, seed=args.seed)
    probe.fit(samples, options_key=options_key, answer_key=answer_key)
    print(" done")
    print(f"  H-Neurons:  {probe.n_neurons_}  ({probe.neuron_ratio_:.3f}‰ of all features)")
    print(f"  Accuracy:   {probe.accuracy_:.3f}")
    print(f"  Layers:     {dict(sorted(probe.layer_distribution_.items()))}")

    # Score
    print("\n  Scoring...")
    results = probe.score()

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

    # Causal validate
    print("\n  Causal validation (alpha → accuracy):")
    cv = probe.causal_validate()
    for alpha, acc in sorted(cv.items()):
        tag = ""
        if alpha == 0.0:
            tag = "  ← full suppression"
        elif alpha == 1.0:
            tag = "  ← baseline"
        elif alpha == 2.0:
            tag = "  ← amplification"
        print(f"    {alpha:.1f} → {acc:.3f}{tag}")

    # Save results
    out_path = args.output or _default_output_path(args.model, args.data)
    saved = probe.save(out_path)
    print(f"  Saved → {saved}")
    print(sep + "\n")


def main() -> None:
    from hprobe import __version__

    parser = argparse.ArgumentParser(
        prog="hprobe",
        description="Hallucination neuron probe — discover and causally validate H-Neurons",
    )
    parser.add_argument("--version", action="version", version=f"hprobe {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Fit, score, and causal-validate on a dataset")
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
        "--device", default="auto", help="Device: auto, cpu, mps, cuda (default: auto)"
    )
    run_p.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype (default: auto)",
    )
    run_p.add_argument(
        "--l1-c",
        type=float,
        default=0.5,
        dest="l1_c",
        help="Inverse L1 regularisation strength (default: 0.5)",
    )
    run_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_p.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (default: auto-named in cwd)",
    )

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
