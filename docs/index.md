# hprobes

Discover and causally validate hallucination-associated FFN neurons (**H-Neurons**) in transformer LLMs.

Based on [arXiv:2512.01797](https://arxiv.org/abs/2512.01797).

## What It Does

- Identifies a sparse set of FFN neurons whose CETT activations predict hallucination
- Validates them causally via activation scaling
- Provides production-ready hallucination risk scoring (`detect()` / `detect_batch()`)
- Calibrates decision thresholds automatically using Youden's J statistic

## Install

```bash
pip install hprobes
# or
uv add hprobes
```

## Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobes import HProbe

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

probe = HProbe(model, tokenizer)
probe.fit(samples, options_key="choices", answer_key="answer")

results = probe.score()
print(f"AUROC {results['auroc']:.3f}  threshold {results['threshold']:.3f}")

# Production scoring — no ground truth needed
risk = probe.detect("Which organ is most affected? A) Heart B) Lung C) Liver D) Kidney\n\nAnswer:")
```

## Pipeline Overview

| Step | Method | What It Does |
|------|--------|--------------|
| Discover | `fit()` | Extract CETT features, train L1 logistic regression, select H-Neurons |
| Evaluate | `score()` | AUROC on held-out validation split + random neuron baseline |
| Validate | `causal_validate()` | Scale H-Neuron activations to confirm causal role |
| Detect | `detect()` / `detect_batch()` | Production hallucination risk scoring (no labels needed) |
| Transfer | `score_on()` | Score a saved probe on a different model or dataset |

## CLI

```bash
hprobes run --model google/gemma-3-4b-it --data dataset.jsonl --samples 500
hprobes transfer --probe results/probe --model google/gemma-3-4b --data dataset.jsonl
hprobes responses --model google/gemma-3-4b-it --data responses.jsonl
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `l1_C` | `0.01` | Inverse L1 strength — lower = fewer neurons |
| `contrastive` | `True` | 3-vs-1 labeling at the generated answer token |
| `layer_stride` | `1` | Sample every Nth layer (2 = faster) |
| `batch_size` | `1` | GPU batch size for CETT extraction |
| `n_consistency` | `1` | Consistency filter draws (1 = disabled) |

## License

MIT
