# hprobes

Discover and causally validate hallucination-associated FFN neurons (**H-Neurons**) in transformer LLMs.

Based on [arXiv:2512.01797](https://arxiv.org/abs/2512.01797).

## What It Does

- Identifies a sparse set of FFN neurons whose CETT activations predict hallucination
- Validates them causally via activation scaling
- Provides production-ready hallucination risk scoring (`detect()` / `detect_batch()`)
- Calibrates decision threshold automatically using Youden's J

## Install

```bash
pip install hprobes
# or
uv add hprobes
```

## Quick Example

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
risk = probe.detect("Which of the following is true? A) ... B) ...")
print(f"Hallucination risk: {risk:.3f}")
```

## License

MIT
