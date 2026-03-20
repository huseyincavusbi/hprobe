# hprobe

Discover and causally validate hallucination-associated FFN neurons in transformer LLMs.

## Install

```bash
uv add hprobe
```

## Usage

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobe import HProbe

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# Any HuggingFace MCQ dataset works directly
samples = load_dataset("cais/mmlu", "clinical_knowledge", split="test")

probe = HProbe(model, tokenizer)
probe.fit(samples, options_key="choices", answer_key="answer")

print(probe.n_neurons_, probe.layer_distribution_)

results = probe.score()
print(f"AUROC {results['auroc']:.3f}  gap {results['auroc_gap']:+.3f}")

probe.causal_validate()
```

## Reference

[arXiv:2512.01797](https://arxiv.org/abs/2512.01797)
