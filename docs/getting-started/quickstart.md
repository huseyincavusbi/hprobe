# Quick Start

## Fit a Probe

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobes import HProbe

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# samples: list of dicts with question, options (dict or list), and answer
probe = HProbe(model, tokenizer, l1_C=0.01)
probe.fit(samples, options_key="choices", answer_key="answer")

print(f"H-Neurons found: {probe.n_neurons_}")
print(f"Layer distribution: {probe.layer_distribution_}")
```

## Score and Validate

```python
results = probe.score()
print(f"AUROC:    {results['auroc']:.3f}")
print(f"Accuracy: {results['balanced_accuracy']:.3f}")
print(f"Threshold (Youden's J): {results['threshold']:.3f}")

# Causal validation: scale H-Neuron activations and measure accuracy change
cv = probe.causal_validate()
# {0.0: 0.15, 0.5: 0.22, 1.0: 0.30, 1.5: 0.28, 2.0: 0.25}
```

## Production Hallucination Detection

No ground truth required — one or two forward passes per sample.

```python
# Single sample
risk = probe.detect("Which organ is most affected? A) Heart B) Lung C) Liver D) Kidney\n\nAnswer:")
print(f"Risk: {risk:.3f}")  # 0 = likely correct, 1 = likely hallucinating

# With a pre-computed answer letter (saves one forward pass)
risk = probe.detect(prompt, answer_letter="C")

# Batched (GPU-efficient)
scores = probe.detect_batch(prompts, batch_size=8)
```

## Save and Load

```python
probe.save("results/gemma_medqa")          # writes .json + .pkl
probe = HProbe.load("results/gemma_medqa", model, tokenizer)

# Score on a new dataset with the loaded probe
probe.score_on(new_samples, options_key="choices", answer_key="answer")
```

## CLI

```bash
# Fit and score on an MCQ dataset
hprobes run --model google/gemma-3-4b-it --data dataset.jsonl --samples 500

# Transfer a saved probe to a different model
hprobes transfer --probe results/probe --model google/gemma-3-4b --data dataset.jsonl

# Fit from pre-generated responses with judge labels
hprobes responses --model google/gemma-3-4b-it --data responses.jsonl
```

## Output Files

Each `probe.save()` or CLI run produces two files:

| File | Contents |
|------|----------|
| `<name>.json` | Human-readable results: H-Neurons, AUROC, accuracy, causal validation |
| `<name>.pkl` | Serialized classifier, normalization stats, neuron list |
