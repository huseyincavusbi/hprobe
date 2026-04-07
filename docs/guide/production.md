# Production Scoring

After fitting a probe, you can score new prompts without ground truth. This page covers the `detect()` and `detect_batch()` methods for production hallucination risk estimation.

## How Detection Works

Detection requires **no generation** — only forward passes. For MCQ, the model never generates text. Instead:

1. One forward pass reads logits at the last prompt token
2. The highest-logit letter among A–J is the predicted answer
3. In contrastive mode, a second forward pass captures CETT at the predicted answer token
4. The trained classifier scores the CETT vector → hallucination probability

## Single Sample

```python
risk = probe.detect("Which organ is most affected? A) Heart B) Lung\n\nAnswer:")
print(f"Risk: {risk:.3f}")  # 0.0 = likely correct, 1.0 = likely hallucinating
```

If you already know the model's predicted letter (e.g. from your own inference pipeline), pass it to skip the first forward pass:

```python
risk = probe.detect(prompt, answer_letter="C")
```

!!! note
    The `answer_letter` parameter accepts letters A–J (case-insensitive). Passing an unrecognized letter raises `ValueError`.

## Batched Scoring

`detect_batch()` uses GPU-vectorized CETT extraction for throughput:

```python
prompts = [format_prompt(s) for s in samples]
scores = probe.detect_batch(prompts, batch_size=16)

# With pre-computed letters
letters = ["A", "C", "B", "D"]
scores = probe.detect_batch(prompts, answer_letters=letters, batch_size=16)
```

!!! tip
    Use `batch_size=8` or `batch_size=16` on GPU. Larger batches are faster but use more VRAM.

## Threshold-Based Decisions

After `score()`, the probe calibrates a threshold using Youden's J statistic:

```python
probe.score()
print(probe.threshold_)  # e.g. 0.63

risk = probe.detect(prompt)
is_hallucinating = risk >= probe.threshold_
```

The threshold is persisted through `save()`/`load()`:

```python
probe.save("results/probe")
loaded = HProbe.load("results/probe", model, tokenizer)
print(loaded.threshold_)  # same value
```

!!! warning
    The default threshold before calling `score()` is 0.5 (naive prior). Always call `score()` to calibrate before using in production.

## Consistency Filtering

Enable consistency filtering during `fit()` to skip ambiguous samples:

```python
probe = HProbe(model, tokenizer, n_consistency=10)
probe.fit(samples, options_key="choices", answer_key="answer")
```

With `n_consistency=10`:

1. For each sample, 10 draws from `softmax(letter_logits)` are made (single forward pass)
2. If all 10 agree on the same letter → sample is used
3. If draws disagree → sample is skipped (model is uncertain)

This filters out noisy training samples where the model's prediction is unstable, yielding cleaner H-Neuron discovery.

## Save and Load for Production

```python
# After fitting and scoring
probe.save("production/medqa_probe")

# Later, in your serving pipeline
from hprobes import HProbe
probe = HProbe.load("production/medqa_probe", model, tokenizer)

# Score incoming requests
for request in incoming:
    risk = probe.detect(request.prompt, answer_letter=request.predicted_letter)
    if risk >= probe.threshold_:
        flag_for_review(request)
```

## Performance Considerations

| Scenario | Forward Passes | Notes |
|----------|---------------|-------|
| `detect()` without `answer_letter` | 2 | Predicts letter internally |
| `detect()` with `answer_letter` | 1 | Fastest single-sample path |
| `detect_batch()` without `answer_letters` | 2 per batch | Batched letter prediction |
| `detect_batch()` with `answer_letters` | 1 per batch | Fastest batched path |
| Non-contrastive `detect()` | 1 | Always single pass |
