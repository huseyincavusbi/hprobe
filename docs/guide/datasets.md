# Dataset Formats

## Supported File Formats

hprobes accepts three file formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| JSON Lines | `.jsonl` | One JSON object per line (recommended) |
| JSON | `.json` | Array of objects |
| Parquet | `.parquet` | Apache Parquet columnar format |

## MCQ Sample Structure

Each sample is a dict with a question, answer options, and a ground-truth answer:

```json
{
  "question": "Which organ is most commonly affected in sarcoidosis?",
  "options": {"A": "Heart", "B": "Lung", "C": "Liver", "D": "Kidney"},
  "answer": "B"
}
```

Options can also be a list (letters are assigned automatically):

```json
{
  "question": "Which organ is most commonly affected?",
  "options": ["Heart", "Lung", "Liver", "Kidney"],
  "answer": "B"
}
```

## Auto-Detected Formats

The CLI auto-detects three common MCQ dataset layouts:

| Format | Options Key | Answer Key | Source |
|--------|-------------|------------|--------|
| `mmlu` | `choices` | `answer` | MMLU / MMLU-Pro |
| `medqa` | `options` | `answer_idx` | MedQA |
| `medmcqa` | `options` | `cop` | MedMCQA |

Auto-detection works by inspecting the keys in the first sample. Override with `--format`:

```bash
hprobes run --model google/gemma-3-4b-it --data dataset.jsonl --format mmlu
```

In Python, pass the keys directly:

```python
probe.fit(samples, options_key="choices", answer_key="answer")
```

## Answer Formats

The `answer` field is flexible. hprobes handles:

| Input | Parsed As |
|-------|-----------|
| `"A"` | A |
| `"a"` | A |
| `0` | A (zero-indexed) |
| `1` | B (one-indexed for medmcqa) |
| `["A"]` | A (list unwrapped) |
| `"Ans. The key is B..."` | B (free-text extraction) |

## Response-Based Format

For open-ended / free-text mode with `fit_from_responses()`:

```json
{
  "question": "What is the most common cause of community-acquired pneumonia?",
  "response": "The most common cause is Streptococcus pneumoniae, which accounts for...",
  "answer_tokens": ["Streptococcus", "pneumoniae"],
  "judge": "true"
}
```

| Key | Description |
|-----|-------------|
| `question` | The input question |
| `response` | The model's full generated response |
| `answer_tokens` | List of token strings marking the factual answer span |
| `judge` | Correctness label: `"true"`/`"false"` or `1`/`0` |

## Custom Prompt Functions

For non-standard formats, pass a custom `prompt_fn`:

```python
def my_formatter(sample):
    q = sample["query"]
    opts = "\n".join(f"{k}) {v}" for k, v in sample["choices"].items())
    return f"{q}\n{opts}"

probe.fit(samples, prompt_fn=my_formatter, answer_key="correct")
```
