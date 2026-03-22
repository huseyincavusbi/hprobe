"""Tests for hprobe.probe — HProbe class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hprobe import HProbe

# ---------------------------------------------------------------------------
# Minimal mock model + tokenizer
# ---------------------------------------------------------------------------

_H, _I, _V, _L = 8, 16, 128, 4


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(_H, _I, bias=False)
        self.down_proj = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate(x)))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x):
        return x + self.mlp(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block() for _ in range(_L)])

        self.model = _Inner()
        self.lm_head = nn.Linear(_H, _V, bias=True)
        nn.init.normal_(self.lm_head.weight, std=0.2)
        nn.init.normal_(self.lm_head.bias, std=0.2)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        # Use raw token ids as embedding values so different tokens produce different outputs
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, _H)
        for block in self.model.layers:
            x = block(x)

        class _Out:
            pass

        out = _Out()
        out.logits = self.lm_head(x)
        return out


class _Tokenizer:
    """ASCII character-level tokenizer. encode('A') == [65]."""

    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        if max_length:
            ids = ids[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


MODEL = _Model()
TOK = _Tokenizer()

# 20 samples: 10 answer A (index 0), 10 answer B (index 1)
SAMPLES = [
    {"question": f"Q{i}?", "options": ["alpha", "beta", "gamma", "delta"], "answer": i % 2}
    for i in range(20)
]

# Fitted probe shared across tests in this module
_PROBE = HProbe(MODEL, TOK, l1_C=0.5)
_PROBE.fit(SAMPLES, options_key="options", answer_key="answer")


# ---------------------------------------------------------------------------


class TestParseGroundTruth:
    def test_letter(self):
        assert _PROBE._parse_ground_truth({"a": "B"}, "a") == "B"

    def test_lowercase_normalised(self):
        assert _PROBE._parse_ground_truth({"a": "c"}, "a") == "C"

    def test_numeric_index(self):
        assert _PROBE._parse_ground_truth({"a": 0}, "a") == "A"
        assert _PROBE._parse_ground_truth({"a": 3}, "a") == "D"

    def test_missing_returns_none(self):
        assert _PROBE._parse_ground_truth({}, "answer") is None

    def test_invalid_returns_none(self):
        assert _PROBE._parse_ground_truth({"a": "xyz"}, "a") is None


class TestFindAnswerSpan:
    def _ids(self, text):
        return torch.tensor([ord(c) for c in text])

    def test_found(self):
        assert _PROBE._find_answer_span(self._ids("Hello"), ["l", "l"]) == (2, 4)

    def test_not_found(self):
        assert _PROBE._find_answer_span(self._ids("Hello"), ["X"]) is None

    def test_empty_returns_none(self):
        assert _PROBE._find_answer_span(self._ids("Hello"), []) is None


class TestPredictLetter:
    def test_picks_max_logit(self):
        logits = torch.zeros(_V)
        logits[ord("C")] = 10.0
        assert _PROBE._predict_letter(logits) == "C"


class TestWelfordUpdate:
    def test_mean_converges(self):
        p = HProbe(MODEL, TOK)
        p._welford_n = 0
        p._welford_mean = np.zeros(8, dtype=np.float64)
        p._welford_M2 = np.zeros(8, dtype=np.float64)
        rng = np.random.RandomState(0)
        data = [rng.randn(8).astype(np.float32) for _ in range(100)]
        for v in data:
            p._welford_update(v)
        assert np.allclose(p._welford_mean, np.stack(data).mean(0), atol=1e-4)


class TestNotFittedErrors:
    def test_score_raises(self):
        with pytest.raises(RuntimeError):
            HProbe(MODEL, TOK).score()

    def test_causal_validate_raises(self):
        with pytest.raises(RuntimeError):
            HProbe(MODEL, TOK).causal_validate()


class TestFitAndScore:
    def test_fitted_attributes(self):
        assert _PROBE.is_fitted_
        assert 0.0 <= _PROBE.accuracy_ <= 1.0
        assert _PROBE.n_neurons_ == len(_PROBE.h_neurons_)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in _PROBE.h_neurons_)

    def test_score_keys_and_range(self):
        result = _PROBE.score()
        assert "auroc" in result and "balanced_accuracy" in result and "auroc_gap" in result
        assert 0.0 <= result["balanced_accuracy"] <= 1.0

    def test_causal_validate_range(self):
        result = _PROBE.causal_validate(alphas=[0.0, 1.0])
        # Empty dict is valid when no H-Neurons were found
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_contrastive_mode(self):
        p = HProbe(MODEL, TOK, l1_C=0.5, contrastive=True)
        p.fit(SAMPLES, options_key="options", answer_key="answer")
        assert p.is_fitted_

    def test_mmlu_list_options(self):
        samples = [
            {"question": f"Q{i}?", "choices": ["a", "b", "c", "d"], "answer": i % 2}
            for i in range(20)
        ]
        p = HProbe(MODEL, TOK, l1_C=0.5)
        p.fit(samples, options_key="choices", answer_key="answer")
        assert p.is_fitted_

    def test_fit_from_responses(self):
        samples = [
            {
                "question": f"Describe {i}.",
                "response": f"The answer is X{i}.",
                "answer_tokens": ["X"],
                "judge": i % 2 == 0,
            }
            for i in range(20)
        ]
        p = HProbe(MODEL, TOK, l1_C=0.5)
        p.fit_from_responses(samples)
        assert p.is_fitted_


class TestSaveLoadTransfer:
    def test_save_creates_json_and_pkl(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.score()
            saved = _PROBE.save(base)
            assert Path(base).with_suffix(".json").exists()
            assert Path(base).with_suffix(".pkl").exists()
            assert saved == Path(base).with_suffix(".json")

    def test_load_restores_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)
            loaded = HProbe.load(base, MODEL, TOK)
            assert loaded.is_fitted_
            assert loaded.n_neurons_ == _PROBE.n_neurons_
            assert loaded.h_neurons_ == _PROBE.h_neurons_

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            HProbe.load("/nonexistent/probe", MODEL, TOK)

    def test_score_on_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)
            loaded = HProbe.load(base, MODEL, TOK)
            result = loaded.score_on(SAMPLES, options_key="options", answer_key="answer")
            assert "auroc" in result
            assert "balanced_accuracy" in result
