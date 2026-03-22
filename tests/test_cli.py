"""Unit tests for hprobe CLI helpers"""

import json
import tempfile

from hprobe.cli import detect_format, format_keys, load_samples


_MMLU_SAMPLE = {
    "question": "Which organ produces insulin?",
    "choices": ["Liver", "Pancreas", "Kidney", "Spleen"],
    "answer": 1,
    "subject": "anatomy",
}

_MEDQA_SAMPLE = {
    "question": "A patient presents with chest pain. What is the first test?",
    "options": {"A": "ECG", "B": "CXR", "C": "Echo", "D": "CT"},
    "answer_idx": "A",
    "meta_info": "step2",
}

_MEDMCQA_SAMPLE = {
    "question": "Drug of choice for TB meningitis?",
    "options": {"A": "Rifampicin", "B": "INH", "C": "Pyrazinamide", "D": "Ethambutol"},
    "cop": 0,
    "exp": "Rifampicin crosses BBB",
}

_UNKNOWN_SAMPLE = {
    "question": "What is 2+2?",
    "answer": "4",
}


class TestDetectFormat:
    def test_mmlu(self):
        assert detect_format(_MMLU_SAMPLE) == "mmlu"

    def test_medqa(self):
        assert detect_format(_MEDQA_SAMPLE) == "medqa"

    def test_medmcqa(self):
        assert detect_format(_MEDMCQA_SAMPLE) == "medmcqa"

    def test_unknown_returns_none(self):
        assert detect_format(_UNKNOWN_SAMPLE) is None

    def test_mmlu_requires_int_answer(self):
        # choices + string answer should NOT be detected as mmlu
        s = {**_MMLU_SAMPLE, "answer": "B"}
        assert detect_format(s) != "mmlu"


class TestFormatKeys:
    def test_mmlu_keys(self):
        assert format_keys("mmlu") == ("choices", "answer")

    def test_medqa_keys(self):
        assert format_keys("medqa") == ("options", "answer_idx")

    def test_medmcqa_keys(self):
        assert format_keys("medmcqa") == ("options", "cop")

    def test_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown format"):
            format_keys("foobar")


class TestLoadSamples:
    def test_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for s in [_MMLU_SAMPLE, _MEDQA_SAMPLE, _MEDMCQA_SAMPLE]:
                f.write(json.dumps(s) + "\n")
            path = f.name

        samples = load_samples(path, n=10)
        assert len(samples) == 3
        assert samples[0]["question"] == _MMLU_SAMPLE["question"]

    def test_jsonl_respects_n(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for _ in range(20):
                f.write(json.dumps(_MMLU_SAMPLE) + "\n")
            path = f.name

        samples = load_samples(path, n=5)
        assert len(samples) == 5

    def test_json_list(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([_MMLU_SAMPLE, _MEDQA_SAMPLE], f)
            path = f.name

        samples = load_samples(path, n=10)
        assert len(samples) == 2

    def test_missing_file_exits(self):
        import pytest

        with pytest.raises(SystemExit):
            load_samples("/nonexistent/path.jsonl", n=10)

    def test_unsupported_extension_exits(self):
        import pytest

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("a,b,c\n")
            path = f.name

        with pytest.raises(SystemExit):
            load_samples(path, n=10)
