"""HProbe — toolkit for H-Neuron discovery and causal validation."""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .cett import (
    available_layers,
    forward_cett,
    forward_cett_at_token,
    forward_cett_batch,
    forward_cett_span,
    precompute_col_norms,
    scale_h_neurons,
)

_NOT_FITTED_MSG = "Call fit() before using this method."
_MCQ_LETTERS = list("ABCDEFGHIJ")


class HProbe:
    """Discover and causally validate hallucination-associated FFN neurons in a transformer LLM.

    Implements the CETT metric to identify a sparse set of neurons whose activations
    predict whether the model will hallucinate, then validates them causally via
    activation scaling.

    Parameters
    ----------
    model : transformers CausalLM
        Any HuggingFace causal language model. Must already be loaded and on the
        correct device.
    tokenizer : transformers tokenizer
        Matching tokenizer for the model.
    l1_C : float
        Inverse L1 regularisation strength. Lower = sparser neuron set. Default 0.01.
    contrastive : bool
        If True (default), uses 3-vs-1 labeling: CETT captured at the generated answer token,
        hallucinatory answers labeled 1, everything else 0.
        If False, binary correct/incorrect labels at last prompt token.
    layer_stride : int
        Sample every Nth layer. 1 = all layers, 2 = even layers only (faster).
    validation_split : float
        Fraction of samples held out for scoring and causal validation.
    seed : int
    max_tokens : int
        Max input tokens before truncation.

    Attributes (set after fit)
    --------------------------
    h_neurons_ : list of (layer_idx, neuron_idx) tuples
    n_neurons_ : int
    neuron_ratio_ : float  — ratio in ‰ relative to total features
    layer_distribution_ : dict[int, int]
    accuracy_ : float  — model accuracy on the fit dataset
    is_fitted_ : bool
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        l1_C: float = 0.01,
        layer_stride: int = 1,
        validation_split: float = 0.2,
        seed: int = 42,
        max_tokens: int = 1024,
        batch_size: int = 1,
        n_consistency: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.l1_C = l1_C
        self.batch_size = batch_size
        self.layer_stride = layer_stride
        self.validation_split = validation_split
        self.seed = seed
        self.max_tokens = max_tokens
        self.n_consistency = n_consistency

        # Set after fit()
        self.h_neurons_: List[Tuple[int, int]] = []
        self.n_neurons_: int = 0
        self.neuron_ratio_: float = 0.0
        self.layer_distribution_: Dict[int, int] = {}
        self.accuracy_: float = 0.0
        self.threshold_: float = 0.5
        self.is_fitted_: bool = False

        # Internal state
        self._layers: List[int] = []
        self._col_norms: Dict[int, torch.Tensor] = {}
        self._intermediate_dim: int = 0
        self._n_features: int = 0
        self._top_k_idx: Optional[np.ndarray] = None
        self._col_mean: Optional[np.ndarray] = None
        self._col_std: Optional[np.ndarray] = None
        self._clf: Optional[LogisticRegression] = None
        self._letter_ids: Dict[str, int] = {}
        self._answer_cue: str = ""

        # Validation split cache
        self._val_prompts: List[str] = []
        self._val_gt: List[str] = []
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_train_cache: Optional[np.ndarray] = None
        self._y_train_cache: Optional[np.ndarray] = None

        # Results storage (set after score() / causal_validate())
        self.score_results_: Optional[Dict] = None
        self.cv_results_: Optional[Dict[float, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        samples: List[Dict],
        question_key: str = "question",
        options_key: str = "options",
        answer_key: str = "answer",
        prompt_fn: Optional[Callable[[Dict], str]] = None,
        answer_cue: str = "\n\nAnswer:",
    ) -> "HProbe":
        """Discover H-Neurons from MCQ samples.

        Parameters
        ----------
        samples : list of dict
            Each dict should contain at minimum a question and a ground-truth answer.
            For MCQ: {"question": "...", "options": {"A": "...", ...}, "answer": "A"}
        question_key : str
            Key for the question text in each sample dict. Default "question".
        options_key : str
            Key for the options dict in each sample dict. Default "options".
        answer_key : str
            Key for the ground-truth answer letter in each sample dict. Default "answer".
        prompt_fn : callable, optional
            Custom function that takes a sample dict and returns a formatted string.
            If None, uses tokenizer.apply_chat_template() with MCQ formatting.
        answer_cue : str
            String appended to every prompt to elicit a single-letter answer.
            Default "\\n\\nAnswer:".

        Returns
        -------
        self
        """
        self._answer_cue = answer_cue
        self._layers = available_layers(self.model)[:: self.layer_stride]
        self._col_norms = precompute_col_norms(self.model, self._layers)
        self._intermediate_dim = next(iter(self._col_norms.values())).shape[0]
        self._n_features = len(self._layers) * self._intermediate_dim
        self._letter_ids = self._get_letter_ids()
        top_k = min(5000, self._n_features)

        print(f"[hprobes] Layers: {len(self._layers)}  |  Features: {self._n_features:,}")

        # --- Phase 1: extract CETT features ---
        cett_raw, train_labels, row_to_sample, valid_prompts, valid_gt, per_sample = (
            self._extract_features(
                samples, question_key, options_key, answer_key, prompt_fn, answer_cue, top_k
            )
        )

        n_valid = len(valid_prompts)
        self.accuracy_ = sum(p["is_correct"] for p in per_sample) / n_valid if n_valid > 0 else 0.0

        print(f"[hprobes] Valid: {n_valid}  |  Accuracy: {self.accuracy_:.3f}")
        if n_valid < 20:
            print(f"  WARNING: only {n_valid} valid samples — probe may be unreliable.")

        # --- Variance pre-selection ---
        feature_var = self._welford_M2 / max(self._welford_n - 1, 1)
        self._top_k_idx = np.argsort(feature_var)[-top_k:]

        X = np.stack([v[self._top_k_idx] for v in cett_raw], axis=0)
        del cett_raw
        y = np.array(train_labels)

        self._col_mean = X.mean(axis=0)
        self._col_std = X.std(axis=0)
        self._col_std[self._col_std == 0] = 1.0
        X = (X - self._col_mean) / self._col_std

        # --- Train/val split at sample level ---
        # Ground-truth correctness labels per sample (for stratification)
        sample_correct = np.array([int(p["is_correct"]) for p in per_sample])
        sample_arr = np.arange(n_valid)
        can_strat = sample_correct.sum() > 1 and (n_valid - sample_correct.sum()) > 1
        train_s, val_s = train_test_split(
            sample_arr,
            test_size=self.validation_split,
            random_state=self.seed,
            stratify=sample_correct if can_strat else None,
        )
        train_set, val_set = set(train_s.tolist()), set(val_s.tolist())
        train_rows = np.array([i for i, si in enumerate(row_to_sample) if si in train_set])
        val_rows = np.array([i for i, si in enumerate(row_to_sample) if si in val_set])

        X_train, X_val = X[train_rows], X[val_rows]
        y_train, y_val = y[train_rows], y[val_rows]

        self._X_train_cache = X_train
        self._y_train_cache = y_train
        self._X_val = X_val
        self._y_val = y_val

        # Store val prompts + ground truth for causal_validate()
        self._val_prompts = [valid_prompts[i] for i in val_s]
        self._val_gt = [valid_gt[i] for i in val_s]

        # --- Phase 2: L1 probe ---
        self._clf = LogisticRegression(
            solver="liblinear",
            l1_ratio=1,
            C=self.l1_C,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.seed,
        )
        self._clf.fit(X_train, y_train)

        coef = self._clf.coef_[0]
        selected = np.where(coef > 0)[0]

        self.h_neurons_ = []
        for sel_idx in selected:
            flat_idx = int(self._top_k_idx[sel_idx])
            layer_pos = flat_idx // self._intermediate_dim
            neuron_pos = flat_idx % self._intermediate_dim
            if layer_pos < len(self._layers):
                self.h_neurons_.append((self._layers[layer_pos], int(neuron_pos)))

        self.n_neurons_ = len(self.h_neurons_)
        self.neuron_ratio_ = self.n_neurons_ / self._n_features * 1000
        self.layer_distribution_ = {}
        for li, _ in self.h_neurons_:
            self.layer_distribution_[li] = self.layer_distribution_.get(li, 0) + 1

        self.is_fitted_ = True

        print(f"[hprobes] H-Neurons: {self.n_neurons_}  |  Ratio: {self.neuron_ratio_:.3f}‰")
        if self.layer_distribution_:
            top = sorted(self.layer_distribution_.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"[hprobes] Top layers: {top}")

        return self

    def fit_from_responses(
        self,
        samples: List[Dict],
        question_key: str = "question",
        response_key: str = "response",
        answer_tokens_key: str = "answer_tokens",
        label_key: str = "judge",
        aggregation: str = "mean",
    ) -> "HProbe":
        """Discover H-Neurons from pre-generated responses (3-vs-1 labeling).

        Feeds the full Q+A sequence, captures CETT over the answer token span,
        aggregates with mean/max. Hallucinatory answer tokens=1, everything else=0.

        Parameters
        ----------
        samples : list of dict
            Each dict must contain question, response, answer_tokens (list of token
            strings marking the factual span), and a correctness label.
        question_key : str
            Key for the question string. Default "question".
        response_key : str
            Key for the generated response string. Default "response".
        answer_tokens_key : str
            Key for list of answer token strings (the factual span). Default "answer_tokens".
        label_key : str
            Key for the correctness label. Accepts "true"/"false" strings or 1/0 ints.
            Default "judge".
        aggregation : "mean" | "max"
            How to aggregate CETT over the answer token span. Default "mean".

        Returns
        -------
        self
        """
        self._layers = available_layers(self.model)[:: self.layer_stride]
        self._col_norms = precompute_col_norms(self.model, self._layers)
        self._intermediate_dim = next(iter(self._col_norms.values())).shape[0]
        self._n_features = len(self._layers) * self._intermediate_dim
        self._letter_ids = self._get_letter_ids()
        top_k = min(5000, self._n_features)

        print(
            f"[hprobes] Layers: {len(self._layers)}  |  Features: {self._n_features:,}  |  Mode: 3-vs-1"
        )

        self._welford_n = 0
        self._welford_mean = np.zeros(self._n_features, dtype=np.float64)
        self._welford_M2 = np.zeros(self._n_features, dtype=np.float64)

        cett_ans, cett_other, labels_ans, labels_other = [], [], [], []
        valid_prompts, valid_gt = [], []
        per_sample, skipped = [], 0

        for sample in tqdm(samples, desc="CETT extraction (responses)"):
            raw_label = sample.get(label_key)
            if raw_label is None:
                skipped += 1
                continue
            is_correct = str(raw_label).lower() in ("true", "1", "t")

            question = sample.get(question_key, "")
            response = sample.get(response_key, "")
            ans_tokens = sample.get(answer_tokens_key, [])

            # Build full Q+A prompt via chat template
            if (
                hasattr(self.tokenizer, "apply_chat_template")
                and self.tokenizer.chat_template is not None
            ):
                msgs = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ]
                full_text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            else:
                full_text = f"{question}\n{response}"

            tokens = self._tokenize(full_text)

            # Find answer token span in the tokenized sequence
            span = self._find_answer_span(tokens["input_ids"][0], ans_tokens)
            if span is None:
                skipped += 1
                continue
            span_start, span_end = span

            try:
                vec_ans = forward_cett_span(
                    self.model,
                    tokens,
                    span_start,
                    span_end,
                    self._layers,
                    self._col_norms,
                    aggregation,
                )
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                logging.warning(f"Error: {e}")
                skipped += 1
                continue

            valid_prompts.append(full_text)
            valid_gt.append("correct" if is_correct else "incorrect")
            per_sample.append({"is_correct": is_correct})

            ans = np.nan_to_num(vec_ans.numpy().astype(np.float32))
            cett_ans.append(ans)
            labels_ans.append(0 if is_correct else 1)  # 1 = hallucinatory
            self._welford_update(ans)

            # Other tokens: CETT at last prompt token (before answer span) — 3-vs-1
            try:
                vec_other, _ = forward_cett(
                    self.model, tokens, self._layers, self._col_norms, token_position=span_start - 1
                )
                oth = np.nan_to_num(vec_other.numpy().astype(np.float32))
                cett_other.append(oth)
                labels_other.append(0)  # always negative
                self._welford_update(oth)
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                logging.warning(f"Error: {e}")
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if skipped:
            print(f"[hprobes] Skipped: {skipped}")

        # Combine rows
        cett_all = cett_ans + cett_other
        y_all = labels_ans + labels_other
        row_to_sample = list(range(len(cett_ans))) + list(range(len(cett_ans)))

        n_valid = len(valid_prompts)
        self.accuracy_ = sum(p["is_correct"] for p in per_sample) / n_valid if n_valid > 0 else 0.0
        print(f"[hprobes] Valid: {n_valid}  |  Accuracy: {self.accuracy_:.3f}")

        # Variance pre-selection
        feature_var = self._welford_M2 / max(self._welford_n - 1, 1)
        self._top_k_idx = np.argsort(feature_var)[-top_k:]

        X = np.stack([v[self._top_k_idx] for v in cett_all], axis=0)
        y = np.array(y_all)
        self._col_mean = X.mean(axis=0)
        self._col_std = X.std(axis=0)
        self._col_std[self._col_std == 0] = 1.0
        X = (X - self._col_mean) / self._col_std

        # Sample-level train/val split
        sample_correct = np.array([int(p["is_correct"]) for p in per_sample])
        sample_arr = np.arange(n_valid)
        can_strat = sample_correct.sum() > 1 and (n_valid - sample_correct.sum()) > 1
        train_s, val_s = train_test_split(
            sample_arr,
            test_size=self.validation_split,
            random_state=self.seed,
            stratify=sample_correct if can_strat else None,
        )
        train_set, val_set = set(train_s.tolist()), set(val_s.tolist())
        train_rows = np.array([i for i, si in enumerate(row_to_sample) if si in train_set])
        val_rows = np.array([i for i, si in enumerate(row_to_sample) if si in val_set])

        X_train, X_val = X[train_rows], X[val_rows]
        y_train, y_val = y[train_rows], y[val_rows]
        self._X_train_cache, self._y_train_cache = X_train, y_train
        self._X_val, self._y_val = X_val, y_val
        self._val_prompts = [valid_prompts[i] for i in val_s]
        self._val_gt = [valid_gt[i] for i in val_s]

        self._clf = LogisticRegression(
            solver="liblinear",
            l1_ratio=1,
            C=self.l1_C,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.seed,
        )
        self._clf.fit(X_train, y_train)

        coef = self._clf.coef_[0]
        selected = np.where(coef > 0)[0]
        self.h_neurons_ = []
        for sel_idx in selected:
            flat_idx = int(self._top_k_idx[sel_idx])
            layer_pos = flat_idx // self._intermediate_dim
            neuron_pos = flat_idx % self._intermediate_dim
            if layer_pos < len(self._layers):
                self.h_neurons_.append((self._layers[layer_pos], int(neuron_pos)))

        self.n_neurons_ = len(self.h_neurons_)
        self.neuron_ratio_ = self.n_neurons_ / self._n_features * 1000
        self.layer_distribution_ = {}
        for li, _ in self.h_neurons_:
            self.layer_distribution_[li] = self.layer_distribution_.get(li, 0) + 1
        self.is_fitted_ = True

        print(f"[hprobes] H-Neurons: {self.n_neurons_}  |  Ratio: {self.neuron_ratio_:.3f}‰")
        if self.layer_distribution_:
            top = sorted(self.layer_distribution_.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"[hprobes] Top layers: {top}")

        return self

    def score(self) -> Dict:
        """Evaluate probe AUROC and random neuron baseline on the held-out val split.

        Returns
        -------
        dict with keys:
            auroc, balanced_accuracy,
            random_baseline_auroc, random_baseline_balanced_accuracy,
            auroc_gap, n_h_neurons, neuron_ratio_permille, threshold
        """
        if not self.is_fitted_:
            raise RuntimeError(_NOT_FITTED_MSG)

        X_val, y_val = self._X_val, self._y_val

        try:
            scores = self._clf.predict_proba(X_val)[:, 1]
            auroc = roc_auc_score(y_val, scores)
            fpr, tpr, thresholds = roc_curve(y_val, scores)
            J = tpr - fpr
            self.threshold_ = float(thresholds[int(J.argmax())])
        except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
            logging.warning(f"Error: {e}")
            auroc = None

        preds = self._clf.predict(X_val)
        bal_acc = balanced_accuracy_score(y_val, preds)

        # Random neuron baseline — same N neurons, same probe, same hyperparams
        rand_auroc, rand_bal_acc = None, None
        if self.n_neurons_ > 0:
            rng = np.random.RandomState(self.seed + 1)
            top_k = len(self._top_k_idx)
            rand_idx = rng.choice(top_k, size=min(self.n_neurons_, top_k), replace=False)

            clf_rand = LogisticRegression(
                solver="liblinear",
                l1_ratio=1,
                C=self.l1_C,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.seed,
            )
            clf_rand.fit(self._X_train_cache[:, rand_idx], self._y_train_cache)

            try:
                rand_scores = clf_rand.predict_proba(X_val[:, rand_idx])[:, 1]
                rand_auroc = roc_auc_score(y_val, rand_scores)
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                logging.warning(f"Error: {e}")
                pass
            rand_preds = clf_rand.predict(X_val[:, rand_idx])
            rand_bal_acc = balanced_accuracy_score(y_val, rand_preds)

        gap = (auroc - rand_auroc) if (auroc is not None and rand_auroc is not None) else None

        if gap is not None:
            print(f"[hprobes] AUROC: {auroc:.3f}  |  Random: {rand_auroc:.3f}  |  Gap: {gap:+.3f}")

        self.score_results_ = {
            "auroc": auroc,
            "balanced_accuracy": bal_acc,
            "random_baseline_auroc": rand_auroc,
            "random_baseline_balanced_accuracy": rand_bal_acc,
            "auroc_gap": gap,
            "n_h_neurons": self.n_neurons_,
            "neuron_ratio_permille": self.neuron_ratio_,
            "threshold": self.threshold_,
        }
        return self.score_results_

    def causal_validate(
        self,
        alphas: List[float] = None,
    ) -> Dict[float, float]:
        """Scale H-Neuron activations by each alpha and measure accuracy on val split.

        Default labeling:    suppression (alpha<1) lowers accuracy,
                             amplification (alpha>1) raises it.
        Contrastive labeling: direction is inverted.

        Returns
        -------
        dict mapping alpha → accuracy
        """
        if not self.is_fitted_:
            raise RuntimeError(_NOT_FITTED_MSG)
        if not self.h_neurons_:
            print("[hprobes] No H-Neurons found — skipping causal validation.")
            return {}

        alphas = alphas or [0.0, 0.5, 1.0, 1.5, 2.0]
        results = {}

        for alpha in alphas:
            correct, total = 0, 0
            for prompt, gt in zip(self._val_prompts, self._val_gt):
                tokens = self._tokenize(prompt)
                try:
                    logits = scale_h_neurons(
                        self.model, tokens, self.h_neurons_, alpha, self._layers
                    )
                    pred = self._predict_letter(logits)
                    correct += int(pred == gt)
                    total += 1
                except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                    logging.warning(f"Error: {e}")
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results[alpha] = correct / total if total > 0 else 0.0

        self.cv_results_ = results
        return results

    def save(self, path: str) -> Path:
        """Save probe results and classifier to disk.

        Writes two files:
        - ``<path>.json`` — human-readable results (neurons, scores, cv)
        - ``<path>.pkl``  — serialized classifier for transfer experiments

        Parameters
        ----------
        path : str
            Base path (e.g. "results/gemma_medqa"). Extensions are added automatically.

        Returns
        -------
        Path to the JSON file.
        """
        if not self.is_fitted_:
            raise RuntimeError(_NOT_FITTED_MSG)

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Ensure .json extension
        json_path = p.with_suffix(".json")

        out = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model": getattr(self, "model_id", None),
            "dataset": getattr(self, "dataset_name", None),
            "n_samples": getattr(self, "n_samples_used", None),
            "fit": {
                "n_h_neurons": self.n_neurons_,
                "neuron_ratio_permille": self.neuron_ratio_,
                "accuracy": self.accuracy_,
                "layer_distribution": {str(k): v for k, v in self.layer_distribution_.items()},
                "h_neurons": [list(n) for n in self.h_neurons_],
            },
        }
        if self.score_results_ is not None:
            out["score"] = self.score_results_
        if self.cv_results_ is not None:
            out["causal_validation"] = {str(k): v for k, v in self.cv_results_.items()}

        out["config"] = {
            "h_neurons": self.h_neurons_,
            "layers": self._layers,
            "intermediate_dim": self._intermediate_dim,
            "n_features": self._n_features,
            "l1_C": self.l1_C,
            "contrastive": self.contrastive,
            "layer_stride": self.layer_stride,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "answer_cue": self._answer_cue,
            "threshold": self.threshold_,
            "n_consistency": self.n_consistency,
            "clf_classes": self._clf.classes_.tolist() if hasattr(self._clf, "classes_") else [],
        }

        json_path.write_text(json.dumps(out, indent=2))

        # Save classifier state for transfer experiments
        tensors = {}
        if hasattr(self._clf, "coef_"):
            tensors["clf_coef"] = torch.tensor(self._clf.coef_)
        if hasattr(self._clf, "intercept_"):
            tensors["clf_intercept"] = torch.tensor(self._clf.intercept_)
        if self._top_k_idx is not None:
            tensors["top_k_idx"] = torch.tensor(self._top_k_idx)
        if self._col_mean is not None:
            tensors["col_mean"] = torch.tensor(self._col_mean)
        if self._col_std is not None:
            tensors["col_std"] = torch.tensor(self._col_std)

        sf_path = p.with_suffix(".safetensors")
        save_file(tensors, sf_path)

        return json_path

    @classmethod
    def load(cls, path: str, model: torch.nn.Module, tokenizer) -> "HProbe":
        """Load a saved probe classifier and attach it to a (possibly different) model.

        Use this to run transfer experiments: fit on an IT model, then load onto the
        corresponding PT base model to test whether H-Neurons transfer.

        Parameters
        ----------
        path : str
            Base path used when saving (e.g. "results/gemma_medqa"). Will look for
            ``<path>.pkl``.
        model : transformers CausalLM
            Model to attach the loaded probe to (can differ from the original).
        tokenizer :
            Matching tokenizer for the model.

        Returns
        -------
        HProbe instance ready for score_on() or causal_validate().
        """
        sf_path = Path(path).with_suffix(".safetensors")
        json_path = Path(path).with_suffix(".json")

        if not sf_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Saved probe missing .safetensors or .json at {path}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        if "config" not in metadata:
            raise ValueError(
                f"Invalid format: missing 'config' in {json_path}. Please re-fit the probe."
            )

        config = metadata["config"]
        tensors = load_file(sf_path)

        probe = cls(
            model=model,
            tokenizer=tokenizer,
            l1_C=config["l1_C"],
            contrastive=config["contrastive"],
            layer_stride=config["layer_stride"],
            seed=config["seed"],
            max_tokens=config["max_tokens"],
        )

        # Reconstruct the LogisticRegression model
        probe._clf = LogisticRegression(
            solver="liblinear",
            l1_ratio=1,
            C=config["l1_C"],
            class_weight="balanced",
            max_iter=1000,
            penalty="l1" if probe.contrastive else "elasticnet",
        )

        if "clf_coef" in tensors:
            probe._clf.coef_ = tensors["clf_coef"].numpy()
        if "clf_intercept" in tensors:
            probe._clf.intercept_ = tensors["clf_intercept"].numpy()
        if config.get("clf_classes"):
            probe._clf.classes_ = np.array(config["clf_classes"])

        probe._top_k_idx = tensors["top_k_idx"].numpy() if "top_k_idx" in tensors else None
        probe._col_mean = tensors["col_mean"].numpy() if "col_mean" in tensors else None
        probe._col_std = tensors["col_std"].numpy() if "col_std" in tensors else None

        probe.h_neurons_ = [(layer, neuron) for layer, neuron in config["h_neurons"]]
        probe._layers = config["layers"]
        probe._intermediate_dim = config["intermediate_dim"]
        probe._n_features = config["n_features"]
        probe._answer_cue = config["answer_cue"]
        probe.threshold_ = config.get("threshold", 0.5)
        probe.n_consistency = config.get("n_consistency", 1)
        probe.n_neurons_ = len(probe.h_neurons_)

        probe.layer_distribution_ = {}
        for layer, _ in probe.h_neurons_:
            probe.layer_distribution_[layer] = probe.layer_distribution_.get(layer, 0) + 1

        total = probe._n_features if probe._n_features > 0 else 1
        probe.neuron_ratio_ = (probe.n_neurons_ / total) * 1000
        probe._col_norms = precompute_col_norms(model, probe._layers)
        probe._letter_ids = probe._get_letter_ids()
        probe.is_fitted_ = True

        return probe

    def score_on(
        self,
        samples: List[Dict],
        question_key: str = "question",
        options_key: str = "options",
        answer_key: str = "answer",
        prompt_fn: Optional[Callable[[Dict], str]] = None,
    ) -> Dict:
        """Extract activations from the attached model and score with the loaded classifier.

        Used for transfer experiments: the classifier was fitted on a different model,
        and we test whether the same H-Neurons predict hallucination on this model.

        Parameters
        ----------
        samples : list of dict
            MCQ samples in the same format used during fit().
        question_key, options_key, answer_key, prompt_fn :
            Same as fit().

        Returns
        -------
        dict with auroc, balanced_accuracy, random_baseline_auroc, auroc_gap
        """
        if not self.is_fitted_:
            raise RuntimeError(_NOT_FITTED_MSG)

        X, y = [], []
        for sample in tqdm(samples, desc="CETT extraction (transfer)"):
            gt = self._parse_ground_truth(sample, answer_key)
            if gt is None:
                continue
            prompt = self._build_prompt(
                sample, question_key, options_key, prompt_fn, self._answer_cue
            )
            tokens = self._tokenize(prompt)
            try:
                cett_vec, logits = forward_cett(self.model, tokens, self._layers, self._col_norms)
                pred = self._predict_letter(logits)
                label = 0 if pred == gt else 1
                X.append(cett_vec.numpy())
                y.append(label)
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                logging.warning(f"Error: {e}")
                continue

        if not X:
            return {
                "auroc": None,
                "balanced_accuracy": None,
                "random_baseline_auroc": None,
                "auroc_gap": None,
            }

        X_arr = np.array(X)
        y_arr = np.array(y)

        # Normalise with the original training statistics
        if self._col_mean is not None and self._col_std is not None:
            X_norm = (X_arr[:, self._top_k_idx] - self._col_mean) / (self._col_std + 1e-8)
        else:
            X_norm = X_arr[:, self._top_k_idx]

        try:
            scores = self._clf.predict_proba(X_norm)[:, 1]
            auroc = roc_auc_score(y_arr, scores)
        except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
            logging.warning(f"Error: {e}")
            auroc = None

        preds = self._clf.predict(X_norm)
        bal_acc = balanced_accuracy_score(y_arr, preds)

        rand_auroc = None
        if self.n_neurons_ > 0:
            rng = np.random.RandomState(self.seed + 1)
            rand_idx = rng.choice(
                self._top_k_idx.shape[0],
                size=min(self.n_neurons_, self._top_k_idx.shape[0]),
                replace=False,
            )
            clf_rand = LogisticRegression(
                solver="liblinear",
                l1_ratio=1,
                C=self.l1_C,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.seed,
            )
            try:
                clf_rand.fit(X_norm[: len(X_norm) // 2, rand_idx], y_arr[: len(y_arr) // 2])
                rand_scores = clf_rand.predict_proba(X_norm[len(X_norm) // 2 :, rand_idx])[:, 1]
                rand_auroc = roc_auc_score(y_arr[len(y_arr) // 2 :], rand_scores)
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                logging.warning(f"Error: {e}")
                pass

        gap = (auroc - rand_auroc) if (auroc is not None and rand_auroc is not None) else None
        rand_str = f"{rand_auroc:.3f}" if rand_auroc is not None else "n/a"
        gap_str = f"{gap:+.3f}" if gap is not None else "n/a"
        print(f"[hprobes transfer] AUROC: {auroc:.3f}  |  Random: {rand_str}  |  Gap: {gap_str}")

        result = {
            "auroc": auroc,
            "balanced_accuracy": bal_acc,
            "random_baseline_auroc": rand_auroc,
            "auroc_gap": gap,
            "n_samples": len(X),
        }
        self.score_results_ = result
        return result

    def detect(
        self,
        prompt: str,
        answer_letter: Optional[str] = None,
    ) -> float:
        """Estimate hallucination risk for a single prompt (production inference).

        Runs one or two forward passes on ``prompt`` and returns a risk score
        using the fitted probe — no ground truth required.

        Parameters
        ----------
        prompt : str
            Fully formatted prompt string, including the answer cue
            (e.g. the output of ``tokenizer.apply_chat_template(...)`` + ``"\\n\\nAnswer:"``).
        answer_letter : str, optional
            The letter the model already predicted (e.g. ``"A"``).
            If provided, skips the first forward pass (faster — piggybacks on your
            existing generation call). If None, the probe runs its own forward pass
            to predict the letter.

        Returns
        -------
        float
            Hallucination risk score in ``[0, 1]``.
            Higher → model more likely to be wrong / hallucinating.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        ValueError
            If ``answer_letter`` is not a recognised MCQ letter (A–J).
        """
        if not self.is_fitted_:
            raise RuntimeError(_NOT_FITTED_MSG)

        tokens = self._tokenize(prompt)

        if self.contrastive:
            if answer_letter is None:
                _, logits = forward_cett(self.model, tokens, self._layers, self._col_norms)
                answer_letter = self._predict_letter(logits)
            else:
                answer_letter = answer_letter.strip().upper()

            letter_token_id = self._letter_ids.get(answer_letter)
            if letter_token_id is None:
                raise ValueError(
                    f"Unknown answer letter {answer_letter!r}. "
                    f"Expected one of {list(self._letter_ids)}"
                )
            cett_vec = forward_cett_at_token(
                self.model, tokens, letter_token_id, self._layers, self._col_norms
            )
        else:
            cett_vec, _ = forward_cett(self.model, tokens, self._layers, self._col_norms)

        x = np.nan_to_num(cett_vec.numpy().astype(np.float32))
        x_sel = x[self._top_k_idx]
        x_norm = (x_sel - self._col_mean) / (self._col_std + 1e-8)
        return float(self._clf.predict_proba(x_norm.reshape(1, -1))[0, 1])

    def detect_batch(
        self,
        prompts: List[str],
        answer_letters: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Estimate hallucination risk for a batch of prompts.

        Batched version of :meth:`detect` — uses the same GPU-vectorized CETT
        extraction as ``fit()``, so throughput scales with batch size.

        Parameters
        ----------
        prompts : list of str
            Fully formatted prompt strings including the answer cue.
        answer_letters : list of str, optional
            Predicted answer letter per prompt (e.g. ``["A", "C", "B"]``).
            If provided, skips the first forward pass for the whole batch.
            If None, the probe runs a batched forward pass to predict all letters.
        batch_size : int, optional
            Number of prompts per forward pass. Defaults to ``self.batch_size``.

        Returns
        -------
        list of float
            Hallucination risk score in ``[0, 1]`` for each prompt, in the same
            order as ``prompts``.
        """
        if not self.is_fitted_:
            raise RuntimeError(_NOT_FITTED_MSG)

        bs = batch_size or self.batch_size
        device = next(self.model.parameters()).device
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        all_scores: List[float] = []

        try:
            for start in range(0, len(prompts), bs):
                batch_prompts = prompts[start : start + bs]

                enc = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                ).to(device)

                if "attention_mask" in enc:
                    last_positions = (enc["attention_mask"].sum(dim=1) - 1).tolist()
                else:
                    last_positions = [enc["input_ids"].shape[1] - 1] * len(batch_prompts)

                cett_matrix, _ = forward_cett_batch(
                    self.model,
                    enc,
                    self._layers,
                    self._col_norms,
                    [int(p) for p in last_positions],
                )
                scores_batch = []
                for i in range(len(batch_prompts)):
                    x = np.nan_to_num(cett_matrix[i].numpy().astype(np.float32))
                    x_norm = (x[self._top_k_idx] - self._col_mean) / (self._col_std + 1e-8)
                    scores_batch.append(float(self._clf.predict_proba(x_norm.reshape(1, -1))[0, 1]))

                all_scores.extend(scores_batch)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            self.tokenizer.padding_side = orig_padding_side

        return all_scores

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        sample: Dict,
        question_key: str,
        options_key: str,
        prompt_fn: Optional[Callable],
        answer_cue: str,
    ) -> str:
        """Build a formatted prompt string from a sample dict.

        If prompt_fn is provided, uses it directly.
        Otherwise, uses tokenizer.apply_chat_template() with MCQ formatting.
        """
        if prompt_fn is not None:
            content = prompt_fn(sample)
        else:
            q = sample.get(question_key, "")
            opts = sample.get(options_key, {})
            if isinstance(opts, str):
                import ast

                try:
                    opts = ast.literal_eval(opts)
                except (
                    ValueError,
                    KeyError,
                    RuntimeError,
                    IndexError,
                    TypeError,
                    SyntaxError,
                ) as e:
                    logging.warning(f"Error evaluating ast literal: {e}")
                    opts = {}
            if isinstance(opts, list):
                # HuggingFace datasets like MMLU use a list of choice texts
                opts = {_MCQ_LETTERS[i]: v for i, v in enumerate(opts)}
            if opts:
                choices = "\n".join(f"{k}. {v}" for k, v in opts.items())
                content = f"{q}\n{choices}"
            else:
                content = q

        # Use model's own chat template if available, else plain text
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            messages = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = content

        return prompt + answer_cue

    def _parse_ground_truth(self, sample: Dict, answer_key: str) -> Optional[str]:
        """Extract ground-truth answer letter from sample dict.

        Handles:
            - Single letter: "A", "B", ...
            - Numeric index: 0→A, 1→B, ... (common in HuggingFace MCQ datasets)
            - List wrapping: ["A"] or ["Ans. The key is B. ..."] (e.g. PLAB)
            - Free-text: "Ans. The key is B. ..." → extracts first A-J letter after
              common answer cue phrases
        """
        import re

        raw = sample.get(answer_key)
        if raw is None:
            return None
        # Unwrap single-element lists
        if isinstance(raw, list):
            raw = raw[0] if raw else None
            if raw is None:
                return None
        raw = str(raw).strip()
        if raw.upper() in _MCQ_LETTERS:
            return raw.upper()
        if raw.isdigit() and int(raw) < len(_MCQ_LETTERS):
            return _MCQ_LETTERS[int(raw)]
        # Free-text fallback: "The key is B", "answer is C", "key: D", etc.
        m = re.search(r"(?:key\s+is|answer\s+is|key\s*:)\s*([A-J])\b", raw, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    # ------------------------------------------------------------------
    # Tokenizer / logit helpers
    # ------------------------------------------------------------------

    def _get_letter_ids(self) -> Dict[str, int]:
        """Map MCQ letters → single token id each."""
        letter_ids = {}
        for letter in _MCQ_LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if ids:
                letter_ids[letter] = ids[0]
        return letter_ids

    def _predict_letter(self, logits: torch.Tensor) -> str:
        """Pick the MCQ letter with the highest logit."""
        return max(self._letter_ids.items(), key=lambda kv: logits[kv[1]].item())[0]

    def _consistency_predict(self, tokens: Dict[str, torch.Tensor], n: int) -> Optional[str]:
        """Sample n predictions from the letter distribution and return the agreed letter.

        Runs a single forward pass, restricts logits to MCQ letter tokens, then
        draws n samples with temperature=1.0. Returns the agreed letter if all n
        samples match, else None (inconsistent → skip the sample).
        """
        letters = list(self._letter_ids.keys())
        token_ids = torch.tensor(list(self._letter_ids.values()))

        with torch.no_grad():
            out = self.model(**tokens)

        logits = out.logits[0, -1, :].float().cpu()
        letter_logits = logits[token_ids]  # (n_letters,) — temperature=1.0 (no scaling)
        probs = torch.softmax(letter_logits, dim=0)

        indices = torch.multinomial(probs, num_samples=n, replacement=True).tolist()
        preds = [letters[i] for i in indices]
        return preds[0] if len(set(preds)) == 1 else None

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        return {k: v.to(device) for k, v in tokens.items()}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        samples: List[Dict],
        question_key: str,
        options_key: str,
        answer_key: str,
        prompt_fn: Optional[Callable],
        answer_cue: str,
        top_k: int,
    ):
        """Extract CETT feature vectors for all samples.

        Runs inference internally to:
          1. Predict the model's answer letter
          2. Compare to ground truth → correctness label
          3. Extract CETT in the same forward pass

        Returns
        -------
        cett_raw, train_labels, row_to_sample, valid_prompts, valid_gt, per_sample
        """
        self._welford_n = 0
        self._welford_mean = np.zeros(self._n_features, dtype=np.float64)
        self._welford_M2 = np.zeros(self._n_features, dtype=np.float64)

        cett_raw, train_labels, row_to_sample = [], [], []
        valid_prompts, valid_gt = [], []
        per_sample = []
        skipped = 0

        device = next(self.model.parameters()).device

        if self.batch_size > 1:
            orig_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"

        batch_buf: list = []  # list of (sample, gt, prompt) waiting to be processed

        def _flush_batch(buf):
            nonlocal skipped
            if not buf:
                return

            prompts = [b[2] for b in buf]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
            ).to(device)

            # last real token position per sample (for logits)
            if "attention_mask" in enc:
                last_positions = (enc["attention_mask"].sum(dim=1) - 1).tolist()
            else:
                last_positions = [enc["input_ids"].shape[1] - 1] * len(buf)

            try:
                cett_matrix, logits_matrix = forward_cett_batch(
                    self.model,
                    enc,
                    self._layers,
                    self._col_norms,
                    [int(p) for p in last_positions],
                )
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                logging.warning(f"Error: {e}")
                skipped += len(buf)
                return

            for i, (sample, gt, prompt) in enumerate(buf):
                pred = self._predict_letter(logits_matrix[i])
                is_correct = pred == gt
                sample_pos = len(valid_prompts)
                valid_prompts.append(prompt)
                valid_gt.append(gt)
                per_sample.append({"predicted": pred, "ground_truth": gt, "is_correct": is_correct})

                vec = np.nan_to_num(cett_matrix[i].numpy().astype(np.float32))
                cett_raw.append(vec)
                train_labels.append(int(is_correct))
                row_to_sample.append(sample_pos)
                self._welford_update(vec)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for sample in tqdm(samples, desc="CETT extraction"):
            gt = self._parse_ground_truth(sample, answer_key)
            if gt is None:
                skipped += 1
                continue

            prompt = self._build_prompt(sample, question_key, options_key, prompt_fn, answer_cue)

            if self.batch_size > 1 and self.n_consistency <= 1:
                batch_buf.append((sample, gt, prompt))
                if len(batch_buf) >= self.batch_size:
                    _flush_batch(batch_buf)
                    batch_buf = []
                continue

            # --- single-sample path ---
            tokens = self._tokenize(prompt)

            if self.n_consistency > 1:
                pred = self._consistency_predict(tokens, self.n_consistency)
                if pred is None:
                    skipped += 1
                    continue
                try:
                    cett_vec, _ = forward_cett(self.model, tokens, self._layers, self._col_norms)
                except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                    logging.warning(f"Error: {e}")
                    skipped += 1
                    continue
            else:
                try:
                    cett_vec, logits = forward_cett(
                        self.model, tokens, self._layers, self._col_norms
                    )
                except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as e:
                    logging.warning(f"Error: {e}")
                    skipped += 1
                    continue
                pred = self._predict_letter(logits)
            is_correct = pred == gt

            sample_pos = len(valid_prompts)
            valid_prompts.append(prompt)
            valid_gt.append(gt)
            per_sample.append(
                {
                    "predicted": pred,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                }
            )

            vec = np.nan_to_num(cett_vec.numpy().astype(np.float32))
            cett_raw.append(vec)
            train_labels.append(int(is_correct))
            row_to_sample.append(sample_pos)
            self._welford_update(vec)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # flush remaining batch
        if batch_buf:
            _flush_batch(batch_buf)

        if self.batch_size > 1:
            self.tokenizer.padding_side = orig_padding_side

        if skipped:
            print(f"[hprobes] Skipped: {skipped}")

        return cett_raw, train_labels, row_to_sample, valid_prompts, valid_gt, per_sample

    def _find_answer_span(
        self, input_ids: torch.Tensor, answer_tokens: List[str]
    ) -> Optional[Tuple[int, int]]:
        """Find the contiguous span of answer tokens in the tokenized sequence.

        Normalizes ▁ (SentencePiece) and Ġ (BPE) word boundary markers before
        matching.

        Returns (start, end) indices (end exclusive), or None if not found.
        """
        if not answer_tokens:
            return None

        full_tokens = [self.tokenizer.decode([tid]) for tid in input_ids]
        ans_norm = [t.replace("▁", " ").replace("Ġ", " ") for t in answer_tokens]
        m = len(ans_norm)

        for i in range(len(full_tokens) - m + 1):
            window = [full_tokens[j].replace("▁", " ").replace("Ġ", " ") for j in range(i, i + m)]
            if window == ans_norm:
                return i, i + m
        return None

    def _welford_update(self, vec: np.ndarray):
        self._welford_n += 1
        delta = vec.astype(np.float64) - self._welford_mean
        self._welford_mean += delta / self._welford_n
        self._welford_M2 += delta * (vec.astype(np.float64) - self._welford_mean)
