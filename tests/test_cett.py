"""Tests for hprobes.cett — CETT metric and hook utilities."""

import pytest
import torch
import torch.nn as nn

from hprobes.cett import (
    _get_transformer_layers,
    available_layers,
    forward_cett,
    forward_cett_span,
    get_mlp_down_proj,
    precompute_col_norms,
    scale_h_neurons,
)

# ---------------------------------------------------------------------------
# Minimal mock models
# ---------------------------------------------------------------------------

_H, _I, _V, _L = 8, 16, 32, 4


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


class _CausalLM(nn.Module):
    def __init__(self, n=_L):
        super().__init__()
        torch.manual_seed(0)

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block() for _ in range(n)])

        self.model = _Inner()
        self.lm_head = nn.Linear(_H, _V, bias=True)

    def forward(self, input_ids=None, **kw):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, _H)
        for block in self.model.layers:
            x = block(x)

        class _Out:
            pass

        out = _Out()
        out.logits = self.lm_head(x)
        return out


class _MultimodalLM(nn.Module):
    """Mimics MedGemma-4B: model.model.language_model.layers."""

    def __init__(self):
        super().__init__()

        class _Lang(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block() for _ in range(2)])

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _Lang()

        self.model = _Inner()


M = _CausalLM()


def _tok(text="ABCDE"):
    ids = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ---------------------------------------------------------------------------


class TestArchitectureDetection:
    def test_standard_causal_lm(self):
        assert len(_get_transformer_layers(M)) == _L

    def test_multimodal_wrapper(self):
        assert len(_get_transformer_layers(_MultimodalLM())) == 2

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            _get_transformer_layers(nn.Linear(4, 4))

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            get_mlp_down_proj(M, 999)


class TestColNorms:
    def test_shape_and_sign(self):
        norms = precompute_col_norms(M, available_layers(M))
        for li, v in norms.items():
            assert v.shape == (_I,)
            assert (v >= 0).all()


class TestForwardCett:
    def setup_method(self):
        self.layers = available_layers(M)
        self.norms = precompute_col_norms(M, self.layers)

    def test_output_shapes(self):
        cett, logits = forward_cett(M, _tok(), self.layers, self.norms)
        assert cett.shape == (len(self.layers) * _I,)
        assert logits.shape == (_V,)

    def test_last_token_equals_explicit(self):
        toks = _tok("ABC")
        seq = toks["input_ids"].shape[1]
        c1, _ = forward_cett(M, toks, self.layers, self.norms, token_position=-1)
        c2, _ = forward_cett(M, toks, self.layers, self.norms, token_position=seq - 1)
        assert torch.allclose(c1, c2)


class TestForwardCettSpan:
    def setup_method(self):
        self.layers = available_layers(M)
        self.norms = precompute_col_norms(M, self.layers)

    def test_output_shape(self):
        result = forward_cett_span(M, _tok("ABCDE"), 1, 3, self.layers, self.norms)
        assert result.shape == (len(self.layers) * _I,)

    def test_max_gte_mean(self):
        toks = _tok("ABCDE")
        mean = forward_cett_span(M, toks, 1, 4, self.layers, self.norms, "mean")
        mx = forward_cett_span(M, toks, 1, 4, self.layers, self.norms, "max")
        assert (mx >= mean - 1e-5).all()


class TestScaleHNeurons:
    def setup_method(self):
        self.layers = available_layers(M)
        self.norms = precompute_col_norms(M, self.layers)

    def test_alpha_one_is_identity(self):
        toks = _tok("XY")
        _, baseline = forward_cett(M, toks, self.layers, self.norms)
        scaled = scale_h_neurons(M, toks, [(0, 1), (2, 5)], 1.0, self.layers)
        assert torch.allclose(baseline, scaled, atol=1e-5)

    def test_alpha_zero_suppresses(self):
        toks = _tok("XY")
        # Use all neurons in layer 0 to guarantee at least one fires
        neurons = [(0, i) for i in range(_I)]
        l1 = scale_h_neurons(M, toks, neurons, 1.0, self.layers)
        l0 = scale_h_neurons(M, toks, neurons, 0.0, self.layers)
        assert not torch.allclose(l1, l0)
