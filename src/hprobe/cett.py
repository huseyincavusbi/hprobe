"""CETT metric computation.

CETT (Contribution to rEsidual sTream norm of Token t) measures how much
a single FFN neuron contributes to the hidden state at a given token position.

Formula
-------
    CETT(j, t) = |z_{j,t}| · ‖W_down[:, j]‖₂ / ‖h_t‖₂

where:
    z_{j,t}  = SwiGLU activation of neuron j at token t (input to W_down)
    h_t      = W_down · z_t  (FFN output vector at token t)

Reference: arXiv:2512.01797
"""

from typing import Dict, List, Tuple

import torch


def _get_transformer_layers(model: torch.nn.Module):
    """Return the transformer layer list for any supported architecture.

    Handles:
      - Standard causal LMs:  model.model.layers
        (Gemma-3, Llama, Mistral)
      - Multimodal wrappers:  model.model.language_model.layers
        (MedGemma-4B-IT / Gemma3ForConditionalGeneration)
    """
    # Multimodal: model.model.language_model.layers (MedGemma-4B-IT)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    # Standard causal LM: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        f"Unsupported architecture: {type(model).__name__}. "
        "Expected model.model.layers or model.model.language_model.layers."
    )


def get_mlp_down_proj(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    """Return the down-projection linear layer for a given transformer layer index."""
    layers = _get_transformer_layers(model)
    if layer_idx >= len(layers):
        raise IndexError(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")
    block = layers[layer_idx]
    if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
        return block.mlp.down_proj
    raise ValueError(f"Cannot find down_proj in layer {layer_idx} of {type(model).__name__}")


def available_layers(model: torch.nn.Module) -> List[int]:
    """Return list of all available layer indices for the model."""
    return list(range(len(_get_transformer_layers(model))))


def precompute_col_norms(
    model: torch.nn.Module,
    layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Precompute ‖W_down[:, j]‖₂ for each layer.

    Returns dict mapping layer_idx → (intermediate_dim,) tensor of column norms.
    Computed once and reused across all samples.
    """
    col_norms = {}
    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)
        W = down_proj.weight.detach().float()  # (hidden_dim, intermediate_dim)
        col_norms[layer_idx] = torch.norm(W, dim=0).cpu()  # (intermediate_dim,)
    return col_norms


def forward_cett(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    token_position: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass — extract CETT at a given token position.

    Parameters
    ----------
    model : causal LM
    tokens : tokenizer output (input_ids, attention_mask on correct device)
    layers : list of layer indices to hook
    col_norms : precomputed column norms from precompute_col_norms()
    token_position : which token to extract CETT at (-1 = last token)

    Returns
    -------
    cett_vec : (n_layers * intermediate_dim,) float32 — concatenated CETT values
    logits   : (vocab_size,) float32 — output logits at the last token
    """
    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z = input[0]
                h = output
                z_cache[idx] = z[0, token_position, :].detach().float().cpu()
                h_cache[idx] = h[0, token_position, :].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            out = model(**tokens)
    finally:
        for h in handles:
            h.remove()

    logits = out.logits[0, -1, :].detach().float().cpu()

    cett_parts = []
    for layer_idx in layers:
        z = z_cache[layer_idx]
        h = h_cache[layer_idx]
        h_norm = torch.norm(h).item() + 1e-8
        cett = (z * col_norms[layer_idx]) / h_norm
        cett_parts.append(cett)

    return torch.cat(cett_parts, dim=0), logits


def forward_cett_at_token(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    extra_token_id: int,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Append one token to the input and capture CETT at that appended position.

    Used by contrastive mode: appends the predicted answer token (A/B/C/D)
    and captures CETT specifically when the model is generating that token.

    Returns
    -------
    cett_answer : (n_layers * intermediate_dim,) float32
    """
    input_ids = tokens["input_ids"]
    extra_t = torch.tensor([[extra_token_id]], device=input_ids.device)
    extended_ids = torch.cat([input_ids, extra_t], dim=1)

    extended: Dict[str, torch.Tensor] = {"input_ids": extended_ids}
    if "attention_mask" in tokens:
        m = tokens["attention_mask"]
        extended["attention_mask"] = torch.cat(
            [m, torch.ones((1, 1), device=m.device, dtype=m.dtype)], dim=1
        )

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0][0, -1, :].detach().float().cpu()
                h_cache[idx] = output[0, -1, :].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(**extended)
    finally:
        for h in handles:
            h.remove()

    cett_parts = []
    for layer_idx in layers:
        h_norm = torch.norm(h_cache[layer_idx]).item() + 1e-8
        cett = (z_cache[layer_idx] * col_norms[layer_idx]) / h_norm
        cett_parts.append(cett)

    return torch.cat(cett_parts, dim=0)


def forward_cett_span(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    span_start: int,
    span_end: int,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    aggregation: str = "mean",
) -> torch.Tensor:
    """Forward pass over a full sequence — extract CETT aggregated over a token span.

    Used by fit_from_responses(): feeds the full Q+A sequence and captures
    CETT over the answer token span, then aggregates with mean or max.

    Parameters
    ----------
    tokens : tokenizer output for the full Q+A sequence
    span_start : start index of the answer token span (inclusive)
    span_end   : end index of the answer token span (exclusive)
    aggregation : "mean" or "max" over the span

    Returns
    -------
    cett_vec : (n_layers * intermediate_dim,) float32
    """
    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                # capture full sequence: [seq_len, intermediate_dim]
                z_cache[idx] = input[0][0].detach().float().cpu()
                h_cache[idx] = output[0].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(**tokens)
    finally:
        for h in handles:
            h.remove()

    cett_parts = []
    for layer_idx in layers:
        z_span = z_cache[layer_idx][span_start:span_end]  # [span_len, intermediate_dim]
        h_span = h_cache[layer_idx][span_start:span_end]  # [span_len, hidden_dim]
        h_norms = torch.norm(h_span, dim=-1, keepdim=True) + 1e-8  # [span_len, 1]
        cett_span = (
            torch.abs(z_span) * col_norms[layer_idx].unsqueeze(0)
        ) / h_norms  # [span_len, inter_dim]
        if aggregation == "max":
            cett_agg = cett_span.max(dim=0).values
        else:
            cett_agg = cett_span.mean(dim=0)
        cett_parts.append(cett_agg)

    return torch.cat(cett_parts, dim=0)


def scale_h_neurons(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    h_neurons: List[Tuple[int, int]],
    alpha: float,
    layers: List[int],
) -> torch.Tensor:
    """Forward pass scaling H-Neuron activations by alpha (causal intervention).

    alpha < 1 suppresses, alpha = 1 is baseline, alpha > 1 amplifies.

    Returns logits (vocab_size,) at the last token.
    """
    neurons_by_layer: Dict[int, List[int]] = {}
    for layer_idx, neuron_idx in h_neurons:
        neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

    handles = []
    for layer_idx in layers:
        if layer_idx not in neurons_by_layer:
            continue
        indices = torch.tensor(neurons_by_layer[layer_idx], dtype=torch.long)
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_pre_hook(idx: torch.Tensor, a: float):
            def pre_hook(module, input):
                z = input[0].clone()
                z[..., idx.to(z.device)] *= a
                return (z,) + input[1:]

            return pre_hook

        handles.append(down_proj.register_forward_pre_hook(make_pre_hook(indices, alpha)))

    try:
        with torch.no_grad():
            out = model(**tokens)
    finally:
        for h in handles:
            h.remove()

    return out.logits[0, -1, :].detach().float().cpu()
