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
      - Standard causal LMs:  model.model.layers (Gemma, Llama, Mistral)
      - Multimodal wrappers:  model.model.language_model.layers (MedGemma)
      - GPT-2:                model.transformer.h
      - OPT:                  model.model.decoder.layers
    """
    # Multimodal: model.model.language_model.layers (MedGemma-4B-IT)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    # Standard causal LM: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # OPT: model.model.decoder.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers

    raise ValueError(
        f"Unsupported architecture: {type(model).__name__}. "
        "Expected model.model.layers or model.transformer.h."
    )


def get_mlp_down_proj(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    """Return the down-projection linear layer for a given transformer layer index.

    Supports:
      - down_proj (Llama, Gemma, Mistral)
      - c_proj (GPT2)
      - fc2 (OPT)
    """
    layers = _get_transformer_layers(model)
    if layer_idx >= len(layers):
        raise IndexError(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")
    block = layers[layer_idx]

    # Handle various MLP layer names
    if hasattr(block, "mlp"):
        mlp = block.mlp
        for name in ["down_proj", "c_proj", "fc2"]:
            if hasattr(mlp, name):
                return getattr(mlp, name)

    # Some architectures might have it at the block level directly
    for name in ["down_proj", "c_proj", "fc2"]:
        if hasattr(block, name):
            return getattr(block, name)

    raise AttributeError(
        f"Could not find MLP down-projection layer in {type(block).__name__}. "
        "Checked: .mlp.down_proj, .mlp.c_proj, .mlp.fc2"
    )


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
        W = down_proj.weight.detach().float()  # (hidden_dim, intermediate_dim) usually

        # GPT-2 uses Conv1D where weight is (intermediate_dim, hidden_dim)
        if type(down_proj).__name__ == "Conv1D":
            col_norms[layer_idx] = torch.norm(W, dim=1).cpu()  # (intermediate_dim,)
        else:
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


def forward_cett_batch(
    model: torch.nn.Module,
    batch_tokens: Dict[str, torch.Tensor],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    token_positions: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched forward pass — extract CETT for each sample at its token position.

    Parameters
    ----------
    batch_tokens : tokenizer output with batch_size > 1 (must be right-padded)
    token_positions : last real token position per sample (index into padded seq)

    Returns
    -------
    cett_matrix : (batch_size, n_layers * intermediate_dim) float32
    logits_matrix : (batch_size, vocab_size) float32
    """
    batch_size = batch_tokens["input_ids"].shape[0]
    device = batch_tokens["input_ids"].device
    batch_idx = torch.arange(batch_size, device=device)
    token_pos_t = torch.tensor(token_positions, device=device)

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                # Slice to needed positions on GPU — avoids copying full (B, T, D) off-device
                z_cache[idx] = input[0][batch_idx, token_pos_t].detach().float()  # (B, D)
                h_cache[idx] = output[batch_idx, token_pos_t].detach().float()  # (B, D)
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    # Compute position_ids so real tokens get 0-based positions regardless of left-padding
    if "attention_mask" in batch_tokens:
        position_ids = (batch_tokens["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
    else:
        seq_len = batch_tokens["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    try:
        with torch.no_grad():
            out = model(**batch_tokens, position_ids=position_ids)
    finally:
        for h in handles:
            h.remove()

    logits_matrix = out.logits[batch_idx, token_pos_t].detach().float().cpu()  # (B, vocab)

    # Vectorized CETT on GPU: stack all layers then compute in one shot
    z_all = torch.stack([z_cache[li] for li in layers], dim=0)  # (L, B, D)
    h_all = torch.stack([h_cache[li] for li in layers], dim=0)  # (L, B, D)
    col_norms_gpu = torch.stack([col_norms[li].to(device) for li in layers])  # (L, D)

    h_norm = torch.norm(h_all, dim=-1, keepdim=True) + 1e-8  # (L, B, 1)
    cett = (z_all * col_norms_gpu.unsqueeze(1)) / h_norm  # (L, B, D)
    cett_matrix = cett.permute(1, 0, 2).reshape(batch_size, -1).cpu()  # (B, L*D)

    return cett_matrix, logits_matrix


def forward_cett_at_token_batch(
    model: torch.nn.Module,
    batch_tokens: Dict[str, torch.Tensor],
    extra_token_ids: List[int],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Batched version of forward_cett_at_token.

    Appends one answer token per sample and captures CETT at the last position.

    Parameters
    ----------
    extra_token_ids : answer token id per sample

    Returns
    -------
    cett_matrix : (batch_size, n_layers * intermediate_dim) float32
    """
    batch_size = batch_tokens["input_ids"].shape[0]
    device = batch_tokens["input_ids"].device

    extra_t = torch.tensor(extra_token_ids, device=device).unsqueeze(1)
    extended_ids = torch.cat([batch_tokens["input_ids"], extra_t], dim=1)

    extended: Dict[str, torch.Tensor] = {"input_ids": extended_ids}
    if "attention_mask" in batch_tokens:
        m = batch_tokens["attention_mask"]
        extended["attention_mask"] = torch.cat(
            [m, torch.ones((batch_size, 1), device=device, dtype=m.dtype)], dim=1
        )

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                # Slice last position on GPU — avoids copying full (B, T, D) off-device
                z_cache[idx] = input[0][:, -1, :].detach().float()  # (B, D)
                h_cache[idx] = output[:, -1, :].detach().float()  # (B, D)
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    if "attention_mask" in extended:
        position_ids = (extended["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
    else:
        seq_len = extended["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    try:
        with torch.no_grad():
            model(**extended, position_ids=position_ids)
    finally:
        for h in handles:
            h.remove()

    # Vectorized CETT on GPU
    z_all = torch.stack([z_cache[li] for li in layers], dim=0)  # (L, B, D)
    h_all = torch.stack([h_cache[li] for li in layers], dim=0)  # (L, B, D)
    col_norms_gpu = torch.stack([col_norms[li].to(device) for li in layers])  # (L, D)

    h_norm = torch.norm(h_all, dim=-1, keepdim=True) + 1e-8  # (L, B, 1)
    cett = (z_all * col_norms_gpu.unsqueeze(1)) / h_norm  # (L, B, D)
    cett_matrix = cett.permute(1, 0, 2).reshape(batch_size, -1).cpu()  # (B, L*D)

    return cett_matrix


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
