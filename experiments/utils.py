"""Shared utilities for Emotion-Neuron RQ1/RQ2/RQ3 experiments.

Contains:
  * :class:`ActivationMode` — enum of the three hook-semantics choices
    documented in HOOK_CHOICE.md.
  * :class:`FFNActivationHook` — forward-hook capture of per-neuron
    activations on a Llama-3.1 ``LlamaMLP``.
  * :class:`MaskingHook` — forward-pre-hook that zero-ablates selected
    intermediate columns on ``down_proj``, matching ``gated``-mode
    semantics used during selection.
  * :func:`get_last_assistant_content_token_idx` — returns the index of
    the last assistant content token immediately before EOS for a
    fully-formatted chat prompt. This is the "single-word emotion
    prediction" token referenced in the paper.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


class ActivationMode(str, Enum):
    """SwiGLU interpretation choice — see HOOK_CHOICE.md."""

    GATED = "gated"            # silu(gate_proj(x)) * up_proj(x)  [DEFAULT, TENTATIVE]
    SILU_ONLY = "silu_only"    # silu(gate_proj(x))
    PRE_SILU = "pre_silu"      # gate_proj(x)  (pre-nonlinearity)


class FFNActivationHook:
    """Capture per-neuron FFN activations during a forward pass on
    Llama-3.1 (SwiGLU).

    Paper Eq. 4 uses ``n = max(0, h)`` (ReLU semantics); Llama-3.1 has no
    ReLU anywhere, so we interpret ``h`` per ``activation_mode``. See
    HOOK_CHOICE.md for rationale and the empirical comparison template.

    For ``GATED``: hook ``down_proj`` with a ``forward_pre_hook`` so we
    read the tensor actually consumed by ``down_proj``, which equals
    ``silu(gate_proj(x)) * up_proj(x)``.

    For ``SILU_ONLY`` / ``PRE_SILU``: hook ``gate_proj`` directly and
    optionally apply silu.

    Captures are stored on CPU as FP32 to keep memory bounded — callers
    should read and reduce them per batch, then call :meth:`clear`.
    """

    def __init__(self, model, activation_mode: ActivationMode = ActivationMode.GATED):
        self.model = model
        self.mode = activation_mode
        self.handles: list = []
        # layer_idx -> tensor [B, T, d_ff] on CPU (FP32)
        self.captures: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self) -> "FFNActivationHook":
        if self.mode == ActivationMode.GATED:
            self._register_gated()
        else:
            self._register_alt()
        return self

    def _register_gated(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp

            def make_pre_hook(li: int):
                def hook(_module, inputs):
                    # inputs[0] is the tensor handed to down_proj,
                    # exactly silu(gate_proj(x)) * up_proj(x).
                    self.captures[li] = inputs[0].detach().float().cpu()

                return hook

            h = mlp.down_proj.register_forward_pre_hook(make_pre_hook(layer_idx))
            self.handles.append(h)

    def _register_alt(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            gate = layer.mlp.gate_proj

            def make_hook(li: int, mode: ActivationMode):
                def hook(_module, _inputs, output):
                    if mode == ActivationMode.SILU_ONLY:
                        val = F.silu(output)
                    else:  # PRE_SILU
                        val = output
                    self.captures[li] = val.detach().float().cpu()

                return hook

            h = gate.register_forward_hook(make_hook(layer_idx, self.mode))
            self.handles.append(h)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.captures.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "FFNActivationHook":
        return self.register()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


class MaskingHook:
    """Zero-ablate neuron indices per layer via ``forward_pre_hook`` on
    ``down_proj``.

    Parameters
    ----------
    model:
        The ``LlamaForCausalLM`` (or compatible) whose ``model.layers``
        list exposes ``.mlp.down_proj``.
    mask_map:
        Mapping ``layer_idx -> 1-D bool tensor of length ``d_ff``. A
        ``True`` entry zeroes that intermediate-dim column before
        ``down_proj`` consumes it.
    """

    def __init__(self, model, mask_map: dict[int, torch.Tensor]):
        self.model = model
        self.mask_map = mask_map
        self.handles: list = []

    def register(self) -> "MaskingHook":
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx not in self.mask_map:
                continue
            mask = self.mask_map[layer_idx]

            def make_hook(m: torch.Tensor):
                def hook(_module, inputs):
                    x = inputs[0]
                    # Clone to avoid in-place mutation of upstream tensor.
                    x = x.clone()
                    x[..., m.to(x.device)] = 0.0
                    return (x,)

                return hook

            h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(mask))
            self.handles.append(h)
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "MaskingHook":
        return self.register()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


# ----------------------------------------------------------------------
# token indexing
# ----------------------------------------------------------------------
def get_last_assistant_content_token_idx(
    input_ids: torch.Tensor,
    tokenizer,
) -> int:
    """Return the index of the last non-special assistant content token.

    We use the chat-formatted prompt ending in Llama-3.1-Instruct's
    ``<|eot_id|>`` (end-of-turn) or ``<|end_of_text|>`` (EOS). The
    single-word emotion prediction sits immediately before that sentinel;
    if generation has not run yet and the prompt ends at
    ``assistant``-turn-open, the last token IS the last content token
    we can use as a proxy (the model is about to emit the prediction
    there).

    Sentinel choice: we look for ``<|eot_id|>`` first (Llama-3.1's
    per-turn terminator), fall back to ``tokenizer.eos_token_id``, and
    otherwise return ``input_ids.shape[-1] - 1``.
    """
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    seq_len = ids.shape[-1]

    # Resolve possible sentinels.
    sentinel_ids: list[int] = []
    eot = getattr(tokenizer, "convert_tokens_to_ids", lambda _x: None)("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id:
        sentinel_ids.append(eot)
    if tokenizer.eos_token_id is not None:
        sentinel_ids.append(int(tokenizer.eos_token_id))

    for i in range(seq_len - 1, -1, -1):
        tok = int(ids[i].item())
        if tok in sentinel_ids:
            continue
        return i
    return seq_len - 1


def build_layer_range(
    n_layers: int, spec: str
) -> range:
    """Resolve a layer-range spec to a :class:`range`.

    spec ∈ {"Bottom", "Middle", "Top", "All"}; the paper partitions
    ``[0, L)`` into thirds.
    """
    spec = spec.capitalize()
    if spec == "Bottom":
        return range(0, n_layers // 3)
    if spec == "Middle":
        return range(n_layers // 3, 2 * n_layers // 3)
    if spec == "Top":
        return range(2 * n_layers // 3, n_layers)
    if spec == "All":
        return range(0, n_layers)
    raise ValueError(f"Unknown layer-range spec: {spec!r}")
