"""RQ1: emotion-neuron selection via per-neuron activation entropy.

Algorithm (paper Sec. 3.4, reformulated for Llama-3.1 SwiGLU — see
HOOK_CHOICE.md for the activation-mode interpretation):

    For each neuron u = (layer l, neuron_idx i):
      For each emotion e in EMOTIONS:
        f_{u,e}  = #{activated tokens} on e-labeled dialogue tokens
        P_{u,e}  = f_{u,e} / T_e                      # T_e total tokens for e
      P_hat_{u,e} = P_{u,e} / sum_e' P_{u,e'}         # L1 normalize across e
      H_u = -sum_e P_hat_{u,e} * log(P_hat_{u,e} + eps)
    Rank all neurons ascending by H_u; take top ``--top_pct`` global.
    emotion(u) = argmax_e P_hat_{u,e}.

Output under ``--output``:
  * ``emotion_neurons.pt`` — torch dict
      ``{emotion: [(layer, neuron_idx), ...], "all_ranked": [...]}``
  * ``layer_distribution.csv`` — rows: layer, emotion, count
  * ``counts_summary.csv``     — rows: emotion, total_count
  * ``run_config.json``        — CLI args + dataset stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompts import EMOTIONS, format_messages  # noqa: E402
from utils import (  # noqa: E402
    ActivationMode,
    FFNActivationHook,
    get_last_assistant_content_token_idx,
)


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------
# accumulators
# ----------------------------------------------------------------------
def make_counters(
    n_layers: int, d_ff: int, n_emotions: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(f, T)`` where
    ``f`` is a ``[L, d_ff, E]`` float64 count tensor and
    ``T`` is ``[E]`` float64 token totals.
    """
    f = torch.zeros((n_layers, d_ff, n_emotions), dtype=torch.float64, device=device)
    T = torch.zeros((n_emotions,), dtype=torch.float64, device=device)
    return f, T


def collect_activations_for_sample(
    hook: FFNActivationHook,
    token_positions: torch.Tensor,  # [K]
    f: torch.Tensor,
    emotion_idx: int,
) -> int:
    """Fold this sample's activations into ``f``. Returns #tokens counted.

    ``hook.captures[l]`` is ``[B=1, T, d_ff]`` FP32 on CPU.
    Activated := (> 0). This matches the ``max(0, h)`` semantics under
    any of the three activation modes.
    """
    device = f.device
    n_tokens = int(token_positions.shape[0])
    if n_tokens == 0:
        return 0
    for layer_idx, act in hook.captures.items():
        # [1, T, d_ff] -> select token positions -> [K, d_ff]
        sel = act[0, token_positions].to(device=device)
        activated = (sel > 0).to(torch.float64)  # [K, d_ff]
        # accumulate into emotion column
        f[layer_idx, :, emotion_idx] += activated.sum(dim=0)
    return n_tokens


# ----------------------------------------------------------------------
# main selection pass
# ----------------------------------------------------------------------
def run_selection(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] data: {args.train}")
    data = load_jsonl(args.train)
    if args.subsample and args.subsample < len(data):
        # Stratified subsample: roughly equal samples per label, deterministic.
        # Sort first so the per-label slice is reproducible across runs.
        data.sort(key=lambda x: (x["label"], x["topic"], x["dialogue"][:64]))
        from collections import defaultdict as _dd

        by_label: dict[str, list[dict]] = _dd(list)
        for item in data:
            by_label[item["label"]].append(item)
        labels_sorted = sorted(by_label.keys())
        per_label = max(1, args.subsample // len(labels_sorted))
        sampled: list[dict] = []
        for lbl in labels_sorted:
            sampled.extend(by_label[lbl][:per_label])
        data = sampled
    n = len(data)
    print(f"[load] {n} dialogues for selection")

    print(f"[load] model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_ff = model.config.intermediate_size
    n_emotions = len(EMOTIONS)
    print(f"[config] n_layers={n_layers}  d_ff={d_ff}  n_emotions={n_emotions}")
    print(f"[config] activation_mode={args.activation_mode}  token_mode={args.token_mode}")

    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    # Accumulate on CPU to avoid pressuring the GPUs used by device_map="auto".
    counter_device = torch.device("cpu")
    f, T = make_counters(n_layers, d_ff, n_emotions, counter_device)

    activation_mode = ActivationMode(args.activation_mode)
    hook = FFNActivationHook(model, activation_mode=activation_mode).register()

    n_skipped = 0

    try:
        with torch.no_grad():
            for sample in tqdm(data, desc="RQ1 activation pass"):
                label = sample.get("label")
                if label not in emotion_to_idx:
                    n_skipped += 1
                    continue
                e_idx = emotion_to_idx[label]

                messages = format_messages(sample["dialogue"])
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                # transformers >=5.x returns a BatchEncoding; earlier versions return a Tensor.
                input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]

                # Place input on the device of the first layer's embedding.
                input_ids = input_ids.to(model.device)

                hook.clear()
                _ = model(input_ids=input_ids, use_cache=False)

                # Pick token positions per --token_mode.
                if args.token_mode == "last":
                    pos = get_last_assistant_content_token_idx(input_ids, tokenizer)
                    token_positions = torch.tensor([pos], dtype=torch.long)
                else:  # "all"
                    T_seq = int(input_ids.shape[-1])
                    token_positions = torch.arange(T_seq, dtype=torch.long)

                k = collect_activations_for_sample(hook, token_positions, f, e_idx)
                T[e_idx] += k
    finally:
        hook.remove()

    print(f"[stats] skipped {n_skipped} samples with invalid label")
    print(f"[stats] token totals T_e = {T.tolist()}")

    if (T <= 0).any():
        raise RuntimeError(
            "Some emotion had zero tokens counted; cannot normalize. "
            f"T = {T.tolist()}"
        )

    # P[l, i, e] = f[l, i, e] / T[e]
    P = f / T.view(1, 1, -1)

    # L1 normalize across emotions.
    denom = P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    P_hat = P / denom

    # Entropy per neuron.
    eps = 1e-12
    H = -(P_hat * (P_hat + eps).log()).sum(dim=-1)  # [L, d_ff]

    # Global ranking: flatten, take lowest-entropy top_pct.
    H_flat = H.reshape(-1)
    total_neurons = int(H_flat.numel())
    k = max(1, math.ceil(args.top_pct * total_neurons))
    print(f"[select] total_neurons={total_neurons}  k(top {args.top_pct*100:.3f}%)={k}")

    # torch.topk largest=False returns lowest values.
    vals, idx = torch.topk(H_flat, k=k, largest=False, sorted=True)

    layers = (idx // d_ff).tolist()
    cols = (idx % d_ff).tolist()

    # Emotion assignment per selected neuron: argmax of P_hat.
    P_hat_flat = P_hat.reshape(total_neurons, n_emotions)
    chosen_emotions = P_hat_flat[idx].argmax(dim=-1).tolist()

    by_emotion: dict[str, list[tuple[int, int]]] = {e: [] for e in EMOTIONS}
    all_ranked: list[dict] = []
    for rank, (li, ci, ei, hv) in enumerate(zip(layers, cols, chosen_emotions, vals.tolist())):
        emotion = EMOTIONS[ei]
        by_emotion[emotion].append((int(li), int(ci)))
        all_ranked.append(
            {
                "rank": rank,
                "layer": int(li),
                "neuron_idx": int(ci),
                "entropy": float(hv),
                "emotion": emotion,
            }
        )

    counts = {e: len(v) for e, v in by_emotion.items()}
    print(f"[select] per-emotion counts: {counts}")

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------
    torch.save(
        {
            "emotions": EMOTIONS,
            "by_emotion": by_emotion,
            "all_ranked": all_ranked,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "top_pct": args.top_pct,
            "k": k,
            "activation_mode": args.activation_mode,
            "token_mode": args.token_mode,
            "model": args.model,
        },
        str(out_dir / "emotion_neurons.pt"),
    )

    with open(out_dir / "counts_summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["emotion", "total_count"])
        for e in EMOTIONS:
            w.writerow([e, counts[e]])

    with open(out_dir / "layer_distribution.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "emotion", "count"])
        for l in range(n_layers):
            for e in EMOTIONS:
                c = sum(1 for (ll, _ci) in by_emotion[e] if ll == l)
                if c:
                    w.writerow([l, e, c])

    run_config = {
        "model": args.model,
        "train": args.train,
        "subsample": args.subsample,
        "activation_mode": args.activation_mode,
        "token_mode": args.token_mode,
        "top_pct": args.top_pct,
        "output": str(out_dir),
        "n_dialogues_used": n - n_skipped,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "total_neurons": total_neurons,
        "k_selected": k,
        "T_per_emotion": {EMOTIONS[i]: float(T[i].item()) for i in range(n_emotions)},
        "counts_by_emotion": counts,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    print(f"[done] wrote outputs under {out_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RQ1: select emotion neurons via per-neuron activation entropy.",
    )
    ap.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF model id. Default: meta-llama/Meta-Llama-3.1-8B-Instruct.",
    )
    ap.add_argument(
        "--train",
        required=True,
        help="JSONL from split_data.py (train split).",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Smoke-test subsample size; 0 = use all. 1000 recommended for smoke.",
    )
    ap.add_argument(
        "--activation_mode",
        choices=[m.value for m in ActivationMode],
        default=ActivationMode.GATED.value,
        help="SwiGLU interpretation (see HOOK_CHOICE.md).",
    )
    ap.add_argument(
        "--token_mode",
        choices=["last", "all"],
        default="last",
        help=(
            "last = last assistant content token before EOS (default, paper proxy). "
            "all = every input token (~30x compute; see README)."
        ),
    )
    ap.add_argument(
        "--top_pct",
        type=float,
        default=0.01,
        help="Top fraction (lowest entropy) to keep. Default 0.01 = 1%%.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output directory.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    run_selection(args)


if __name__ == "__main__":
    main()
