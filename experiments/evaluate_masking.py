"""RQ2: zero-mask each emotion's neuron set, compare classification accuracy.

For each target emotion ``e``:
  1. Build mask_map from ``emotion_neurons[e]``.
  2. Run zero-shot classification on the eval split WITHOUT mask (baseline).
     We cache the unmasked predictions once across emotions.
  3. Run with mask installed via :class:`utils.MaskingHook`.
  4. Compute per-true-emotion accuracy and emotion-of-interest delta.

Output:
  * ``masking_results.csv`` rows:
      emotion_masked, emotion_label, n, acc_normal, acc_masked, delta_pct
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompts import EMOTIONS, format_messages  # noqa: E402
from utils import MaskingHook  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_mask_map(
    neuron_list: list[tuple[int, int]],
    n_layers: int,
    d_ff: int,
) -> dict[int, torch.Tensor]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for (l, i) in neuron_list:
        by_layer[int(l)].append(int(i))
    mask_map: dict[int, torch.Tensor] = {}
    for l, idxs in by_layer.items():
        m = torch.zeros(d_ff, dtype=torch.bool)
        m[torch.tensor(idxs, dtype=torch.long)] = True
        mask_map[l] = m
    return mask_map


def classify_batch(
    model,
    tokenizer,
    dialogues: list[str],
    max_new_tokens: int = 8,
) -> list[str]:
    """Greedy single-word emotion predictions, one-by-one to keep hooks
    simple. Batch-size=1 is sufficient here since the bulk of cost is
    the forward pass itself."""
    preds: list[str] = []
    for dialogue in dialogues:
        messages = format_messages(dialogue)
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        # Canonicalize: first matching emotion word wins.
        pred = "unknown"
        for e in EMOTIONS:
            if e in text:
                pred = e
                break
        preds.append(pred)
    return preds


def per_emotion_accuracy(
    preds: list[str], labels: list[str]
) -> dict[str, tuple[int, int]]:
    """Return ``{emotion: (correct, total)}``."""
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for p, y in zip(preds, labels):
        total[y] += 1
        if p == y:
            correct[y] += 1
    out: dict[str, tuple[int, int]] = {}
    for e in EMOTIONS:
        out[e] = (correct[e], total[e])
    return out


def run(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load selection artifacts ---
    neurons_path = run_dir / "emotion_neurons.pt"
    blob = torch.load(str(neurons_path), map_location="cpu")
    by_emotion: dict[str, list[tuple[int, int]]] = blob["by_emotion"]
    n_layers = int(blob["n_layers"])
    d_ff = int(blob["d_ff"])
    print(f"[load] {neurons_path}  L={n_layers}  d_ff={d_ff}")
    for e in EMOTIONS:
        print(f"       {e}: {len(by_emotion.get(e, []))} neurons")

    # --- load eval data ---
    eval_data = load_jsonl(args.eval)
    if args.subsample and args.subsample < len(eval_data):
        eval_data.sort(key=lambda x: (x["label"], x["topic"], x["dialogue"][:64]))
        eval_data = eval_data[: args.subsample]
    print(f"[load] eval n={len(eval_data)}")

    labels = [d["label"] for d in eval_data]
    dialogues = [d["dialogue"] for d in eval_data]

    # --- load model ---
    print(f"[load] model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # --- baseline: no mask ---
    print("[eval] baseline (no mask)")
    preds_baseline = classify_batch(model, tokenizer, dialogues)
    acc_baseline = per_emotion_accuracy(preds_baseline, labels)

    # --- per-emotion masking ---
    rows: list[dict] = []
    for em_mask in EMOTIONS:
        neurons = by_emotion.get(em_mask, [])
        if not neurons:
            print(f"[skip] {em_mask}: empty neuron set")
            continue
        mask_map = build_mask_map(neurons, n_layers, d_ff)
        print(f"[eval] mask {em_mask} ({len(neurons)} neurons across {len(mask_map)} layers)")
        with MaskingHook(model, mask_map):
            preds_masked = classify_batch(model, tokenizer, dialogues)
        acc_masked = per_emotion_accuracy(preds_masked, labels)

        for em_label in EMOTIONS:
            c0, n0 = acc_baseline[em_label]
            c1, n1 = acc_masked[em_label]
            a0 = (c0 / n0) if n0 else 0.0
            a1 = (c1 / n1) if n1 else 0.0
            rows.append(
                {
                    "emotion_masked": em_mask,
                    "emotion_label": em_label,
                    "n": n0,
                    "acc_normal": a0,
                    "acc_masked": a1,
                    "delta_pct": (a1 - a0) * 100.0,
                }
            )

    # --- write csv ---
    csv_path = out_dir / "masking_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["emotion_masked", "emotion_label", "n", "acc_normal", "acc_masked", "delta_pct"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {csv_path}")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": args.model,
                "run_dir": str(run_dir),
                "eval": args.eval,
                "subsample": args.subsample,
                "n_eval": len(eval_data),
            },
            fh,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RQ2: zero-mask each emotion's neuron set; compare classification accuracy."
    )
    ap.add_argument("--run-dir", required=True, help="Dir from neuron_selection.py")
    ap.add_argument("--eval", required=True, help="Eval JSONL from split_data.py")
    ap.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
