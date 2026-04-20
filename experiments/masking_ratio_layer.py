"""RQ3: sweep mask_ratio x layer_range for each emotion.

For each (emotion, ratio, layer_range):
  * Restrict ``emotion_neurons[emotion]`` to neurons whose layer is in
    the layer_range's slice; take the first ``ratio`` fraction of that
    filtered list (neurons are already globally ranked by entropy
    ascending within the selection run).
  * Zero-mask via :class:`utils.MaskingHook`.
  * Evaluate per-emotion accuracy on the eval split.

Layer ranges (paper Sec. 3.4):
  Bottom = [0, L/3)
  Middle = [L/3, 2L/3)
  Top    = [2L/3, L)
  All    = [0, L)

Output: ``ratio_layer_results.csv`` with one row per
(emotion_masked, ratio, layer_range, emotion_label).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompts import EMOTIONS  # noqa: E402
from evaluate_masking import (  # noqa: E402
    build_mask_map,
    classify_batch,
    load_jsonl,
    per_emotion_accuracy,
)
from utils import MaskingHook, build_layer_range  # noqa: E402


def run(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(str(run_dir / "emotion_neurons.pt"), map_location="cpu")
    by_emotion: dict[str, list[tuple[int, int]]] = blob["by_emotion"]
    n_layers = int(blob["n_layers"])
    d_ff = int(blob["d_ff"])
    print(f"[load] selection L={n_layers}  d_ff={d_ff}")

    eval_data = load_jsonl(args.eval)
    if args.subsample and args.subsample < len(eval_data):
        eval_data.sort(key=lambda x: (x["label"], x["topic"], x["dialogue"][:64]))
        eval_data = eval_data[: args.subsample]
    print(f"[load] eval n={len(eval_data)}")

    labels = [d["label"] for d in eval_data]
    dialogues = [d["dialogue"] for d in eval_data]

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    layer_ranges = [x.strip() for x in args.layer_ranges.split(",") if x.strip()]
    print(f"[config] ratios={ratios}  layer_ranges={layer_ranges}")

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

    # Baseline once (no mask).
    print("[eval] baseline (no mask)")
    preds_baseline = classify_batch(model, tokenizer, dialogues)
    acc_baseline = per_emotion_accuracy(preds_baseline, labels)

    rows: list[dict] = []
    for em_mask in EMOTIONS:
        full_list = by_emotion.get(em_mask, [])
        if not full_list:
            continue
        for lr_spec in layer_ranges:
            lr = build_layer_range(n_layers, lr_spec)
            lr_set = set(lr)
            filtered = [(l, i) for (l, i) in full_list if l in lr_set]
            for ratio in ratios:
                k = max(0, int(round(ratio * len(full_list))))
                # Take first k from the ratio*len(full_list) pool, filtered to this layer range.
                # Interpretation: rank order is preserved by filtering, and the 'ratio'
                # defines fraction of the emotion's neurons to ablate.
                sel = filtered[:k]
                if not sel:
                    print(
                        f"[skip] mask={em_mask} ratio={ratio} range={lr_spec}: "
                        f"no neurons in this layer range"
                    )
                    continue

                mask_map = build_mask_map(sel, n_layers, d_ff)
                print(
                    f"[eval] mask={em_mask} ratio={ratio} range={lr_spec} "
                    f"n_masked={len(sel)} layers={len(mask_map)}"
                )
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
                            "ratio": ratio,
                            "layer_range": lr_spec,
                            "n_masked": len(sel),
                            "emotion_label": em_label,
                            "n": n0,
                            "acc_normal": a0,
                            "acc_masked": a1,
                            "delta_pct": (a1 - a0) * 100.0,
                        }
                    )

    csv_path = out_dir / "ratio_layer_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "emotion_masked",
                "ratio",
                "layer_range",
                "n_masked",
                "emotion_label",
                "n",
                "acc_normal",
                "acc_masked",
                "delta_pct",
            ],
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
                "ratios": ratios,
                "layer_ranges": layer_ranges,
                "n_eval": len(eval_data),
            },
            fh,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RQ3: sweep mask_ratio x layer_range for each emotion."
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--ratios",
        default="0.01,0.02,0.03,0.04,0.05",
        help="Comma-separated fractions of an emotion's neuron set to mask.",
    )
    ap.add_argument(
        "--layer-ranges",
        default="Bottom,Middle,Top,All",
        help="Comma-separated layer-range names.",
    )
    ap.add_argument("--subsample", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
