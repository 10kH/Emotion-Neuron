"""Canonical stratified 95/5 split for the Emotion-Neuron labeled dataset.

User-locked spec:
  1. Sort the entire corpus by ``(label, topic, dialogue[:64])`` for
     deterministic ordering independent of the source file's row order.
  2. Stratify by ``label`` (the six paper emotions).
  3. Shuffle each per-label bucket with ``random.Random(seed)`` then take
     the first ``train_ratio`` fraction for train and the remainder for eval.

The same seed (=42) MUST be used for every downstream script; the
canonical split is what RQ2/RQ3 evaluate against.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_data(path: str) -> list[dict]:
    """Load either a JSON array or a JSONL file of dialogue records."""
    path = str(path)
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(
    data: list[dict],
    seed: int = 42,
    train_ratio: float = 0.95,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, eval_)`` per the user-locked deterministic spec."""
    # Sort first for determinism (independent of input order).
    data = sorted(
        data,
        key=lambda x: (x["label"], x["topic"], x["dialogue"][:64]),
    )

    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)

    rng = random.Random(seed)
    train: list[dict] = []
    eval_: list[dict] = []
    for label in sorted(by_label.keys()):
        items = by_label[label]
        indices = list(range(len(items)))
        rng.shuffle(indices)
        split_pt = int(len(items) * train_ratio)
        for idx in indices[:split_pt]:
            train.append(items[idx])
        for idx in indices[split_pt:]:
            eval_.append(items[idx])
    return train, eval_


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Deterministic stratified 95/5 split of the labeled Emotion-Neuron "
            "corpus. Sorts by (label, topic, dialogue[:64]) then stratifies by "
            "label with seed=42 by default."
        ),
    )
    ap.add_argument(
        "--data",
        required=True,
        help="Path to labeled JSON array or JSONL (expects keys: label, topic, dialogue).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory. Writes train.jsonl and eval.jsonl here.",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="RNG seed for stratified shuffle (default: 42)."
    )
    ap.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Train fraction per label (default: 0.95).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    print(f"Loaded {len(data)} records from {args.data}")

    label_counts = Counter(d["label"] for d in data)
    print(f"Overall label distribution: {dict(sorted(label_counts.items()))}")

    train, eval_ = stratified_split(data, seed=args.seed, train_ratio=args.train_ratio)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, str(out_dir / "train.jsonl"))
    write_jsonl(eval_, str(out_dir / "eval.jsonl"))

    train_counts = Counter(d["label"] for d in train)
    eval_counts = Counter(d["label"] for d in eval_)

    print(f"\nWrote {len(train)} train -> {out_dir / 'train.jsonl'}")
    print(f"Wrote {len(eval_)} eval  -> {out_dir / 'eval.jsonl'}")
    print(f"\nTrain label distribution: {dict(sorted(train_counts.items()))}")
    print(f"Eval  label distribution: {dict(sorted(eval_counts.items()))}")
    print(f"\nSeed={args.seed}  train_ratio={args.train_ratio}")


if __name__ == "__main__":
    main()
