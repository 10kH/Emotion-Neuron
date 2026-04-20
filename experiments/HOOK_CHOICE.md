# Activation hook choice — SwiGLU vs. paper's ReLU formulation

The paper (Sec 3.4, Eq. 4) defines `n^l_u = max(0, h^l_u)` — ReLU semantics.
Llama-3.1's FFN is SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
There is no ReLU anywhere in the model. Any hook choice is an interpretation,
not a direct restoration.

We implement three candidates:

| Mode | Captures | Rationale |
|---|---|---|
| `gated` (DEFAULT, TENTATIVE) | `silu(gate_proj(x)) * up_proj(x)` (pre-`down_proj` input) | Value actually consumed by `down_proj`; closest semantic analogue to a "neuron activation" |
| `silu_only` | `silu(gate_proj(x))` | Matches "post-nonlinearity" reading of Eq. 4 |
| `pre_silu` | `gate_proj(x)` | Matches "linear output before activation" literal reading |

All three threshold at `> 0` for "activated" to satisfy `max(0, ·)` semantics.
`silu_only` is strictly `>= 0` in practice (silu floors near `-0.28` but our
activation count is `> 0`, so tiny-negative values are correctly excluded —
consistent with the ReLU framing).

## Paper reference counts (Llama-3.1-8B, top-1% lowest H_u)

From paper Sec 4.1:

| Emotion | Paper count |
|---|---|
| anger | 1,882 |
| fear | 1,629 |
| disgust | 1,598 |
| sadness | 1,570 |
| happiness | 1,320 |
| surprise | 1,070 |

Sum = 9,069 total unique emotion neurons (not 4,588 — paper appears to
count per-emotion assignments; ours uses global top 1% = 4,588 on 8B).
This is itself an ambiguity in the paper — if the paper is per-emotion
top 1%, replace our global selection with a per-emotion split before
closing this comparison.

## Empirical comparison (fill in after smoke test)

Run on 1,000-sample subset of training split with each activation mode:

```bash
python experiments/neuron_selection.py --activation_mode gated     --train runs/split/train.jsonl --subsample 1000 --output runs/cmp/gated/
python experiments/neuron_selection.py --activation_mode silu_only --train runs/split/train.jsonl --subsample 1000 --output runs/cmp/silu/
python experiments/neuron_selection.py --activation_mode pre_silu  --train runs/split/train.jsonl --subsample 1000 --output runs/cmp/pre_silu/
```

Fill this table with per-emotion counts and `|Δ|` vs paper:

| Emotion | Paper | gated | silu_only | pre_silu | best mode |
|---|---|---|---|---|---|
| anger | 1,882 | — | — | — | — |
| disgust | 1,598 | — | — | — | — |
| fear | 1,629 | — | — | — | — |
| happiness | 1,320 | — | — | — | — |
| sadness | 1,570 | — | — | — | — |
| surprise | 1,070 | — | — | — | — |
| **avg \|Δ\|** | — | — | — | — | — |

Default `gated` will remain unless another mode shows strictly smaller
avg `|Δ|` on the full training split. If the paper's per-emotion-top-1%
interpretation is what they used, rerun with that selection variant
before declaring a winner.
