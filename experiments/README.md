# Emotion-Neuron experiments

Reconstruction of the experimental code for ACL 2025 Findings
"Do LLMs Have Emotion Neurons?". The user produced the labeled corpus
(`data/emoprism.json.gz`) via `data_generation/`; this directory contains
the RQ1/RQ2/RQ3 analysis code.

See [`HOOK_CHOICE.md`](./HOOK_CHOICE.md) for the SwiGLU-vs-ReLU
interpretation (the primary pre-decided ambiguity).

## Prerequisites

- HuggingFace access to `meta-llama/Meta-Llama-3.1-8B-Instruct` and
  (optional) `meta-llama/Meta-Llama-3.1-70B-Instruct`.
- PyTorch + `transformers` with Llama-3.1 support, `tqdm`.
- GPU memory:
  - 8B FP16: ~16 GB VRAM (fits on one A100-80GB with headroom for
    hook captures).
  - 70B FP16: ~140 GB → needs 2–4 × A100 80GB with `device_map="auto"`.
- Disk: ~3 GB for the selection run output (activation tensors are
  streamed, only counters are persisted).

## Pipeline

```
data/emoprism.json.gz
    |
    v
split_data.py ---> runs/split/{train,eval}.jsonl     (95/5, stratified, seed=42)
    |
    v
neuron_selection.py ---> runs/sel/{emotion_neurons.pt, counts_summary.csv, ...}
    |
    +--> evaluate_masking.py     ---> runs/rq2/masking_results.csv      (RQ2)
    |
    +--> masking_ratio_layer.py  ---> runs/rq3/ratio_layer_results.csv  (RQ3)
```

### 1. Split the corpus

```bash
# Decompress data/emoprism.json.gz to data/emoprism.json first if needed.
python experiments/split_data.py \
    --data data/emoprism.json \
    --out  runs/split \
    --seed 42
```

Canonical split key: `(label, topic, dialogue[:64])`, stratified by
label with `random.Random(42)`. Any deviation breaks reproducibility
with the paper's numbers.

### 2. Select emotion neurons (RQ1)

Smoke-test (1k samples, default `gated` hook, last-token aggregation):

```bash
python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train runs/split/train.jsonl \
    --subsample 1000 \
    --output runs/sel_smoke
```

Full run:

```bash
python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train runs/split/train.jsonl \
    --output runs/sel
```

Hook/token flags:
- `--activation_mode {gated, silu_only, pre_silu}` — see HOOK_CHOICE.md.
- `--token_mode {last, all}` — `last` is the default (single-word
  emotion proxy). `all` counts activations on **every** input token;
  expect ~30× compute (see below).

### 3. RQ2: mask per emotion

```bash
python experiments/evaluate_masking.py \
    --run-dir runs/sel \
    --eval    runs/split/eval.jsonl \
    --model   meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output  runs/rq2
```

Output: `masking_results.csv` — one row per `(emotion_masked,
emotion_label)`: `acc_normal`, `acc_masked`, `delta_pct`.

### 4. RQ3: ratio × layer-range sweep

```bash
python experiments/masking_ratio_layer.py \
    --run-dir runs/sel \
    --eval    runs/split/eval.jsonl \
    --model   meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output  runs/rq3 \
    --ratios  0.01,0.02,0.03,0.04,0.05 \
    --layer-ranges Bottom,Middle,Top,All
```

Layer ranges: `Bottom=[0, L/3)`, `Middle=[L/3, 2L/3)`,
`Top=[2L/3, L)`, `All=[0, L)`.

## Compute estimates

| Run | Model | Dataset | Token mode | Time (A100) |
|---|---|---|---|---|
| smoke | 8B | 1k | last | ~5 min |
| full RQ1 | 8B | ~279k | last | ~6 hr |
| full RQ1 | 8B | ~279k | all | ~7 days (**not recommended**) |
| full RQ1 | 70B | ~279k | last | ~24 hr (4 × A100 FP16) |
| RQ2 | 8B | ~14k eval | — | ~45 min |
| RQ3 | 8B | ~14k eval | — | ~4 hr |

All estimates assume batch-size 1 (which is what this code uses — see
`evaluate_masking.classify_batch`). Batching across prompts would
reduce wall time but complicates hook state.

## Output artifact layout

```
runs/
  split/
    train.jsonl
    eval.jsonl
  sel/
    emotion_neurons.pt       # torch dict — see neuron_selection.py
    counts_summary.csv       # emotion, total_count
    layer_distribution.csv   # layer, emotion, count
    run_config.json          # reproducibility metadata
  rq2/
    masking_results.csv
    run_config.json
  rq3/
    ratio_layer_results.csv
    run_config.json
```

## Known interpretation choices

1. **Activation semantics**: `gated` (SwiGLU's pre-`down_proj`
   product) is the default. See [`HOOK_CHOICE.md`](./HOOK_CHOICE.md).
2. **Token aggregation**: `last` = last assistant content token
   immediately before EOS (`<|eot_id|>` / `<|end_of_text|>`). The
   paper is ambiguous here; `all` is available with the compute
   caveat above.
3. **Neuron ranking**: **global** across all `L × d_ff` neurons, not
   per-layer. Paper doesn't specify; global top-1% matches the
   "structural" reading.
4. **Masking hook site**: `down_proj`'s `forward_pre_hook` — the
   ablation is applied to the exact tensor defined by the `gated`
   activation mode, keeping RQ2/RQ3 semantics consistent with RQ1.
5. **Paper reference counts vs ours**: Paper reports per-emotion
   counts that sum to 9,069 on 8B, but `ceil(0.01 × 32 × 14336) = 4588`.
   This is a paper-internal ambiguity captured in HOOK_CHOICE.md.
