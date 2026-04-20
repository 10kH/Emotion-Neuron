# Do Large Language Models Have "Emotion Neurons"? Investigating the Existence and Role

> **ACL 2025 Findings** | Official Repository

[📄 Paper (PDF)](paper/Emotion%20Neuron%20%28ACL%20Findings%202025%29.pdf) · [🌐 ACL Anthology](https://aclanthology.org/2025.findings-acl.806/) · [📊 Dataset (EmoPrism)](data/emoprism.json.gz) · [🔬 Experiments](experiments/)

## Abstract

This study comprehensively explores whether there actually exist "emotion neurons" within large language models (LLMs) that selectively process and express certain emotions, and what functional role they play. Drawing on the representative emotion theory of the six basic emotions, we focus on six core emotions. Using synthetic dialogue data labeled with emotions, we identified sets of neurons that exhibit consistent activation patterns for each emotion. As a result, we confirmed that principal neurons handling emotion information do indeed exist within the model, forming distinct groups for each emotion, and that their distribution varies with model size and architectural depth. We then validated the functional significance of these emotion neurons by analyzing whether the prediction accuracy for a specific emotion significantly decreases when those neurons are artificially removed. We observed that in some emotions, the accuracy drops sharply upon neuron removal, while in others, the model's performance largely remains intact or even improves, presumably due to overlapping and complementary mechanisms among neurons. Furthermore, by examining how prediction accuracy changes depending on which layer range and at what proportion the emotion neurons are masked, we revealed that emotion information is processed in a multilayered and complex manner within the model.

## Research Questions

The paper investigates three research questions through progressively deeper analyses of the internal emotional representations in Llama-3.1-Instruct models:

| RQ | Question | Method |
|:---|:---------|:-------|
| **RQ1** | Do emotion-neuron groups exist within LLMs that are responsible for particular emotions? | Per-neuron activation-entropy selection over Ekman's six emotions ([`neuron_selection.py`](experiments/neuron_selection.py)) |
| **RQ2** | What are the functional effects of manipulating those neurons on emotion prediction? | Zero-ablation via `down_proj` hook + accuracy comparison ([`evaluate_masking.py`](experiments/evaluate_masking.py)) |
| **RQ3** | How does masking proportion × layer range shape the decline? | Sweep 1–5 % of the neuron set across Bottom / Middle / Top / All layers ([`masking_ratio_layer.py`](experiments/masking_ratio_layer.py)) |

## Key Results

### Emotion-neuron counts per model (RQ1, paper § 4.1)

Top-1 % lowest-entropy FFN neurons across all layers, grouped by dominant emotion:

| Emotion  | Llama-3.1-8B | Llama-3.1-70B |
|:---------|-------------:|--------------:|
| anger    | 1,882        | 7,910         |
| fear     | 1,629        | —             |
| disgust  | 1,598        | —             |
| sadness  | 1,570        | **8,314**     |
| happiness| 1,320        | —             |
| surprise | 1,070        | 4,976         |

The relative ordering shifts with scale: in 8B, **anger** dominates; in 70B, **sadness** overtakes it. Emotion neurons cluster in the **middle layers** for 8B but spread across the entire stack in 70B.

### Masking effect on classification accuracy (RQ2, paper § 4.2)

| Model | Target emotion | Normal acc. | After masking | Δ |
|:------|:---------------|------------:|--------------:|------:|
| Llama-3.1-8B  | **disgust** | 96.11 % | **37.22 %** | **−58.89 %p** |
| Llama-3.1-70B | **surprise**| 93.17 % | **73.83 %** | **−19.34 %p** |
| Llama-3.1-8B  | happiness   | —       | ≈ no change | ≈ 0 %p |
| Llama-3.1-70B | disgust     | —       | ≈ no change | ≈ 0 %p |

Removing emotion-specific neurons collapses accuracy for *some* emotions (disgust@8B, surprise@70B) but leaves others intact — evidence of **overlapping and complementary mechanisms** across neuron groups.

### Ratio × layer-range sweep (RQ3, paper § 4.3)

Masking 1–5 % of an emotion's neurons within {Bottom, Middle, Top, All} layer thirds reveals:

- **Surprise** depends on neurons distributed across the full stack: `−4.15 %p` in Bottom only, `−10.46 %p` in Middle, `−20.75 %p` in AllLayers.
- **Disgust** undergoes final processing primarily in Top layers: `−1.63 %p` in Top vs. ≈ 0 %p elsewhere.
- **Fear** is broadly distributed; partial ablation is robust but AllLayers drops `−2.44 %p`.

See [`experiments/HOOK_CHOICE.md`](experiments/HOOK_CHOICE.md) for the precise paper-vs-reconstruction numeric reference table.

## Methodology Overview

For each FFN neuron `u = (layer l, index i)` and emotion `e ∈ {anger, disgust, fear, happiness, sadness, surprise}`:

```
f_{u,e}   = #{activated tokens} on e-labeled dialogue tokens           (Eq. 5)
P_{u,e}   = f_{u,e} / T_e                                               (Eq. 5)
P̂_{u,e}  = P_{u,e} / Σ_{e'} P_{u,e'}                                   (Eq. 6)
H_u       = −Σ_e P̂_{u,e} · log P̂_{u,e}                                (Eq. 7)
```

The top 1 % of neurons with lowest entropy are deemed **emotion neurons**, each assigned to `argmax_e P̂_{u,e}`. Because Llama-3.1 uses **SwiGLU** (not the ReLU of Eq. 4), we interpret `h^l_u` as the pre-`down_proj` intermediate `silu(gate_proj(x)) · up_proj(x)` and threshold at `> 0` for "activated". Three hook variants (`gated`, `silu_only`, `pre_silu`) are implemented — see [`experiments/HOOK_CHOICE.md`](experiments/HOOK_CHOICE.md).

## EmoPrism Dataset

**EmoPrism** (data/emoprism.json.gz) is a 293,725-dialogue synthetic corpus produced specifically for emotion-neuron exploration. Each dialogue contains exactly one Ekman basic emotion — a design decision that contrasts with existing conversational datasets (DailyDialog, MELD, EmpatheticDialogues) where multiple emotions coexist per conversation.

| Property | Value |
|:---------|------:|
| Total dialogues | **293,725** |
| Unique topics | **5,040** |
| Emotions | 6 (Ekman basic) |
| Single-emotion per dialogue | ✓ |
| Per-emotion class balance | 13.18 % – 18.74 % |
| SHA-256 (decompressed JSON) | `886d80a5…dddb9a` |
| Compressed size | 94.6 MB |

Label distribution (final, after 3-model majority-vote filtering):

| Emotion | Count | % |
|:---|---:|---:|
| happiness | 55,054 | 18.74 % |
| sadness   | 51,549 | 17.55 % |
| fear      | 50,166 | 17.08 % |
| disgust   | 49,232 | 16.76 % |
| anger     | 49,015 | 16.69 % |
| surprise  | 38,709 | 13.18 % |
| **Total** | **293,725** | **100.00 %** |

Full schema, generation chain, and integrity proof are in [`VERIFICATION.md`](VERIFICATION.md).

### Construction pipeline (paper Appendix B)

```
 315 base FITS topics
   │
   ▼  [B.1]  topic augmentation (gpt-4o-mini, 4 rounds)
 5,040 topics
   │
   ▼  [B.2]  emotion-conditioned dialogue synthesis (gemini-1.5-flash-8b)
 302,400 dialogues  (6 emotions × 10 dialogues × 5,040 topics)
   │
   ▼  [B.3]  3-model labeling (gpt-4o-mini, gemini-1.5-flash, claude-3-haiku)
          majority vote  ≥ 3 of {theme, gpt, gemini, claude} → keep
 293,725 dialogues  (8,685 filtered out)
   │
   ▼
 data/emoprism.json.gz
```

## Repository Layout

```
Emotion-Neuron/
├── data/
│   ├── emoprism.json.gz        # Final dataset (94.6 MB, SHA-256 verified)
│   ├── emoprism_stats.json     # Per-step label distribution
│   └── download.sh             # Decompress + verify
├── data_generation/            # Paper Appendix B pipeline (authors' scripts)
│   ├── topic_augmentation/     #   B.1 — 315 → 5,040 topics
│   ├── dialogue_synthesis/     #   B.2 — 302,400 dialogues
│   ├── labeling/               #   B.3 — 3-model + majority vote
│   ├── merging/                #   step_merge → emoprism.json
│   └── README.md
├── experiments/                # Reconstructed analysis code
│   ├── prompts.py              # Zero-shot classification prompt (Fig 5)
│   ├── split_data.py           # Stratified 95/5 split (seed 42)
│   ├── utils.py                # FFN hooks (SwiGLU-aware) + masking
│   ├── neuron_selection.py     # RQ1 — entropy-based emotion-neuron selection
│   ├── evaluate_masking.py     # RQ2 — zero-ablation accuracy comparison
│   ├── masking_ratio_layer.py  # RQ3 — ratio × layer sweep
│   ├── HOOK_CHOICE.md          # SwiGLU-vs-ReLU interpretation
│   └── README.md               # Reproduction guide + compute estimates
├── paper/
│   └── Emotion Neuron (ACL Findings 2025).pdf
├── VERIFICATION.md             # Dataset integrity (SHA-256, schema, counts)
├── LICENSE                     # MIT (code)
├── LICENSE-DATA                # CC-BY-4.0 (EmoPrism dataset)
├── requirements.txt
└── .env.example                # OPENAI/GOOGLE/ANTHROPIC keys template
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA GPU with ≥ 16 GB VRAM (Llama-3.1-8B FP16) or ≥ 140 GB aggregate (Llama-3.1-70B FP16, 2-4 × A100/H100)
- HuggingFace access to `meta-llama/Meta-Llama-3.1-8B-Instruct` (and optionally `-70B-Instruct`) — both are gated models

### Setup

```bash
# Clone
git clone https://github.com/10kH/Emotion-Neuron.git
cd Emotion-Neuron

# Virtual environment (conda or venv)
conda create -n emoprism python=3.11 -y
conda activate emoprism

# Install dependencies
pip install -r requirements.txt

# HuggingFace login (for gated Llama models)
huggingface-cli login

# Decompress + verify dataset
cd data && bash download.sh && cd ..
```

### API keys (only needed to re-run the data-generation pipeline)

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)
```

## Usage

### Quick start — RQ1 smoke test (≈ 3 min on one A100-80GB)

```bash
# 1. Produce canonical 95/5 stratified split (deterministic, seed=42)
python experiments/split_data.py \
    --data data/emoprism.json \
    --out  runs/split

# 2. RQ1 — Select emotion neurons on 1,200 stratified samples (smoke test)
CUDA_VISIBLE_DEVICES=0 python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train runs/split/train.jsonl \
    --subsample 1200 \
    --output runs/smoke
```

Outputs: `emotion_neurons.pt` · `counts_summary.csv` · `layer_distribution.csv` · `run_config.json`.

### Full runs (reproduce paper numbers)

```bash
# RQ1 — full run (8B, last-token, ~6 h on A100)
CUDA_VISIBLE_DEVICES=0 python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train runs/split/train.jsonl \
    --output runs/sel_8b

# RQ2 — masking evaluation (~45 min)
python experiments/evaluate_masking.py \
    --run-dir runs/sel_8b \
    --eval runs/split/eval.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output runs/rq2_8b

# RQ3 — ratio × layer sweep (~4 h)
python experiments/masking_ratio_layer.py \
    --run-dir runs/sel_8b \
    --eval runs/split/eval.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output runs/rq3_8b \
    --ratios 0.01,0.02,0.03,0.04,0.05 \
    --layer-ranges Bottom,Middle,Top,All

# 70B scale (4 × A100-80GB FP16, device_map="auto")
python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --train runs/split/train.jsonl \
    --output runs/sel_70b
```

### Key CLI options

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--activation_mode` | `gated` | `gated` = `silu(gate_proj)·up_proj`; `silu_only` = post-SiLU; `pre_silu` = raw `gate_proj`. See [HOOK_CHOICE.md](experiments/HOOK_CHOICE.md). |
| `--token_mode` | `last` | `last` = final assistant content token before EOS; `all` = every token (≈ 30× compute). |
| `--top_pct` | `0.01` | Fraction of lowest-entropy neurons kept as "emotion neurons" (globally ranked). |
| `--subsample` | `0` (use all) | Stratified per-label smoke-test subsample. |

## Compute Estimates

| Run | Model | Dataset | Token mode | Approx. time |
|:----|:------|:--------|:----------|:-------------|
| Smoke | Llama-3.1-8B | 1,200 stratified | last | ~3 min (A100) |
| RQ1 full | Llama-3.1-8B | 279,039 train | last | ~6 hr (A100) |
| RQ1 full | Llama-3.1-8B | 279,039 train | all | ~7 days ⚠ |
| RQ1 full | Llama-3.1-70B | 279,039 train | last | ~24 hr (4×A100) |
| RQ2 | Llama-3.1-8B | 14,686 eval | — | ~45 min |
| RQ3 | Llama-3.1-8B | 14,686 eval | — | ~4 hr |

## Dataset Generation (Optional)

The published `data/emoprism.json.gz` is the canonical artifact. Re-running the 5-step pipeline produces a *different* dataset (LLM outputs are stochastic), but the code is provided for transparency:

```bash
cd data_generation
# See data_generation/README.md for the full 5-stage sequence:
#   1. topic_augmentation/fits_iterate_gpt.py
#   2. dialogue_synthesis/synth_gemini.py (→ synth_check → synth_gemini_add)
#   3. labeling/labeling_{gpt,gemini,claude}.py (+ labeling_unknown_* passes)
#   4. labeling/labeling_sum.py + labeling_split.py
#   5. merging/step_merge.py
```

Total wall-clock for full pipeline: ≈ 160 hr, dominated by API rate limits.

## Citation

```bibtex
@inproceedings{lee-etal-2025-large,
    title     = "Do Large Language Models Have {``}Emotion Neurons{''}? Investigating the Existence and Role",
    author    = "Lee, Jaewook  and
                 Lee, Woojin  and
                 Kwon, Oh-Woog  and
                 Kim, Harksoo",
    editor    = "Che, Wanxiang  and
                 Nabende, Joyce  and
                 Shutova, Ekaterina  and
                 Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month     = jul,
    year      = "2025",
    address   = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.findings-acl.806/",
    doi       = "10.18653/v1/2025.findings-acl.806",
    pages     = "15617--15639",
}
```

## License

- **Code** — [MIT](LICENSE)
- **EmoPrism dataset** (`data/emoprism.json.gz`) — [CC-BY-4.0](LICENSE-DATA)

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea Government (MSIT) (RS-2025-00553041, *Enhancement of Rational and Emotional Intelligence of Large Language Models for Implementing Dependable Conversational Agents*), and by Institute of Information & Communications Technology Planning & Evaluation (IITP) grants (RS-2023-00216011, RS-2024-00338140), also funded by the Korea Government (MSIT).
