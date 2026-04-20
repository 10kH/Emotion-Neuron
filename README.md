# Emotion-Neuron — ACL 2025 Findings

Official data & code release for:

> **Do Large Language Models Have "Emotion Neurons"? Investigating the Existence and Role**
> Jaewook Lee\*, Woojin Lee\*, Oh-Woog Kwon, Harksoo Kim
> *Findings of the Association for Computational Linguistics: ACL 2025*, pp. 15617–15639
> \*Main contributors

[Paper PDF](paper/Emotion%20Neuron%20%28ACL%20Findings%202025%29.pdf)

## Overview

This repository contains:

- **Synthetic dialogue dataset** — 293,725 dialogues across 5,040 topics, labeled with one of six emotions (anger, disgust, fear, happiness, sadness, surprise). Used to identify emotion-selective neurons in Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct.
- **Data generation pipeline** — topic augmentation → emotion-conditioned dialogue synthesis → multi-model labeling → majority-vote filtering, exactly as described in Appendix B of the paper.
- **Experiment code** — reconstruction of the neuron selection (RQ1), masking evaluation (RQ2), and layer/ratio analysis (RQ3) procedures.

## Repository layout

```
├── data/                      Final dataset (emoprism.json.gz, 94.6 MB)
│   ├── emoprism.json.gz
│   ├── emoprism_stats.json
│   └── download.sh
├── data_generation/           Five-step pipeline (Paper Appendix B)
│   ├── topic_augmentation/
│   ├── dialogue_synthesis/
│   ├── labeling/
│   └── merging/
├── experiments/               RQ1 / RQ2 / RQ3 reconstruction
├── paper/                     Paper PDF
├── VERIFICATION.md            SHA-256, label distribution, schema proof
├── LICENSE                    MIT (code)
├── LICENSE-DATA               CC-BY-4.0 (dataset)
├── requirements.txt
└── .env.example               Template for API keys
```

## Data

Decompress and verify integrity:

```bash
cd data && bash download.sh
```

`download.sh` gunzips `emoprism.json.gz` and checks the SHA-256 against
`886d80a549c81e4ca77a17c4f8605450bfe4ba557de35efb644f4c6711dddb9a`.
See [VERIFICATION.md](VERIFICATION.md) for full schema and integrity details.

## Data generation (reproduction)

Install dependencies and configure API keys:

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)
```

Each subdirectory under `data_generation/` corresponds to one pipeline stage:

| Directory | Stage | Paper section |
|---|---|---|
| `topic_augmentation/` | 315 base FITS topics → 5,040 topics (gpt-4o-mini) | B.1 |
| `dialogue_synthesis/` | 6 emotions × 10 dialogues/topic (gemini-1.5-flash-8b) | B.2 |
| `labeling/` | 3-model labeling + unknown/error re-labeling + majority vote | B.3 |
| `merging/` | Per-step `unvalid` filter + merge into final `emoprism.json` | B.3 |

Each script accepts `INPUT_FILE` / `OUTPUT_FILE` environment variables; see the scripts for per-file details.

## Experiments

```bash
# 1. Stratified 95/5 split (279,039 train / 14,686 eval)
python experiments/split_data.py --data data/emoprism.json --out experiments/runs/split/

# 2. Neuron selection (RQ1)
python experiments/neuron_selection.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train experiments/runs/split/train.jsonl \
  --output experiments/runs/8b/

# 3. Masking evaluation (RQ2)
python experiments/evaluate_masking.py --run-dir experiments/runs/8b/

# 4. Layer/ratio sweep (RQ3)
python experiments/masking_ratio_layer.py --run-dir experiments/runs/8b/
```

See [experiments/README.md](experiments/README.md) for compute estimates, hook interpretation choices, and the paper-vs-reconstruction reference in [experiments/HOOK_CHOICE.md](experiments/HOOK_CHOICE.md).

## Citation

```bibtex
@inproceedings{lee2025emotion,
  title     = {Do Large Language Models Have ``Emotion Neurons''? Investigating the Existence and Role},
  author    = {Lee, Jaewook and Lee, Woojin and Kwon, Oh-Woog and Kim, Harksoo},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {15617--15639},
  year      = {2025}
}
```

## License

- **Code** — [MIT](LICENSE)
- **Dataset** (`data/emoprism.json.gz`) — [CC-BY-4.0](LICENSE-DATA)

## Acknowledgments

Supported by the National Research Foundation of Korea (NRF, RS-2025-00553041) and Institute of Information & Communications Technology Planning & Evaluation (IITP, RS-2023-00216011 and RS-2024-00338140), funded by the Korea Government (MSIT).
