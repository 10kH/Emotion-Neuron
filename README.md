# Do Large Language Models Have "Emotion Neurons"? Investigating the Existence and Role

> **ACL 2025 Findings** | [Paper](paper/Emotion%20Neuron%20%28ACL%20Findings%202025%29.pdf) · [ACL Anthology](https://aclanthology.org/2025.findings-acl.806/)

Official data & code release for Lee et al., *Findings of ACL 2025*, pp. 15617–15639.

The paper identifies FFN neurons in Llama-3.1-Instruct that selectively respond to each of Ekman's six basic emotions, then analyzes their role by masking them and measuring the effect on emotion classification across layers and ratios.

## Contents

| Path | Purpose |
|:---|:---|
| `data/emoprism.json.gz` | **EmoPrism** — 293,725 single-emotion synthetic dialogues across 5,040 topics. |
| `data_generation/` | 5-step pipeline that produced EmoPrism (Paper Appendix B). |
| `experiments/` | Reconstructed analysis code — **RQ1** neuron selection, **RQ2** masking evaluation, **RQ3** ratio/layer sweep. |
| `VERIFICATION.md` | Dataset SHA-256, schema, integrity proof. |
| `paper/` | Paper PDF. |

## Installation

```bash
git clone https://github.com/10kH/Emotion-Neuron.git
cd Emotion-Neuron
pip install -r requirements.txt
huggingface-cli login          # gated access to meta-llama/Meta-Llama-3.1-*-Instruct
cd data && bash download.sh    # decompress + verify SHA-256
```

## Quick start — 3-minute smoke test

```bash
# 1. Deterministic 95/5 stratified split
python experiments/split_data.py --data data/emoprism.json --out runs/split

# 2. RQ1 — select emotion neurons (1,200-sample smoke test, ~3 min on A100)
CUDA_VISIBLE_DEVICES=0 python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train runs/split/train.jsonl --subsample 1200 \
    --output runs/smoke
```

Outputs `emotion_neurons.pt`, `counts_summary.csv`, `layer_distribution.csv`.

## Full reproduction

```bash
# RQ1 — full 8B run (~6 hr on one A100)
python experiments/neuron_selection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train runs/split/train.jsonl --output runs/sel_8b

# RQ2 — zero-ablate each emotion's neurons, compare accuracy
python experiments/evaluate_masking.py --run-dir runs/sel_8b \
    --eval runs/split/eval.jsonl --output runs/rq2_8b

# RQ3 — ratio × layer-range sweep
python experiments/masking_ratio_layer.py --run-dir runs/sel_8b \
    --eval runs/split/eval.jsonl --output runs/rq3_8b
```

For 70B, use `--model meta-llama/Meta-Llama-3.1-70B-Instruct` with 4 × A100 / `device_map="auto"`.

### Key flags

| Flag | Default | Purpose |
|:---|:---|:---|
| `--activation_mode` | `gated` | SwiGLU hook site (`gated` \| `silu_only` \| `pre_silu`) — see [HOOK_CHOICE.md](experiments/HOOK_CHOICE.md). |
| `--token_mode` | `last` | `last` = final assistant-content token (default); `all` ≈ 30× compute. |
| `--top_pct` | `0.01` | Fraction of globally lowest-entropy neurons kept as "emotion neurons". |

See [`experiments/README.md`](experiments/README.md) for compute estimates and artifact layout.

## Methodology (one-paragraph summary)

For each FFN neuron, count the tokens on which it fires (>0) per emotion, normalize across emotions, and compute the entropy of the resulting distribution. Neurons with the lowest entropy (top 1 % globally) are declared **emotion neurons**, each assigned to its argmax emotion. Zero-masking those neurons via a `down_proj` forward hook then measures their functional contribution to emotion classification. Because Llama-3.1 uses SwiGLU rather than the ReLU of the paper's Eq. 4, `h^l_u` is interpreted as `silu(gate_proj(x)) · up_proj(x)` by default — alternate interpretations are available behind `--activation_mode`.

## Citation

```bibtex
@inproceedings{lee-etal-2025-large,
    title     = "Do Large Language Models Have {``}Emotion Neurons{''}? Investigating the Existence and Role",
    author    = "Lee, Jaewook  and
                 Lee, Woojin  and
                 Kwon, Oh-Woog  and
                 Kim, Harksoo",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year      = "2025",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.findings-acl.806/",
    doi       = "10.18653/v1/2025.findings-acl.806",
    pages     = "15617--15639",
}
```

## License

- **Code** — [MIT](LICENSE)
- **EmoPrism dataset** — [CC-BY-4.0](LICENSE-DATA)

## Acknowledgments

Supported by NRF (RS-2025-00553041) and IITP (RS-2023-00216011, RS-2024-00338140) grants funded by the Korea Government (MSIT).
