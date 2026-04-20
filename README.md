# Do Large Language Models Have "Emotion Neurons"? Investigating the Existence and Role

> **ACL 2025 Findings** | [Paper](paper/Emotion%20Neuron%20%28ACL%20Findings%202025%29.pdf) · [ACL Anthology](https://aclanthology.org/2025.findings-acl.806/)

Post-publication release of the synthetic dialogue dataset and a reconstruction of the experiment code associated with Lee et al., *Findings of ACL 2025*, pp. 15617–15639. See [Contributions](#contributions-for-this-release) below for the scope of this release.

The paper identifies FFN neurons in Llama-3.1-Instruct that selectively respond to each of Ekman's six basic emotions, then analyzes their role by masking them and measuring the effect on emotion classification across layers and ratios.

## Abstract

This study comprehensively explores whether there actually exist "emotion neurons" within large language models (LLMs) that selectively process and express certain emotions, and what functional role they play. Drawing on the representative emotion theory of the six basic emotions, we focus on six core emotions. Using synthetic dialogue data labeled with emotions, we identified sets of neurons that exhibit consistent activation patterns for each emotion. As a result, we confirmed that principal neurons handling emotion information do indeed exist within the model, forming distinct groups for each emotion, and that their distribution varies with model size and architectural depth. We then validated the functional significance of these emotion neurons by analyzing whether the prediction accuracy for a specific emotion significantly decreases when those neurons are artificially removed. We observed that in some emotions, the accuracy drops sharply upon neuron removal, while in others, the model's performance largely remains intact or even improves, presumably due to overlapping and complementary mechanisms among neurons. Furthermore, by examining how prediction accuracy changes depending on which layer range and at what proportion the emotion neurons are masked, we revealed that emotion information is processed in a multilayered and complex manner within the model.

## Repository layout

```
Emotion-Neuron/
├── data/
│   ├── emoprism.json.gz        # EmoPrism — 293,725 single-emotion synthetic dialogues (94.6 MB)
│   ├── emoprism_stats.json
│   └── download.sh             # decompress + verify SHA-256
├── data_generation/            # 5-step pipeline that produced EmoPrism (Paper Appendix B)
│   ├── topic_augmentation/     #   315 → 5,040 topics
│   ├── dialogue_synthesis/     #   302,400 dialogues (6 emotions × 10 × 5,040)
│   ├── labeling/               #   3-model labeling + majority vote
│   └── merging/                #   per-step filter + concat → emoprism.json
├── experiments/                # Reconstructed analysis code
│   ├── prompts.py              #   zero-shot classification prompt (Paper Fig 5)
│   ├── split_data.py           #   stratified 95/5 split
│   ├── utils.py                #   FFN hooks + zero-ablation
│   ├── neuron_selection.py     # RQ1 — entropy-based neuron selection
│   ├── evaluate_masking.py     # RQ2 — masking accuracy evaluation
│   ├── masking_ratio_layer.py  # RQ3 — ratio × layer-range sweep
│   ├── HOOK_CHOICE.md          #   SwiGLU-vs-ReLU interpretation
│   └── README.md               #   flags, compute estimates
├── paper/                      # Paper PDF
├── VERIFICATION.md             # dataset SHA-256, schema, integrity proof
├── LICENSE                     # MIT (code)
├── LICENSE-DATA                # CC-BY-4.0 (dataset)
├── requirements.txt
└── .env.example                # OPENAI/GOOGLE/ANTHROPIC keys template
```

## Installation

```bash
git clone https://github.com/10kH/Emotion-Neuron.git
cd Emotion-Neuron
pip install -r requirements.txt
huggingface-cli login          # gated access to meta-llama/Meta-Llama-3.1-*-Instruct
cd data && bash download.sh    # decompress + verify SHA-256
```

## Usage

```bash
# Deterministic 95/5 stratified split
python experiments/split_data.py --data data/emoprism.json --out runs/split

# RQ1 — select emotion neurons
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

For 70B, use `--model meta-llama/Meta-Llama-3.1-70B-Instruct`.

See [`experiments/README.md`](experiments/README.md) for all CLI flags (activation mode, token mode, subsampling) and the SwiGLU hook interpretation in [`experiments/HOOK_CHOICE.md`](experiments/HOOK_CHOICE.md).

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

## Contributions for this release

This repository is a **post-publication release** of artifacts associated with the paper cited above. It is not a claim to be the authoritative implementation used to produce the paper's published numbers. Attribution for this release specifically:

- **Dataset** — `data/emoprism.json.gz` (EmoPrism) was created, verified (SHA-256 documented in [`VERIFICATION.md`](VERIFICATION.md)), and released under [CC-BY-4.0](LICENSE-DATA) by **Woojin Lee**, who was the sole author of the data-generation work described in the paper's Appendix B.
- **Data-generation pipeline** (`data_generation/`) — authored by **Woojin Lee** as part of the paper's Appendix B pipeline. Released here with hardcoded API keys removed and path handling parameterized; no algorithmic logic was modified.

> [!NOTE]
> **The EmoPrism dataset and the data-generation pipeline (`data_generation/`) were authored solely by Woojin Lee.** No other co-author contributed to the construction or release of these artifacts, and sole copyright over this release is held by Woojin Lee.
- **Experiment code** (`experiments/`) — an **independent reconstruction** from the paper's Section 3.4 methodology, produced after publication specifically for this release. Where the paper's Eq. 4 (`n = max(0, h)`, ReLU semantics) diverges from Llama-3.1's SwiGLU architecture, the interpretation documented in [`experiments/HOOK_CHOICE.md`](experiments/HOOK_CHOICE.md) was adopted. Any numerical discrepancies between this reconstruction and the paper's reported results are attributable to interpretation choices of this release, not to the paper's authors.

> [!CAUTION]
> **The original experiment code written by Jaewook Lee is not included in this repository, nor was it consulted during reconstruction.** All code under `experiments/` is a clean-room implementation derived from the paper's published methodology.
- **Institutional approval** — release of the data-generation artifacts has been authorized by **ETRI**.
- **Paper PDF** (`paper/`) — redistributed under the ACL's CC-BY-4.0 license applicable to Findings of ACL 2025 materials.

### Disclaimer

> [!IMPORTANT]
> Co-authors who did not participate in preparing this release **retain all their rights under applicable copyright and authorship laws**. The inclusion of their names in the citation and in paper-related acknowledgments reflects only their roles as authors of the published paper — **not** as contributors to, endorsers of, or maintainers of this release. Any reservations from a co-author regarding any specific artifact in this repository should be directed to the contact below and will be addressed in good faith.

### Contact

For questions or concerns regarding this release, contact **Woojin Lee** — <writerwoody@gmail.com>.

## License

- **Code** — [MIT](LICENSE)
- **EmoPrism dataset** — [CC-BY-4.0](LICENSE-DATA)

## Acknowledgments

Supported by NRF (RS-2025-00553041) and IITP (RS-2023-00216011, RS-2024-00338140) grants funded by the Korea Government (MSIT).
