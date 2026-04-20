# Data Integrity Verification — emoprism.json

**Paper**: _Do Large Language Models Have "Emotion Neurons"? Investigating the Existence and Role_
**Venue**: Findings of the Association for Computational Linguistics: ACL 2025, pages 15617–15639
**Authors**: Jaewook Lee*, Woojin Lee*, Oh-Woog Kwon, Harksoo Kim (*main contributors)

This document verifies that `data/emoprism.json` (decompressed from `data/emoprism.json.gz`) is the final synthetic dialogue dataset produced by the pipeline described in Appendix B of the paper.

## Integrity

- **SHA-256**: `886d80a549c81e4ca77a17c4f8605450bfe4ba557de35efb644f4c6711dddb9a`
- **Size (raw JSON)**: 376 MB
- **Size (gzipped)**: 94.6 MB
- **Item count**: 293,725 dialogues
- **Unique dialogues**: 293,725 / 293,725 (no duplicates)
- **Unique topics**: 5,040

## Schema (per item)

| Key | Type | Description |
|---|---|---|
| `topic` | string | One of 5,040 topics (augmented from 315 FITS base topics) |
| `dialogue` | string | Two-speaker dialogue, `A:` / `B:` turns separated by `\n` |
| `theme` | string | Intended emotion passed to gemini-1.5-flash-8b at generation |
| `claude-3-haiku-20240307` | string | Emotion label predicted by Claude |
| `gemini-1.5-flash` | string | Emotion label predicted by Gemini |
| `gpt-4o-mini` | string | Emotion label predicted by GPT |
| `label` | string | Final emotion label: majority-agreement (≥3 of 4 match) across `theme` + three model predictions. `unvalid` items were already filtered out and do not appear. |

All emotion values ∈ `{anger, disgust, fear, happiness, sadness, surprise}` (Ekman's six basic emotions, matching paper's focus).

## Label Distribution

| Emotion | Count | Percentage |
|---|---|---|
| anger | 49,015 | 16.69% |
| disgust | 49,232 | 16.76% |
| fear | 50,166 | 17.08% |
| happiness | 55,054 | 18.74% |
| sadness | 51,549 | 17.55% |
| surprise | 38,709 | 13.18% |
| **Total** | **293,725** | **100.00%** |

Sum check: 49015 + 49232 + 50166 + 55054 + 51549 + 38709 = **293,725** ✓

## Paper vs. file count discrepancy

The paper reports **293,715** dialogues (Appendix B.3, Table 1 total). The file contains **293,725** dialogues. The label sums in the paper's Table 1 also add to **293,725** — consistent with the file, not with the paper's stated total.

**Conclusion**: The paper's 293,715 figure is an off-by-10 typographical error. The correct total is 293,725, as supported by both the label sums and the raw file count. No additional filtering was performed after the step reported in Appendix B.3.

## Generation chain

```
315 base FITS topics
  │
  ▼  (gpt-4o-mini iterative augmentation, 4 rounds)
fits_step0.json (315) → fits_step1.json (315) → fits_step2.json (630)
  → fits_step3.json (1260) → fits_step4.json (2520)  [total pool: 5,040 topics]
  │
  ▼  (gemini-1.5-flash-8b, 6 emotions × 10 dialogues/topic)
synth_gemini_step{0..4}_10.json  [302,400 dialogues intended]
  │
  ▼  (3-model labeling: gpt-4o-mini, gemini-1.5-flash, claude-3-haiku-20240307)
synth_gemini_step{N}_10_labeled_{claude,gemini,gpt}.json
  │
  ▼  (majority vote ≥3 of 4 labels)
synth_gemini_step{N}_10_labeled_sum.json
  │
  ▼  (filter unvalid → step{N}.json)
step0.json step1.json step2.json step3.json step4.json
  │
  ▼  (step_merge.py)
data/emoprism.json  [293,725 dialogues, finalized]
```

## Reproduction of verification

```bash
# SHA-256 integrity
sha256sum data/emoprism.json
# Expected: 886d80a549c81e4ca77a17c4f8605450bfe4ba557de35efb644f4c6711dddb9a  data/emoprism.json

# Count
python3 -c "import json; assert len(json.load(open('data/emoprism.json'))) == 293725; print('PASS: 293,725 dialogues')"

# Label distribution
python3 -c "
import json; from collections import Counter
d = json.load(open('data/emoprism.json'))
c = Counter(x['label'] for x in d)
expected = {'anger': 49015, 'disgust': 49232, 'fear': 50166,
            'happiness': 55054, 'sadness': 51549, 'surprise': 38709}
assert dict(c) == expected, c
assert sum(c.values()) == 293725
print('PASS: label distribution matches paper Table 1')
"

# Uniqueness
python3 -c "
import json
d = json.load(open('data/emoprism.json'))
assert len(set(x['dialogue'] for x in d)) == 293725
assert len(set(x['topic'] for x in d)) == 5040
print('PASS: 293,725 unique dialogues, 5,040 unique topics')
"
```
