# Data generation pipeline

Five-step pipeline that produces `data/emoprism.json.gz` (293,725 labeled dialogues). Matches Appendix B of the paper.

All scripts read `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` from the environment — copy `.env.example` to `.env` at the repo root and `export $(grep -v '^#' .env | xargs)` before running anything in this directory.

Input/output file paths are configurable via the `INPUT_FILE` / `OUTPUT_FILE` / `SYNTH_DIR` / `BACKUP_DIR` environment variables. Defaults reproduce the original author paths (`/home/woody/workspace/Emotion-Neuron/data/`) for backward compatibility — override them to run in a different workspace.

## Stage 1 — Topic augmentation (`topic_augmentation/`) — Paper B.1

Expands 315 base FITS topics to **5,040 topics** via 4 rounds of gpt-4o-mini-driven paraphrastic substitution.

```
fits_step0.json (315)
  └─▶ fits_iterate_gpt.py ─▶ fits_step1.json (315)
  └─▶ fits_iterate_gpt.py ─▶ fits_step2.json (630)
  └─▶ fits_iterate_gpt.py ─▶ fits_step3.json (1,260)
  └─▶ fits_iterate_gpt.py ─▶ fits_step4.json (2,520)
  └─▶ concat                 fits_sum.json  (5,040)
  └─▶ fits_remove_duplicates.py ─▶ fits_sum_deduped.json
```

## Stage 2 — Dialogue synthesis (`dialogue_synthesis/`) — Paper B.2

For each of the 5,040 topics × 6 emotions, generate 10 dialogues with gemini-1.5-flash-8b → **302,400 target dialogues**.

```
fits_step{N}.json
  └─▶ synth_gemini.py      ─▶ synth_gemini_step{N}_10.json
  └─▶ synth_check.py       ─▶ integrity counts
  └─▶ synth_gemini_add.py  ─▶ top up missing (topic, emotion) pairs
```

Run once per stage (step0 through step4). `synth_check.py` reports any `(topic, emotion)` pairs that are short of 10 dialogues; `synth_gemini_add.py` tops them up.

## Stage 3 — Labeling (`labeling/`) — Paper B.3

Three LLM labelers (gpt-4o-mini, gemini-1.5-flash, claude-3-haiku-20240307) independently predict the emotion for each generated dialogue. Unknown/error rows are re-labeled up to 5 times each.

```
synth_gemini_step{N}_10.json
  ├─▶ labeling_gpt.py       ─▶ synth_gemini_step{N}_10_labeled_gpt.json
  ├─▶ labeling_gemini.py    ─▶ synth_gemini_step{N}_10_labeled_gemini.json
  └─▶ labeling_claude.py    ─▶ synth_gemini_step{N}_10_labeled_claude.json

  then, for each labeler (re-run until converged):
  ├─▶ labeling_unknown_gpt.py
  ├─▶ labeling_unknown_gemini.py
  └─▶ labeling_unknown_claude.py

  then:
  └─▶ labeling_sum.py       ─▶ synth_gemini_step{N}_10_labeled_sum.json
      (majority vote ≥3 of 4 labels among
       theme + gpt-4o-mini + gemini-1.5-flash + claude-3-haiku-20240307;
       otherwise label = "unvalid")
  └─▶ labeling_split.py     ─▶ step{N}.json
      (drops "unvalid" rows)
```

## Stage 4 — Merging (`merging/`)

Concatenates `step0.json … step4.json` into `data/emoprism.json`, computes dataset statistics, and optionally screens for duplicates.

```
step0.json step1.json step2.json step3.json step4.json
  └─▶ step_merge.py ─▶ data/emoprism.json + data/emoprism_stats.json
  └─▶ emoprism_screen.py  ─▶ duplicate report (none in final dataset)
```

## Known issues

These are preserved as-is because `data/emoprism.json.gz` was produced before they were identified; rerunning the pipeline from scratch is not part of this release. Documented here for transparency.

1. **`labeling_split.py`** hardcodes `step4` in the output-file path regardless of input. The authors re-ran it five times with manual edits; an automated sweep would need to parametrize the step index.
2. **`labeling_sum.py`** breaks on the first label that appears ≥3 times in `{theme, claude, gemini, gpt}`. With standard Python dict ordering (insertion order), the `theme` label is checked first, so ties favor `theme` — this matches the authors' intent but is order-dependent rather than explicit.
3. **`fits_iterate_gpt.py`** passes the entire `existing_topics` set in each prompt. For the largest iteration (2,520 topics in context) this inflates token usage and occasionally truncates; the authors manually verified uniqueness after the fact via `fits_remove_duplicates.py`.

## Reproduction cost

Approximate wall-clock time to reproduce the full 5-stage pipeline from scratch (rate limits dominate):

- Topic augmentation: ~1 hr (OpenAI)
- Dialogue synthesis (all 5 stages, 302,400 dialogues): ~40 hr (Gemini)
- Labeling × 3 labelers × 302,400 dialogues + re-labeling passes: ~120 hr combined (rate-limited)
- Merging: minutes

**The published `data/emoprism.json.gz` is the canonical artifact** — re-running the pipeline produces a different dataset (LLM outputs are stochastic) even with the same prompts.
