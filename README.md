# Flawline

Contrastive speech analytics (hackathon Track C). Real recorded speakers read a known text; controlled delivery flaws (pace, pauses, intonation, volume, fluency, clarity, text fidelity) are injected at known places and strengths, so every label is ground truth. An engine then finds the flaws blind, says where and why, and scores the delivery.

- **Dataset:** 564 clips, 9.97 h, 10 baseline recordings (8 VCTK voices + 2 excerpts of JFK's 1962 Rice University address), 15 flaws × 5 levels, 6 recording conditions. Datasheet: [`DATASHEET.md`](DATASHEET.md). Licences: [`flawline-dataset/LICENSES.md`](flawline-dataset/LICENSES.md).
- **Download:** _link to the released zip (with `checksums.sha256`) goes here._
- **Video:** _link goes here._
- **Status, known failures, history:** [`PROGRESS.md`](PROGRESS.md).

## Run it (3 commands)

```bash
make setup                  # pinned dependencies (requirements.lock) + spaCy model; needs ffmpeg
make baselines dataset      # download VCTK + the JFK excerpts, align them, generate every clip   (or unzip the released dataset into flawline-dataset/)
make app                    # dashboard: http://localhost:8501
```

Docker: `docker build -t flawline . && docker run -p 8501:8501 flawline`. The image holds the code and dependencies only; put the released `takes/` (baseline audio + alignment JSON) in `flawline-dataset/takes/` for the engine, because re-aligning on another machine moves word timings by up to 0.37 s. `make eval` re-runs the checks below on the train and dev splits (never the test split).

## What the engine does

Three ways to know what "good" is, chosen in the dashboard:

| Mode | Needs | Use |
|---|---|---|
| **General** (default) | the text read + the audio | expectations from the speaker's own clip, the transcript (pause norms by boundary type, ASR vs text) and population norms from clean speakers |
| **Same speaker** | the speaker's own clean reading | upper bound; the validated headline |
| **Another speaker** (experimental) | other speakers reading the same sentences | a panel of references must agree before a flaw is reported |

Pipeline: 16 kHz ingest and quality gate → (reference modes) degrade the clean reference to the participant's bandwidth/noise/reverb → DTW alignment, lag curve read as insertions/deletions and tempo → one detector per flaw → cross-detector arbitration → isotonic severity → rubric score `S = 100·exp(−P/35)` per category, genre-weighted → plain-language explanation with the measured numbers. The generator and the engine never import each other.

## Results (honest)

Event F1 at IoU 0.5 against the injected truth. Thresholds are fitted on **train** only; **dev** is B03; **extra** is the two JFK baselines (never tuned on); the **test** split (B05, B08) has not been run yet.

| Mode | Train | Dev | JFK 1962 (extra) | Flaws counted |
|---|---|---|---|---|
| Same speaker (headline) | **0.59** | **0.71** | 0.59 | 11 |
| General | 0.31 | 0.20 | 0.07 | 8 |
| Another speaker (experimental) | 0.23 | 0.22 | n/a | 8 |

- **Headline flaws (11):** FADE, FILLER, MONOTONE, PACE_FAST, PACE_SLOW, PAUSE_BAD, PAUSE_LOST, SHOUT, SLUR, WORD_SKIP, WORD_SWAP. FILLER and PAUSE_BAD are leakage-flagged (see below).
- **Experimental, excluded from headline metrics and hidden by default:** EMPH_FLAT (VCTK voices are too flat for the flattening to scale with level; F1 0.09), REPEAT (0.13), UPTALK (0.25), RARE_HESIT (non-monotone dose-response, leakage 0.70).
- **General mode does not reach the 0.4 bar** and does not transfer to the noisy 1962 recording (ASR errors and clean-speaker norms produce false flags). SLUR and MONOTONE are undetectable without a reference (effect about the size of natural variation).
- **Score acceptance tests (same speaker):** score falls with level for 12 of 13 scored flaws (RARE_HESIT fails; EMPH_FLAT and UPTALK hidden); false flags under noise/phone/room/codec 0.35 per minute (target ≤ 0.5, passes); worst clean-speech score shift 13.5 pts (target < 3, **fails**; mean shifts ≤ 1.6); points lost in other categories 3.7 (target < 5 met, < 2 not).
- **Leakage audit** (classifier sees only editing artifacts): AUC 0.598 on held-out test windows (mark 0.60), 0.616 leave-one-speaker-out, about 0.7 for RARE_HESIT, FILLER, PAUSE_BAD, EMPH_FLAT. Reported as is; the audit definition changed during the project (v2, see `PROGRESS.md`).
- **Not done:** the by-ear realism rating of the injected flaws (nobody has listened to them yet), a human panel, speakers over 45 or with slang, a speech-accent corpus.

## Layout

```
flawline-dataset/   generator/ (flaw factory, ingest, QA)  takes/ (baseline audio + alignment)  variants/ (clips + labels)  manifest.csv
engine/             audio, conditions, dtw, compare, detect, free (general mode), consensus, arbitrate, score, explain, calibrate*, model.json, norms.json
eval/               metrics, acceptance tests, leakage audit, headline metrics, harness self-test
schema/             label.schema.json (v1.1.0) + validator
dashboard/          Streamlit: Analyse (default), Baselines, Alterations review, Pilot gate, Dataset
spec-v1.1.md        the build spec
```

Dashboard pages: **Analyse** (score, per-category bars, timeline with the expected range, flaw cards with audio, JSON/HTML export, demo clips), **Alterations review** (original vs altered with a synced spectrogram player), **Pilot gate** (by-ear review sheet), **Dataset** (counts and all measured results).

## Licences

Code: see repository licence file. Data: per subset in `flawline-dataset/LICENSES.md` (VCTK CC BY 4.0, JFK public domain).
