<!-- BRAND:start -->
# Flawline

**Detect. Explain. Improve.**
<!-- BRAND:end -->

Contrastive speech analytics (hackathon Track C). Real recorded speakers read a known text; controlled delivery flaws (pace, pauses, intonation, volume, fluency, clarity, text fidelity) are injected at known places and strengths, so every label is ground truth. An engine then finds the flaws blind, says where and why, and scores the delivery.

- **Dataset:** 564 clips, 9.97 h, 10 baseline recordings (8 VCTK voices + 2 excerpts of JFK's 1962 Rice University address), 15 flaws × 5 levels, 6 recording conditions. Datasheet: [`DATASHEET.md`](DATASHEET.md). Licences: [`flawline-dataset/LICENSES.md`](flawline-dataset/LICENSES.md).
- **Download:** _link to the released zip (with `checksums.sha256`) goes here._
- **Video:** _link goes here._
- **Status, known failures, history:** [`PROGRESS.md`](PROGRESS.md).

## Run it (3 commands)

```bash
make setup                  # pinned dependencies (requirements.lock) + spaCy model; needs ffmpeg
make baselines dataset      # download VCTK + the JFK excerpts, align them, generate every clip   (or unzip the released dataset into flawline-dataset/)
make app                    # the web app: http://localhost:8501 (Analyse, Dataset, About)
```

Docker: `docker build -t flawline . && docker run -p 8501:8501 flawline`. The image holds the code and dependencies only; put the released `takes/` (baseline audio + alignment JSON) in `flawline-dataset/takes/` for the engine, because re-aligning on another machine moves word timings by up to 0.37 s. `make eval` re-runs the checks below on the train and dev splits (never the test split).

## What the engine does

Three ways to know what "good" is, chosen in the dashboard:

| Mode | Needs | Use |
|---|---|---|
| **General** (default) | the text read + the audio | expectations from the speaker's own clip, the transcript (pause norms by boundary type, ASR vs text) and population norms from clean speakers |
| **Same speaker** | the speaker's own clean reading | upper bound; the validated headline |
| **Another speaker** (experimental) | other speakers reading the same sentences | a panel of references must agree before a flaw is reported |

Pipeline: 16 kHz ingest and quality gate → (reference modes) degrade the clean reference to the participant's bandwidth/noise/reverb → DTW alignment, lag curve read as insertions/deletions and tempo → one detector per flaw → cross-detector arbitration → isotonic severity → rubric score `S = 100·exp(−P/35)` per category; overall = 0.6 × genre-weighted mean + 0.4 × worst area, and the band is capped by the worst area (an area under 50 rules out polished and strong) → plain-language explanation with the measured numbers. The generator and the engine never import each other.

## Results (honest)

Event F1 at IoU 0.5 against the injected truth. Thresholds are fitted on **train** only; **dev** is B03; **extra** is the two JFK baselines (never tuned on); the **test** split (B05, B08) has not been run yet.

| Mode | Train | Dev | JFK 1962 (extra) | Flaws counted |
|---|---|---|---|---|
| Same speaker (headline) | **0.61** | **0.66** | 0.55 | 11 |
| General | 0.35 | 0.38 | 0.10 | 5 |
| Another speaker (experimental, not re-run after the 7 Oct regeneration) | 0.23 | 0.22 | n/a | 8 |

In plain terms (same speaker, train): it finds about 6 of 10 flaws and about 6 of 10 of its flags are real (precision 0.61, recall 0.61). With no reference: about 4 of 10 found, 3 of 10 flags real.

- **Headline flaws (11):** FADE, FILLER, MONOTONE, PACE_FAST, PACE_SLOW, PAUSE_BAD, PAUSE_LOST, SHOUT, SLUR, WORD_SKIP, WORD_SWAP. FILLER and PAUSE_BAD are leakage-flagged (see below).
- **Experimental, excluded from headline metrics and hidden by default:** EMPH_FLAT (VCTK voices are too flat for the flattening to scale with level; F1 0.09), REPEAT (0.13), UPTALK (0.25), RARE_HESIT (non-monotone dose-response, leakage 0.70).
- **General mode does not reach the 0.4 bar** and does not transfer to the noisy 1962 recording (ASR errors and clean-speaker norms produce false flags). Localised pace changes, slurring, skipped words and monotone are not detectable without a reference, so only 5 flaws count there.
- **Other ways to count the same events** (headline flaws; strict = flag overlaps the real moment by at least half, standard = at least 30 %, onset = right flaw type and start within 250 ms):

| | Strict F1 | Standard F1 | Onset F1 | Typical gap to the real start | Names the right area |
|---|---|---|---|---|---|
| Same speaker, train | 0.61 | 0.67 | 0.58 | 18 ms | 98 % |
| Same speaker, dev | 0.66 | 0.73 | 0.54 | 30 ms | 100 % |
| Same speaker, JFK 1962 | 0.55 | 0.61 | 0.53 | 26 ms | 96 % |
| General, train | 0.35 | 0.36 | 0.33 | 67 ms | 86 % |
| General, dev (6 of 23 found) | 0.38 | 0.38 | 0.38 | 70 ms | 86 % |
| General, JFK 1962 | 0.10 | 0.10 | 0.09 | 66 ms | 74 % |

- **Score acceptance tests (same speaker):** score falls with level for 12 of 13 scored flaws (RARE_HESIT fails, rho -0.64; UPTALK and EMPH_FLAT are experimental and not scored). One weight per flaw now makes a single level-5 flaw land at about 60 overall for every flaw type (56 to 65; it was 56 to 92 before). False flags under noise/phone/room/codec: 0.45 per minute (target <= 0.5, passes). Worst clean-speech score shift under those conditions: 6.5 points (target < 10, met; the older target < 3 is **not** met); mean shift by condition 0.0 to 1.6. A lone flag in a rough recording counts half and cannot be the weakest area. Points lost in other areas on level-3 clips: 6.1 (target < 2, **not met**; it rose from 3.7 when the flaw weights went up).
- **Leakage audit** (classifier sees only editing artifacts): AUC 0.591 on held-out test windows (mark 0.60), 0.612 leave-one-speaker-out, about 0.7 for RARE_HESIT, FILLER, PAUSE_BAD, EMPH_FLAT. Reported as is; the audit definition changed during the project (v2, see `PROGRESS.md`).
- **Not done:** the by-ear realism rating of the injected flaws (one listener checked 8 pairs on 7 Oct; fixes from that listening are in), a speech-accent corpus, speakers over 45 or with slang. Skipped-word detection with no reference reading was tried again and stays off (precision 0.07 on train); the human reference set below is the independent check.

## Human reference set (listening study)

People listen to a few readings, press Space where something slips, name the area, and give an overall score from 1 to 10. The engine (upload mode, never tuned on this set) is then compared with them: how many of the human marks it flagged, how many of its flags a listener also marked, and whether its score ranks the readings the way listeners do. Five of the clips are dataset clips with hidden injected flaws, so the listeners' own hit rate on known flaws is measured too.

```bash
make label                       # http://localhost:8501/label  (each rater opens it on their own machine and enters a name)
python eval/human_agreement.py   # results/human_agreement.json and .txt, with counts next to every percentage
make review                      # http://localhost:8501/review: listen to engine flags no listener marked; Agree / Disagree / Unsure
```
Clips live in `flawline-dataset/human/clips/` (WAV, M4A or FLAC; neutral file names, order shuffled per rater); labels are saved to `flawline-dataset/human/labels_<rater>.json`. To add your own recordings, drop the files in `clips/` and, optionally, their text in `transcripts.json`. Only recordings whose speakers agreed to it are released. Results appear here and on the About page once raters have finished.

## Layout

```
flawline-dataset/   generator/ (flaw factory, ingest, QA)  takes/ (baseline audio + alignment)  variants/ (clips + labels)  manifest.csv
engine/             audio, conditions, dtw, compare, detect, free (general mode), consensus, arbitrate, score, explain, calibrate*, model.json, norms.json
eval/               metrics, acceptance tests, leakage audit, headline metrics, harness self-test
schema/             label.schema.json (v1.1.0) + validator
app/                web app: FastAPI service + static front end (Analyse, Dataset, About); `LAB=1` / `make lab` adds the lab pages
dashboard/          lab pages only (Streamlit): Alterations review, Baselines, Pilot gate
spec-v1.1.md        the build spec
```

Web app pages (three colours only: grey data, gold = what you can act on, coral = points lost): **Analyse** (drop a recording or pick a demo; a compact verdict with bars only for the areas that lost points, a timeline with numbered flagged moments and the expected range, a flaw card with audio, the highlighted transcript, and an Advanced panel with an optional reference reading and the details table), **Dataset** (tiles, what gets injected in plain words, a Hear-a-pair card, conditions, speakers, quality gates, manifest), **About** (three plain-language headline numbers, glossary, full results collapsed). The mode, genre, rubric, experimental-detector and dataset-clip controls appear with `LAB=1`. Lab pages (`LAB=1`): **Alterations review** (original vs altered with a synced spectrogram player), **Baselines**, **Pilot gate** (the by-ear review sheet). Screenshots at 1440x900 are in `docs/screenshots/`.

## Licences

Code: see repository licence file. Data: per subset in `flawline-dataset/LICENSES.md` (VCTK CC BY 4.0, JFK public domain).
