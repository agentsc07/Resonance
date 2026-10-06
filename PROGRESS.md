# Flawline: progress log and handoff

Updated 7 Oct 2026. Submission closes 15 Oct 2026 00:15 IST (aim: submit 14 Oct 20:00; feature freeze Sat 11 Oct EOD; test split run once on 12 Oct). Build spec: `spec-v1.1.md`. Repo state: committed locally (latest commit `eac5da0`, tag `v0.6` marks the first commit); **nothing pushed** (no remote yet; Sai will create the public repo at the end).

Rules kept throughout: generator and engine never import each other; the test split (B05, B08) has not been touched; every number below was measured, and failures are listed as failures. I cannot listen, so no flaw has been judged by ear.

## 1. Where things stand

| Area | State |
|---|---|
| Dataset | 564 clips, 9.97 h, 10 baselines (8 VCTK + 2 JFK), 15 flaws x 5 levels, 6 conditions. QA: 0 problems. |
| Engine | Three modes: General (reference-free, dashboard default), Same speaker (validated headline), Another speaker (experimental). |
| Dashboard | Analyse, Baselines, Alterations review, Pilot gate, Dataset. Tested in real (headless) Chrome. |
| Packaging | `requirements.lock`, `Dockerfile`, `Makefile`, datasheet, README, technical report draft. Final Docker image build was still downloading packages when this was written. |
| Not done | By-ear pilot gate, test-split run (12 Oct), dataset zip + checksums, public repo push, code LICENSE, video, Speech Accent Archive, human panel. |

### Latest numbers (event F1 at IoU 0.5, thresholds fitted on train only)

| Mode | Train (235) | Dev (B03, 20) | JFK extra (160) | Flaws counted |
|---|---|---|---|---|
| Same speaker, headline flaws | 0.588 | 0.705 | 0.594 | 11 |
| Same speaker, all 15 | 0.489 | 0.628 | 0.496 | 15 |
| General (reference-free), headline flaws | 0.315 | 0.197 | 0.071 | 8 |
| General, all flaws | 0.254 | 0.143 | 0.064 | 15 |
| Another speaker (experimental), headline | 0.230 | 0.217 | n/a | 8 |
| Another speaker, all flaws | 0.200 | 0.162 | n/a | 15 |

Headline flaws (same speaker): FADE, FILLER, MONOTONE, PACE_FAST, PACE_SLOW, PAUSE_BAD, PAUSE_LOST, SHOUT, SLUR, WORD_SKIP, WORD_SWAP. Experimental (hidden by default, excluded from headline): EMPH_FLAT, REPEAT, UPTALK, RARE_HESIT. The plan's bar for making general mode the headline (>= 0.4) is **not met**; same speaker stays the headline and general mode is labelled with its numbers in the dashboard.

Same-speaker acceptance tests (7 Oct, all changes in, experimental detectors hidden, 60 condition clips): dose-response passes for 12 of 13 scored flaws (RARE_HESIT -0.79 fails); invariance 0.35 false flags/min (target <= 0.5, **pass**) but worst clean-speech score shift 13.5 pts (target < 3, **fail**, mean shift per condition <= 1.6); locality 3.65 pts (< 5 met, < 2 not met). Leakage audit: 0.598 AUC on held-out test windows (mark 0.60, bare pass, depends on the v2 definition), 0.616 leave-one-speaker-out (not a pass).

Raw outputs: `results/` (`acceptance_*.txt|json`, `headline_*_*.json`, `leakage*.csv|txt`, `dev_*.txt`, `extra_same.txt`).

## 2. The journey, in order

### Phase 0: before the 7-task plan (5-6 Oct)
1. **Start.** Read the spec, planned, built 8 baselines with text-to-speech and a Streamlit dashboard. You asked whether the voices were real: they were TTS, so the project restarted on real recordings.
2. **Real data.** Tried Svarah (Indian-English, gated; access obtained): rejected because its speakers' own fillers and dropped plurals make it unusable as a clean yardstick. Chose VCTK 0.92 (CC BY 4.0), 8 speakers B01-B08 (ages 18-38, 60-65 s each, via the Hugging Face mirror). Caveat you raised: VCTK is flat, so MONOTONE/SLUR/pace effects need evidence-based rendering.
3. **Natural-sounding flaws** (the biggest time sink). Your feedback: fillers were "random pieces of audio", placement was random, "don't be afraid to extend or compress the entire audio". Rebuilt the generator: Whisper word times matched to the known text and snapped to valleys/glottal epochs; spaCy boundary types so pauses and hesitations go where speakers put them; own-voice schwa fillers and drawls; PSOLA stretches (vowels first, consonants least); UPTALK anchored at the nucleus; nested levels; artifact gate (<= 5% of joins above 6 dB, none above 12); seeded Praat for bit-identical reruns. Bugs fixed on the way: Whisper resample error, stale clips after failed placements, Praat randomness, UPTALK cancelling the natural fall, pitch-tracker octave errors, PSOLA edge clicks, false click alarms in near-silence.
4. **Spec v1.1 contract and measurement.** Frozen label schema (1.1.0), evaluation harness (IoU matching, event F1, timing, category confusion, dose-response, false flags per minute; oracle F1 = 1.0 and null = 0 prove the harness), leakage audit, acceptance tests (dose-response, invariance, locality).
5. **Engine.** 16 kHz ingest + quality gate; condition matching (degrade the clean reference to the participant's bandwidth/noise/reverb); DTW with lag-curve events and tempo; 15 threshold detectors; isotonic severity; rubric scoring `S = 100 exp(-P/tau)` with genre weights; explanation templates. Notable fixes: event localisation via DTW local cost (IoU 0.47 -> 0.76, FILLER F1 0.28 -> 0.59); condition-clip false flags 48 -> 10 across 18 clips (condition matching, jitter cancellation, match-adaptive thresholds). Tried and dropped: noise-floor clamp in DTW.
6. **Analyse dashboard page.** Scorecard, timeline, flaw cards with A/B audio, JSON/HTML export.
7. **Cross-speaker mode scored 0.03.** Root cause found: it chose the other speaker with the longest common text *prefix*, which is 14 words, so only the first seconds of each clip were compared. Fix: a reference stitched sentence by sentence from one other speaker (about 100% coverage). Added reference-free pause norms. Result 0.034 -> 0.061. A dashboard port clash (another project's app on 8765) was found and Flawline moved to 8766.

### Phase 1: the 18-task plan (6-7 Oct)

| # | Task | Status | What was done / found |
|---|---|---|---|
| 1 | Commit, tag v0.6, push | Partly | Committed (96 files, no audio) and tagged `v0.6`; later commits through `eac5da0`. **Not pushed: no remote.** A stale `.git/index.lock` blocked git for a day (held by a macOS virtualization process earlier); I committed through a separate index file, then removed the stale lock once nothing held it and re-synced the index (`git read-tree HEAD`). |
| 2 | Refresh numbers | Done | Acceptance same/cross + leakage re-run, recorded. First refresh: same invariance 0.56 flags/min, max shift 16.7, locality 25 pts; cross 18.8 flags/min, shift 53, locality 231. |
| 3 | Babble leaks words | Done | Babble was built from other baselines (same text). Now time-reversed multi-talker speech; N10/N20 and flawed+condition clips regenerated. False flags 0.563 -> 0.503/min; N20 mean shift 5.15 -> 3.45. |
| 4 | SLUR blindness at L4-L5 | Done | The whole-clip bandwidth estimate read the dulled regions as a band-limited recording, so the reference was degraded to match and the flaw vanished. Bandwidth is now the 75th percentile of 2 s speech windows; phone clips still match. SLUR dose-response passes. |
| 5 | Locality (arbitration) | Done | New `engine/arbitrate.py` (secondary symptoms dropped inside a primary region; overlaps keep the more reliable detector) + per-flaw reliability (train precision) weighting penalties. Cause: EMPH_FLAT firing on other flaws' clips. Locality 25.1 -> 4.3 pts; invariance max shift 16.7 -> 10.3. |
| 6 | Score spread | Partly | tau 23/exp 1.5 -> tau 35/exp 2.0. Per-flaw weights fitted on train made dose-response worse (8 flaws failing) and were reverted (`flaw_weight: {}`, tool kept in `engine/fit_scoring.py`). Target "L5 ~ 55" **not reached**: detector severity saturates at L4-L5. |
| 7 | Cross speaker -> envelope -> general mode | See below | The plan was revised twice. |
| 8 | JFK baselines | Done, needs Sai | B09, B10 ingested from Rice University's Internet Archive copy (Public Domain Mark 1.0; the JFK Library site blocks scripted download). I picked the excerpts (88-154 s "a college noted for knowledge"; 635-697 s "the last twenty four hours"); swap via `generator/ingest_jfk.py`. Split `extra`. 160 flawed + 12 condition clips; QA 0 problems. The baseline audit flags both as not-clean: B09 has ~1 s rhetorical pauses inside phrases; B10 has 3 possible "uh" (unconfirmed by ear, ASR hallucinates fillers in noise). Sai must listen to B10. |
| 9 | Speech Accent Archive | Blocked | Needs Sai's licence confirmation. |
| 10 | Weak flaws + leakage | Done, flagged | See section 3. |
| 11 | Dashboard in a real browser | Done | Extension not connected, so headless Chrome via puppeteer-core. Player loads, region click plays only the change, switch A/B works, no page/console errors; Analyse works in all modes. Bugs fixed: raw HTML pill text, near-black "You" trace on dark theme, noisy loudness trace, experimental flaw as default demo, a Streamlit widget-default warning, a missing `+` in a string join. Not checked: audible quality, mobile layout, Safari/Firefox. |
| 12 | Dataset overview page | Done | `dashboard/results_page.py`: Who/Where/What, per-flaw F1 (modes), dose-response, invariance, leakage tables. |
| 13 | Demo presets | Done | Five buttons: accent with no reference, noisy-room clean speech, sudden shouting, a real 1.7 s pause in an unedited reading, JFK 1962 dropped words. |
| 14 | Test split once | Not started | Planned 12 Oct. `eval/headline.py --split test` is ready. |
| 15 | Packaging | Mostly | `requirements.lock` (see section 4), `Dockerfile`, `Makefile`, `scripts/fetch_jfk.sh`. Clean-machine check in section 4. Final image build (with the lock) still running. |
| 16 | Datasheet / licences | Mostly | `DATASHEET.md`, `LICENSES.md` (VCTK, JFK). Dataset zip + checksums not made (wait for final data). |
| 17 | README | Done | Three-command setup, honest results table. Dataset and video links are placeholders. |
| 18 | Technical report | Draft | `docs/technical_report.md` (about 2,100 words; test row marked "not run yet"). |

### Task 7 in detail (the longest thread)
- **7a Envelope plan (8-9 Oct slot):** the per-word median/MAD envelope across clean speakers was not buildable: with held-out speakers excluded, VCTK gives only 1-4 clean speakers per text (they share sentences, not order). Built what the data supports instead: `engine/consensus.py` + `reference.panel_ids` (a panel of single-voice references; a detection survives only if >= 2 agree), detectors under F1 0.10 switched off in cross mode (EMPH_FLAT, PAUSE_LOST, REPEAT, WORD_SKIP, WORD_SWAP), thresholds re-fitted with the vote in the loop. Cross F1 0.061 -> 0.200 train / 0.162 dev; cross acceptance: 18.8 -> 4.4 false flags/min, shift 53 -> 9 pts, locality 231 -> 14 pts. Exit bar (>= 0.35) missed, so cross is **experimental**.
- **7b Revised plan: general (reference-free) mode, new primary:** `engine/free.py`, `calibrate_free.py`, `norms.json`. Expectations come from (a) the clip's own statistics (loudness, F0 range, a word-duration residual window), (b) transcript rules (boundary-type pause norms, ASR vs text for skipped/misread words and fillers, sentence-final rise), (c) population norms (median/MAD, quantiles, a word-duration model) from the five clean TRAIN speakers only. faster-whisper base.en, two passes (plain + disfluency-prompted), cached. Same candidates, arbitration, severity, reliability and scoring as the other modes; dashboard shows expected-range bands.
  - Works: SHOUT 0.74, PAUSE_BAD 0.36, FILLER 0.31 (a voiced-stretch classifier fitted on train, so in-sample), WORD_SWAP 0.36, PACE_SLOW 0.30. Weak: FADE 0.22, UPTALK 0.20, PACE_FAST 0.19, WORD_SKIP 0.12.
  - Switched off: SLUR (region effect 0.9 dB against 3.6 dB natural spread), MONOTONE (effect about 1 sd), PAUSE_LOST (injected losses sit at commas, where half of natural readers do not pause), REPEAT, RARE_HESIT, EMPH_FLAT.
  - Diagnostics that shaped it: word-duration residuals beat plain syllable rate for pace (effect/sd 1.57 vs 0.67); unexplained-voiced-audio only reaches 34% of fillers.
  - Does **not** transfer to the JFK recording (precision 0.04): ASR errors on noisy oratory and studio-speaker norms create false flags.
  - Bugs found: text detectors reported every detection twice (halved WORD_SWAP/WORD_SKIP precision); a ZeroDivisionError in `compare.py` on degenerate alignments (found on JFK clips).

### UI rebuild (7 Oct): Streamlit replaced by a custom web app
Why: you asked for a major UI update, a default build of Analyse / Dataset / About only, lab pages behind `LAB=1`, and "move away from Streamlit and make something fancy". What:
- **Stack:** `app/server.py` (FastAPI, port 8501) + `app/service.py` (turns engine output into the JSON the page draws) + a static front end (`app/static`: vanilla JS modules, canvas timeline, SVG-free charts in HTML/CSS, no CDN scripts; fonts from Google Fonts with fallbacks). Dark glass look taken from your references (black field, three soft colour glows, glass panels, gold accent, a score ring like the gauge references).
- **Analyse** (single column, no sidebar): input strip (drop zone + optional transcript + Analyse) with five demo presets; verdict row (score ring, seven category bars, one-line summary, mode/quality chips); timeline (gold waveform, selectable second lane Pitch / Loudness / Rate with the expected-range band, one strip per flaw, click to select and play, playhead); flaw card (time, plain-language cause, points lost, previous/next, play); highlighted transcript (click a word to select its flaw); one collapsed **Advanced** panel (mode cards, genre, which text was read, don't-score-fluency, experimental toggle, rubric upload, dataset-clip picker, all-flaws table, JSON and HTML export). "Reveal what was injected" appears only on dataset clips. Keys: space, left/right.
- **Dataset:** four tiles, three charts (flaw x level heat grid, conditions, speakers), quality-gate strip (splice-click gate 90% of 494 clips, leakage AUC 0.598 / 0.616, cross-platform repro 75/80, QA), manifest in a collapsed expander with filter and paging. **About:** three-line description, evaluation table pulled live from `results/*.json|csv`, score checks, limits, link buttons (repo, report, datasheet, dataset, video are placeholders until you provide them: `app/links.json`).
- **Engine additions for it:** pasted transcripts and "no transcript" (use what the recogniser hears; text-fidelity checks switch off and the UI says so) in General mode (`engine/transcript.py`).
- **Lab mode:** `LAB=1 python -m app.server` (or `make lab`) starts the old Streamlit pages on port 8766 (now only Alterations review, Baselines, Pilot gate; `dashboard/analyse.py` and `results_page.py` were removed) and adds a "Lab" link in the nav. The lab pages were not redesigned; they keep the Reviewer box and "(dev)" caption.
- **Tested in headless Chrome at 1440x900:** score and timeline fit on the first screen (timeline card ends at y = 828 px); strip click starts playback at the flaw (checked: t = 34.1 s after 1.5 s of play from 32.6 s); upload flow with a real FLAC; General, Same speaker modes; Advanced; Reveal; mobile width 390 px has no horizontal overflow. Screenshots: `docs/screenshots/`. Bugs found and fixed: JSON NaN crashed the About page; audio streamed without Range support ignored seeks (now loaded as a blob); nav overflow on phones; empty band under the strips.
- **Not done / caveats:** the Another-speaker mode needs a text pick for uploads; first-time uploads need about 30 s for speech recognition (the UI says so); the lab pages are unchanged Streamlit; no light theme (dark only by design); the Docker lock must be regenerated to include fastapi/uvicorn (running at the time of writing).

## 3. Leakage, weak flaws, experimental set (task 10)
- **Audit v2** (`eval/leakage_audit.py`): v1 compared 0.5 s noise floors, which at a pause edge measures the pause itself. v2 compares room-tone floors only, uses a natural silence edge as the control for flaws that are a silence (PAUSE_BAD, RARE_HESIT), and centres both windows on the same silence landmark. Held-out test windows 0.598 (v1 0.606); leave-one-speaker-out over train+dev speakers (about 5x more windows, no test speakers) 0.616. The old UPTALK 0.8-0.9 was small-sample noise (8 windows): LOSO 0.46. Remaining per-flaw (LOSO): RARE_HESIT 0.70, EMPH_FLAT 0.70, FILLER 0.69, PAUSE_BAD 0.68, REPEAT 0.64, WORD_SWAP 0.64; the rest <= 0.62. Generator-side fixes were not attempted (would mean regenerating the dataset).
- **Experimental** (`eval/headline.py`): EMPH_FLAT (VCTK is too flat for flattening to scale with level; generator scales on 2/8 speakers; detector F1 0.09), REPEAT (0.13), UPTALK (0.25), RARE_HESIT (non-monotone dose-response). `engine/detect.py:EXPERIMENTAL`; hidden by default (`predict(..., experimental=False)`, dashboard checkbox).
- **Generator facts:** flaw strength grows with level on 8/8 VCTK speakers for 12 of 15 flaws (UPTALK 7/8, REPEAT 4/8 with tied levels, EMPH_FLAT 2/8). Join clicks above 6 dB: about 0% for PAUSE_BAD/REPEAT, 5-10% for splice flaws.

## 4. Packaging and reproducibility (task 15)
- First `requirements.lock` came from this Mac's environment and Docker refused it (a version conflict). Regenerated by `pip freeze` inside a clean `python:3.13-slim` (linux/arm64) container: 104 packages; spaCy model installed separately.
- Clean-machine run, container with only the repo and pinned packages:
  1. Re-ingesting VCTK p318 reproduces the audio to within one 16-bit step (max sample diff 3.1e-5) but faster-whisper word timings differ by up to 0.37 s, which moves flaw placement. **The released takes (audio + alignment JSON) must ship with the dataset**; `make baselines` is for provenance, not bit-exact rebuilds.
  2. Regenerating the B01 grid (80 clips) from the shipped takes: **75 of 80 bit-identical to the macOS build; 5 differ by at most 1 LSB** (4 PACE clips, 1 multi-flaw; time-stretch rounding).
  3. Bugs found: `pyworld` was missing from `requirements.txt`; the container lacked `setuptools` (`pyworld` imports `pkg_resources`; pinned 80.9.0).
- Docker Desktop was not running; I started it. The final `docker build -t flawline:final .` (Dockerfile + lock) was slow on the network and had not finished at the time of writing; the equivalent environment was tested.

## 5. Dead ends (so nobody repeats them)
TTS baselines; Svarah as a baseline; TTS duration-prefix alignment; DTW noise-floor clamp; threshold tuning alone for cross mode (the reference only covered 14 words); a leakage audit with random negatives; per-flaw score weights fitted on train; a first lock file from the Mac environment; whole-clip bandwidth estimation; hf-energy for SLUR and F0-IQR windows for MONOTONE in reference-free mode; plain syllable rate for pace in reference-free mode; unexplained-audio filler detection by threshold alone.

## 6. Known failing checks and limits
- Worst clean-speech score shift under Where conditions 13.5 pts (target < 3); locality 3.65 pts (stretch target < 2); score spread: L5 does not reach ~55.
- RARE_HESIT dose-response non-monotone; four flaws leak at about 0.7 AUC.
- General mode 0.31 / 0.20 / 0.07; cross 0.23 / 0.22: neither reaches the 0.4 bar. Dev is one speaker (20 clips), so dev numbers are noisy; the FILLER classifier is fitted and scored on train.
- Nobody has listened to the injected flaws: realism is unverified (`flawline-dataset/pilot/realism.csv` is empty). Playback quality of the dashboard is also unchecked.
- Eight studio voices aged 18-38 plus one 1962 voice; no slang, spontaneous speech, over-45 speakers.

## 7. Open items
**Needs Sai:** GitHub repo URL (public repo at the end); a LICENSE for the code (none exists); listen to the pilot clips (`http://localhost:8766`) and especially B10; confirm or change the JFK excerpts; Speech Accent Archive licence; dataset and video links for the README; Devpost, genre weights, bundles.
**Mine next:** wait for the Docker build, then record the result; zip the dataset with checksums once data is final; run the test split once on 12 Oct (`eval/headline.py --split test` for same/general/cross) and write `results/test_*.json`; final acceptance + leakage re-run; push with the final tag.
**Cut (not started):** closed-loop level calibration, scenario multi-flaw sets, lexical fillers, EMPH_WRONG, eye-skip, chimeras, LibriVox, human panel.

## 8. How to run
```bash
make setup && make baselines dataset && make app          # see README; dashboard at http://localhost:8501
# in this working copy (miniconda python 3.13; system python is 3.14 and lacks the dependencies):
export PATH=/Users/saicharan/miniconda3/bin:$PATH
streamlit run dashboard/app.py --server.headless true --server.port 8766    # 8765 is taken by another project's app
python -m engine.calibrate --mode same        # fit on TRAIN only -> engine/model.json
python -m engine.calibrate --mode cross
python -m engine.calibrate_free               # general mode
python eval/acceptance.py --mode same
python eval/leakage_audit.py
python eval/headline.py --split dev --mode free     # train|dev|extra (test only once, 12 Oct)
```

## 9. Gotchas
- Caches: `/tmp/engine_cache` (analyses), `/tmp/engine_pred_cache` (predictions), `/tmp/free_cache` and `engine/.cache` (reference-free analyses and ASR). Clear the relevant ones after changing how a mode builds its analysis or after changing the engine's scoring.
- Calibration is coordinate descent per flaw; a bad default can hide a better detector (`pause_abs` default is 0.7 for that reason).
- `--check-repro` shows a MISMATCH on its first run after a change; run it twice.
- `ingest_corpus.py` with no arguments rewrites `baselines.yaml` without B09/B10; run `scripts/fetch_jfk.sh` after it (the Makefile does).
- Keep the generator and the engine from importing each other: it is what makes the evaluation blind.
