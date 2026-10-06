# Flawline: progress and handoff

Status as of 6 Oct 2026. Submission closes 15 Oct 2026, 00:15 IST. Build spec: `spec-v1.1.md`. The narrative version of this file (what was tried and why) is the published progress page; this file is the working handoff.

## Where I stopped

Stopped after the cross-speaker fix and recalibration. Nothing is half-edited. The last things changed, in order:

1. `engine/reference.py`: cross mode now uses `stitched(bid)` (reference ids look like `cross:B03`), built sentence by sentence from one non-held-out speaker. Cached audio goes to `flawline-dataset/.cache/stitched/`.
2. `engine/predict.py`: uses the stitched reference in cross mode and crops trailing sentences nobody else read (only affects B01).
3. `engine/detect.py`: new `silence_runs`, `pauses_abs` (reference-free pause norms) and the threshold keys `abs_pause`, `abs_lost`, `pause_abs`, `lost_frac`.
4. `engine/calibrate.py`: grid entries for those keys. `engine/model.json` holds the latest `same` and `cross` fits.
5. `dashboard/analyse.py`: default reference mode is same-speaker; cross is labelled experimental.

Not committed. `git init` was done, there are no commits, and nothing has been pushed.

## Current numbers (train, 235 clips, event F1 at IoU 0.5)

| | Same speaker | Cross speaker |
|---|---|---|
| Overall | 0.50 (dev 0.62) | 0.06 |
| Best | PACE_SLOW .87, PAUSE_LOST .86, PACE_FAST .84 | MONOTONE .57, PACE_SLOW .42, PAUSE_BAD .31 |
| Worst | EMPH_FLAT .09, REPEAT .13, UPTALK .25 | WORD_SWAP 0, EMPH_FLAT .01, WORD_SKIP .03 |

The test split (B05, B08) has not been used for tuning or evaluation. Run it once, at the end.

## Refreshed numbers, 6 Oct (task 2; caches cleared, code at commit v0.6)

| Check | Same speaker | Cross speaker | Target |
|---|---|---|---|
| Dose-response median Spearman | -0.943 (fail: PAUSE_LOST, EMPH_FLAT, RARE_HESIT, SLUR) | -0.829 (10 flaws fail) | <= -0.9 each |
| Invariance false flags/min | 0.563 | 18.8 | <= 0.5 |
| Invariance max score shift | 16.7 pts | 53.0 pts (N10) | < 3 |
| Locality (other-category loss) | 25.1 pts | 230.6 pts | < 2 (report if 5 is unreachable) |
| Leakage audit AUC | 0.606 (mark 0.60) | n/a | <= 0.60 |

Raw output: `results/acceptance_same.txt`, `results/acceptance_cross.txt`, `results/leakage.txt`. Cross mode is far worse than same mode on every acceptance check, which is why task 7 replaces the stitched reference.

### Task log

- **T3 babble (7 Oct plan, done 6 Oct):** babble is now time-reversed multi-talker speech (`generator/conditions.py`); N10/N20 and flawed+condition clips regenerated. Same-speaker invariance false flags 0.563 -> 0.503 per minute, N20 mean shift 5.15 -> 3.45 pts. Max shift still 16.7 pts.
- **T4 SLUR blindness:** the global bandwidth estimate misread SLUR L4/L5 (a local band drop) as a band-limited recording, so the reference was degraded to match and the flaw vanished. `engine/conditions.py:bandwidth` is now the 75th percentile of 2 s speech windows. Phone clips still match. SLUR dose-response now passes (Spearman -0.943). Dose-response failing flaws: PAUSE_LOST, EMPH_FLAT, RARE_HESIT.

- **T5 locality (arbitration):** new `engine/arbitrate.py` (secondary symptoms dropped inside a primary region, e.g. pauses inside PACE, prominence/clarity inside SHOUT/FADE; remaining overlaps keep the more reliable detector) plus per-flaw reliability (train precision, `engine/model.json`) weighting every penalty. Biggest cause was EMPH_FLAT firing on other flaws' clips. Same-speaker locality 25.1 -> 4.3 pts after this step; invariance max shift 16.7 -> 10.3.
- **T6 score spread:** tau 23/exp 1.5 -> tau 35/exp 2.0 (`engine/rubric.yaml`). Per-flaw weights fitted on TRAIN (`engine/fit_scoring.py`) made dose-response worse (8 flaws failing) and were reverted (`flaw_weight: {}`). Median own-category score on TRAIN single-flaw clips is now roughly L2 90, L3 70, L5 65, so the "L5 ~ 50-60" target is NOT reached: detector severity saturates at L4-L5 (L4 and L5 get near-identical severity), which scoring cannot fix. Dose-response median Spearman -0.986, failing: EMPH_FLAT, RARE_HESIT.
- **Latest same-speaker acceptance (6 Oct):** dose-response -0.986 (2 fail); invariance 0.52 false flags/min, max shift 13.5 pts, mean shift by condition <= 1.9 pts; locality 5.2 pts (target < 5, stretch < 2). Cross not re-run yet.

- **T7 cross-speaker (time-boxed attempt, 6 Oct):** the per-word median/MAD envelope was NOT built: with the held-out speakers excluded, VCTK gives only 1-4 clean speakers per text (they share sentences, not order), too few for per-word medians. Built the version the data supports (`engine/consensus.py`, `reference.panel_ids`): the participant is compared with a panel of single-voice references and a detection is kept only if >= 2 references agree; detectors under F1 0.10 on TRAIN are disabled in cross mode (EMPH_FLAT, PAUSE_LOST, REPEAT, WORD_SKIP, WORD_SWAP); thresholds re-fitted with the vote in the loop. Result: cross F1@0.5 0.061 -> **0.200 train, 0.162 dev** (same speaker: 0.495 train, 0.628 dev). Exit criterion (>= 0.35) NOT met, so cross mode is labelled **experimental** in the dashboard and docs; headline = same-speaker plus invariance. Cross acceptance: invariance 18.8 -> 4.4 false flags/min, max shift 53 -> 9.0 pts; locality 230 -> 14.4 pts. Best cross detectors: PACE_SLOW .69, MONOTONE .57, PACE_FAST .40, UPTALK .36, PAUSE_BAD .35. More clean speakers (Speech Accent Archive, pending licence) would enable the real per-word envelope.

- **T10 leakage and weak flaws (6 Oct):**
  - *Audit v2* (`eval/leakage_audit.py`): v1 compared 0.5 s noise floors, which at a pause edge measures the pause itself. v2 compares room-tone floors only, uses a natural silence edge as the control for the two flaws that ARE a silence (PAUSE_BAD, RARE_HESIT) and centres both windows on the same silence landmark. Held-out test windows: overall AUC **0.598** (v1 0.606; pass mark 0.60, so a bare pass that depends on the v2 definition). Leave-one-speaker-out over the six train+dev speakers (about 5x more windows, no test speakers): overall **0.616**, so I do NOT claim the audit passes.
  - The old UPTALK 0.8-0.9 was small-sample noise (8 test windows); LOSO UPTALK AUC 0.46. Remaining per-flaw leaks (LOSO): RARE_HESIT 0.70, EMPH_FLAT 0.70, FILLER 0.69, PAUSE_BAD 0.68, REPEAT 0.64, WORD_SWAP 0.64; the rest <= 0.62. Generator-side fixes for these were NOT attempted (would mean regenerating the dataset); flagged instead.
  - *Headline vs experimental* (`eval/headline.py`): experimental = EMPH_FLAT (flattening does not scale with level on VCTK; detector F1 0.09), REPEAT (F1 0.13), UPTALK (F1 0.25), RARE_HESIT (non-monotone dose-response, leak 0.70). Headline same-speaker F1@0.5: **0.588 train, 0.705 dev** over 11 flaws (all-flaws 0.489 / 0.628). Cross (8 flaws): 0.230 train, 0.217 dev. PAUSE_BAD and FILLER stay in the headline with a leakage flag (AUC ~0.7).

- **T11 dashboard in a real browser (6 Oct):** the Claude Chrome extension was not connected, so I drove headless Google Chrome with puppeteer-core (script not committed). Checked: the synced A/B player loads and decodes both clips (status "ready", 1 region); clicking a region plays only that span (+-0.3 s context, lane B); clicking elsewhere plays from there; "switch A/B" toggles lanes at the mapped time; no page or console errors. Analyse page: score, category bars, timeline, flaw card, A/B audio, flaws table, JSON/HTML download all render in same and cross mode with no exceptions. Bugs found and fixed: raw HTML pill text in the flaw card (missing `unsafe_allow_html`), the "You" trace was near-black on the dark theme, noisy loudness trace (now smoothed), default demo clip was an experimental flaw scoring 99 (headline flaws now listed first). Not checked: audible playback quality (headless, no ears), mobile layout, Safari/Firefox.
- **T12 Dataset overview:** `dashboard/results_page.py` adds Who/Where/What counts, headline and per-flaw F1 (same vs cross), dose-response and invariance figures, and the leakage tables, read from `results/`.
- **T13 demo presets:** four buttons on the Analyse page (Indian-accent speaker vs another speaker [cross], noisy-room clean speech, very slow reading, a real 1.7 s pause in an unedited reading). Scores 98 / 100 / 86 / 96. JFK preset pending task 8. "Show experimental detectors" checkbox added; EMPH_FLAT, REPEAT, UPTALK, RARE_HESIT are now hidden from predictions by default (`engine/detect.py:EXPERIMENTAL`, `predict(..., experimental=False)`), so acceptance numbers above predate this change and must be re-run.

- **T8 JFK baselines (6 Oct):** two excerpts of the Rice University address (88-154 s: "a college noted for knowledge"; 635-697 s: "the last twenty four hours") ingested as **B09, B10** from Rice University's Internet Archive copy (Public Domain Mark 1.0; JFK Library JFKWHA-127-002 is also public domain; the JFK Library site itself blocks scripted download, so I did not use it). Excerpts picked by me for continuous speech without applause; Sai can swap them (`generator/ingest_jfk.py`). Split `extra` (never used to fit or select anything). 160 flawed + 12 condition clips generated, QA 0 problems; dataset now 564 clips. Baseline audit flags both as FAIL for a clean-read baseline: B09 has ~1 s rhetorical pauses inside phrases (oratory), B10 has 3 possible "uh" (small.en, prompted; NOT verified by ear and the ASR hallucinates fillers in noise, so Sai must listen to B10 before it counts). Engine on the extra split, same speaker, no tuning: headline F1@0.5 **0.594** (all flaws 0.496), in line with train 0.588 / dev 0.705, so detection transfers to a 1962 open-air oratory recording. Cross-speaker mode does not apply (no other speaker reads these texts). A ZeroDivisionError in `compare.py` on degenerate alignments (found on JFK clips) is fixed. JFK preset added to the Analyse page.

- **T7 revised: reference-free "general" mode (6-7 Oct):** built `engine/free.py` (+ `calibrate_free.py`, `norms.json`). Sources of expectation: (a) the clip's own statistics (loudness, F0 range, word-duration residual windows), (b) transcript rules (boundary-type pause norms, ASR-vs-transcript for skips/swaps/fillers/repeats, statement-final rise), (c) population norms (median/MAD, quantiles, a word-duration model) from the five clean TRAIN speakers only. ASR is faster-whisper base.en, two passes, cached on disk. Same Cand objects, arbitration, isotonic severity, reliability and scoring as the other modes; the Analyse page shows expected-range bands instead of a reference line.
  - Result, event F1@0.5: **train 0.254 all flaws / 0.315 headline-8, dev 0.143 / 0.197, JFK (extra, 1962 recording) 0.064 / 0.071**. The plan's bar (>= 0.4) is NOT met, so general mode is not the headline; same-speaker (0.59 train / 0.71 dev headline-11) stays the upper bound and the validated number. The dashboard defaults to general mode as planned, labelled with these numbers.
  - Works: SHOUT 0.74, PAUSE_BAD 0.36, FILLER 0.31 (voiced-stretch classifier, fitted on train, so in-sample), WORD_SWAP 0.36, PACE_SLOW 0.30. Weak: UPTALK 0.20, FADE 0.22, PACE_FAST 0.19, WORD_SKIP 0.12. Switched off: SLUR (region effect 0.9 dB against 3.6 dB natural spread: undetectable without a reference), MONOTONE (effect about 1 sd), PAUSE_LOST (injected losses are at commas, where 50% of natural readers do not pause anyway), REPEAT, RARE_HESIT, EMPH_FLAT.
  - Does not transfer to the JFK recording (precision 0.04): ASR errors on noisy oratory and VCTK-based norms produce many false misreads and pauses. A general mode needs far more clean and varied speakers (Speech Accent Archive, pending licence) and a better recogniser before it can carry a headline.
  - Bugs found while building it: text detectors reported every detection twice (halved precision for WORD_SWAP/WORD_SKIP); `compare.py` division by zero on degenerate alignments.

- **T15 packaging (6-7 Oct):** `requirements.lock` (104 packages, `pip freeze` from a clean `python:3.13-slim` linux/arm64 container; my first lock, taken from this Mac's environment, had a version conflict and Docker refused it), `Dockerfile`, `Makefile` (`setup`, `baselines`, `dataset`, `eval`, `app`), `scripts/fetch_jfk.sh`. **Clean-machine check** (fresh container, nothing but the repo and the pinned packages): (1) re-ingesting VCTK p318 reproduces the audio to within one 16-bit step (max sample diff 3.1e-5) but faster-whisper word timings differ by up to 0.37 s, which moves flaw placement, so **the released takes (audio + alignment JSON) must ship with the dataset** and `make baselines` is for provenance, not bit-exact rebuilds; (2) regenerating the B01 grid (80 clips) from the shipped takes in the Linux container: **75 of 80 clips bit-identical to the macOS build, 5 differ by at most 1 LSB** (4 PACE clips, 1 multi-flaw clip; time-stretch rounding in ffmpeg). Two packaging bugs found by that run: `pyworld` was missing from `requirements.txt`, and the container lacked `setuptools` (`pyworld` imports `pkg_resources`; pinned 80.9.0). The final Dockerfile image itself has not been built end to end with the lock yet (the equivalent environment was); see below if that changes.

- **Latest same-speaker acceptance (7 Oct, after all changes, experimental detectors hidden; n=60 condition clips incl. JFK):** dose-response: 12 of 13 scored flaws pass (RARE_HESIT -0.79 fails; EMPH_FLAT and UPTALK are hidden so not scored); invariance 0.35 false flags/min (<= 0.5 passes), max score shift 13.5 pts (< 3 fails; mean shift by condition <= 1.6); locality 3.65 pts (< 5 met, < 2 not). `results/acceptance_same.txt`.
- **T18 technical report:** `docs/technical_report.md` (problem, dataset, engine, scoring, results, limitations and ethics; about 6 pages; test-split row marked "not run yet").
- **T16/T17 docs:** `DATASHEET.md` written; README rewritten (3-command setup, honest results table). Dataset link and video link are placeholders for Sai. No LICENSE file for the code exists yet: Sai must choose one before the repo goes public.

## Known failing checks

- Acceptance (`eval/acceptance.py --mode same`): dose-response fails for PAUSE_LOST, EMPH_FLAT, SLUR; invariance 0.54 false flags/min (target 0.5) and 16.7 pt max shift (target 3); locality 26 pts (target 2). These were last run before the newest pause changes and before the cross work. Re-run before quoting.
- Leakage audit AUC 0.606 (mark 0.60). PAUSE_BAD, UPTALK, REPEAT, RARE_HESIT leak editing artifacts.
- Cross mode: pace, loudness, voice-quality and word-level detectors fire on natural speaker differences.

## Not done

- Cross-speaker: per-speaker normalisation of pace and level, then re-check every detector.
- Locality and invariance fixes; weak detectors (EMPH_FLAT, REPEAT, UPTALK, SLUR).
- Reference-free mode (`free` currently falls back to same-speaker).
- Spec generator items: closed-loop level calibration, scenario multi-flaw sets, MONOTONE declination fit, SLUR bundle, lexical fillers, EMPH_WRONG, eye-skip, RARE_HESIT partial attempt.
- New corpora (JFK Rice speech, LibriVox Gettysburg, Speech Accent Archive): need your selection.
- Packaging: pinned environment/Docker, `DATASHEET.md`, technical doc, video, dataset release.
- Your tasks: listen to the pilot clips, GitHub repo, Devpost, genre weights, bundles.

## Not verified

- I cannot listen, so no flaw has been judged by ear. Realism ratings go in `flawline-dataset/pilot/realism.csv` (empty).
- The click-to-play spectrogram (`dashboard/player.py`) has not run in a real browser. The Analyse page passed an automated page test only.

## How to run

```bash
export PATH=/Users/saicharan/miniconda3/bin:$PATH        # the system python is 3.14 and lacks the dependencies
streamlit run dashboard/app.py --server.headless true --server.port 8766   # 8765 is taken by another project's app
python -m engine.calibrate --mode same                    # fit on train only, writes engine/model.json
python -m engine.calibrate --mode cross
python -m engine.tune --split dev --mode cross            # check on dev
python eval/acceptance.py --mode same
python eval/leakage_audit.py
```

## Gotchas

- Analyses are cached in `/tmp/engine_cache` and predictions in `/tmp/engine_pred_cache`. Delete the files for a mode after changing how that mode builds its reference. The cross caches were cleared this session; the same-mode caches were not.
- Calibration is coordinate descent per flaw. `abs_pause` is evaluated at the default `pause_abs`, so a bad default hides a better detector. The default is now 0.7.
- `--check-repro` shows a MISMATCH on its first run after a change; run it a second time.
- The generator does not import the engine, and the engine does not import the generator. Keep it that way: it is what makes the evaluation blind.

## Suggested next steps (pick order with Sai)

1. Re-run acceptance and the leakage audit to refresh the numbers.
2. Cross mode: normalise pace and level per speaker, then re-fit.
3. Locality and invariance.
4. Weak detectors.
5. Test split once, then packaging.
