# Results

All numbers are read from `results/*.json` and `results/*.csv` by `scripts/make_results.py`. Thresholds and weights are fitted on the **train** split only; **dev** is checked, **JFK** (two excerpts of the 1962 Rice University address) is never tuned on, and the **test** split is run once at the end.

## 1. Event grounding

An injected flaw is *found* when a predicted region of the same flaw type overlaps it. Point-like events (a removed word, a short insertion) are widened to 0.30 s on both sides before overlap is computed. Precision = share of predicted regions that match a true flaw; recall = share of true flaws matched; F1 is their harmonic mean. Counts are over all clips of the split.

Metrics reported side by side:

- **Strict F1**: overlap (IoU) of at least 0.5 between predicted and true region, same flaw type.
- **Standard F1**: the same with IoU of at least 0.3.
- **Onset F1**: same flaw type and predicted start within 250 ms of the true start.
- **Median onset error**: median |predicted start − true start| over matched flaws (IoU ≥ 0.3).
- **Area accuracy**: among regions matched on time alone (IoU ≥ 0.3), the share whose predicted area (pacing, pausing, intonation, volume, fluency, clarity, text fidelity) is correct.

**Headline flaws** (11): FADE, FILLER, MONOTONE, PACE_FAST, PACE_SLOW, PAUSE_BAD, PAUSE_LOST, SHOUT, SLUR, WORD_SKIP, WORD_SWAP. In upload mode only the 5 detectable ones count (FADE, FILLER, PAUSE_BAD, SHOUT, WORD_SWAP).

### With a clean reading of the same text (same-speaker mode)

| Split | Clips | Strict F1 | P | R | Standard F1 | Onset F1 | Median onset error | Area accuracy | Found / true | All 15 flaws F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train (B01, B02, B04, B06, B07) | 235 | 0.611 | 0.609 | 0.613 | 0.670 | 0.580 | 17.5 ms | 98% | 280 / 457 | 0.521 |
| Dev (B03) | 20 | 0.659 | 0.667 | 0.651 | 0.729 | 0.541 | 29.5 ms | 100% | 28 / 43 | 0.586 |
| JFK 1962 (B09, B10; never tuned on) | 160 | 0.551 | 0.569 | 0.533 | 0.607 | 0.528 | 26.1 ms | 96% | 168 / 315 | 0.485 |

Per-flaw strict F1, train split (same):

| Flaw | F1 | Status |
|---|---|---|
| PAUSE_LOST | 0.857 | headline |
| SLUR | 0.711 | headline |
| PACE_FAST | 0.700 | headline |
| FILLER | 0.648 | headline |
| RARE_HESIT | 0.630 | experimental |
| WORD_SWAP | 0.624 | headline |
| SHOUT | 0.582 | headline |
| MONOTONE | 0.571 | headline |
| PAUSE_BAD | 0.567 | headline |
| FADE | 0.516 | headline |
| PACE_SLOW | 0.486 | headline |
| WORD_SKIP | 0.462 | headline |
| UPTALK | 0.224 | experimental |
| EMPH_FLAT | 0.082 | experimental |
| REPEAT | 0.069 | experimental |

### Upload mode (no reference reading)

| Split | Clips | Strict F1 | P | R | Standard F1 | Onset F1 | Median onset error | Area accuracy | Found / true | All 15 flaws F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train (B01, B02, B04, B06, B07) | 235 | 0.363 | 0.341 | 0.388 | 0.380 | 0.325 | 84.5 ms | 89% | 104 / 268 | 0.241 |
| Dev (B03) | 20 | 0.412 | 0.636 | 0.304 | 0.412 | 0.412 | 45.7 ms | 88% | 7 / 23 | 0.206 |
| JFK 1962 (B09, B10; never tuned on) | 160 | 0.116 | 0.074 | 0.261 | 0.125 | 0.107 | 47.0 ms | 78% | 52 / 199 | 0.091 |

Per-flaw strict F1, train split (free):

| Flaw | F1 | Status |
|---|---|---|
| SHOUT | 0.756 | headline |
| WORD_SWAP | 0.402 | headline |
| FILLER | 0.337 | headline |
| PAUSE_BAD | 0.270 | headline |
| FADE | 0.239 | headline |
| UPTALK | 0.197 | experimental |
| EMPH_FLAT | 0.000 | experimental |
| MONOTONE | 0.000 | not detectable without a reference |
| PACE_FAST | 0.000 | not detectable without a reference |
| PACE_SLOW | 0.000 | not detectable without a reference |
| PAUSE_LOST | 0.000 | not detectable without a reference |
| RARE_HESIT | 0.000 | experimental |
| REPEAT | 0.000 | experimental |
| SLUR | 0.000 | not detectable without a reference |
| WORD_SKIP | 0.000 | not detectable without a reference |

### Experimental flaws

Reported, excluded from the headline numbers, and hidden in the app by default:

- **EMPH_FLAT**: VCTK speakers are not expressive enough for a flattening to scale with level; detector F1 0.09
- **REPEAT**: detector F1 0.13; leakage AUC 0.64
- **UPTALK**: detector F1 0.25 (no leakage)
- **RARE_HESIT**: dose-response fails (non-monotone); leakage AUC 0.70

### Other modes

Another-speaker mode (a panel of other voices reading the same text must agree) is experimental, available only in the lab build, and was last evaluated on an earlier version of the dataset; no current numbers are reported.

## 2. Score acceptance tests (same-speaker mode)

**Dose-response.** The B01 baseline has all five levels of every flaw. Spearman correlation between level (L0 = unaltered … L5) and overall score; a flaw passes at ρ ≤ −0.9.

| Flaw | L0 | L1 | L2 | L3 | L4 | L5 | ρ | Result |
|---|---|---|---|---|---|---|---|---|
| PACE_FAST | 100.0 | 95.3 | 95.3 | 76.6 | 87.7 | 62.6 | -0.928 | pass |
| PACE_SLOW | 100.0 | 100.0 | 100.0 | 91.2 | 71.1 | 61.1 | -0.941 | pass |
| PAUSE_BAD | 100.0 | 79.1 | 88.6 | 76.1 | 65.9 | 60.0 | -0.943 | pass |
| PAUSE_LOST | 100.0 | 100.0 | 92.4 | 70.9 | 60.1 | 60.1 | -0.971 | pass |
| MONOTONE | 100.0 | 100.0 | 84.3 | 75.2 | 75.2 | 60.1 | -0.971 | pass |
| UPTALK | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | n/a | not scored (experimental) |
| EMPH_FLAT | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | n/a | not scored (experimental) |
| FADE | 100.0 | 100.0 | 100.0 | 81.8 | 72.8 | 65.3 | -0.941 | pass |
| SHOUT | 100.0 | 100.0 | 100.0 | 75.3 | 59.8 | 55.5 | -0.941 | pass |
| FILLER | 100.0 | 89.5 | 81.6 | 75.2 | 66.7 | 60.1 | -1.000 | pass |
| REPEAT | 100.0 | 95.8 | 93.9 | 94.8 | 93.5 | 60.5 | -0.943 | pass |
| RARE_HESIT | 100.0 | 100.0 | 71.8 | 93.6 | 87.4 | 87.2 | -0.638 | fail |
| SLUR | 100.0 | 100.0 | 94.8 | 76.6 | 67.6 | 60.0 | -0.986 | pass |
| WORD_SKIP | 100.0 | 89.4 | 82.2 | 74.3 | 64.3 | 56.2 | -1.000 | pass |
| WORD_SWAP | 100.0 | 93.5 | 80.4 | 67.8 | 71.7 | 60.1 | -0.943 | pass |

**12 of 13** scored flaws pass. Failing: RARE_HESIT. Per-flaw weights (`engine/rubric.yaml`, fitted on train single-flaw clips) make a single level-5 flaw score about 60 for every flaw type.

**Invariance.** Unaltered readings re-recorded under noise, phone-band filtering, room reverb, MP3 and gain change should keep their score and produce no flags.

- False flags on unaltered clips under those conditions: **0.448 per minute** (target ≤ 0.5: pass), over 60 clips.
- Worst score shift against the unaltered reading: **6.5 points** (target < 3: not met).
- Mean score shift by condition: GAIN 0.0, MP3 0.0, N10 0.19, N20 1.58, PHN 0.19, RVB 0.83 points.

**Locality.** On level-3 single-flaw clips, mean points lost in areas other than the flaw's own: **6.08** (target < 2: not met), over 60 clips.

## 3. Leakage audit

A classifier that sees only editing artifacts (click energy at the join, noise-floor step, spectral-flux spike, sample-exact repetition) tries to tell an altered window from the same place in the clean baseline. AUC near 0.5 means the dataset tests delivery, not editing. Pass mark: AUC ≤ 0.60 overall. Trained on the train split and scored on the held-out test windows; the second column is leave-one-speaker-out over train and dev speakers.

| Scope | Held-out AUC | Leave-one-speaker-out AUC |
|---|---|---|
| ALL | 0.591 | 0.612 |
| EMPH_FLAT | 0.297 | 0.627 |
| FADE | 0.583 | 0.61 |
| FILLER | 0.664 | 0.687 |
| MONOTONE | 0.375 | 0.512 |
| PACE_FAST | 0.536 | 0.537 |
| PACE_SLOW | 0.388 | 0.485 |
| PAUSE_BAD | 0.767 | 0.705 |
| PAUSE_LOST | 0.188 | 0.715 |
| RARE_HESIT | 0.689 | 0.717 |
| REPEAT | 0.458 | 0.742 |
| SHOUT | 0.556 | 0.589 |
| SLUR | 0.438 | 0.516 |
| UPTALK | 0.906 | 0.388 |
| WORD_SKIP | 0.583 | 0.492 |
| WORD_SWAP | 0.723 | 0.637 |

Flaws above 0.65 in leave-one-speaker-out (FILLER, PAUSE_BAD, PAUSE_LOST, RARE_HESIT, REPEAT) can be separated from clean speech partly by their editing artifacts. FILLER and PAUSE_BAD remain headline flaws and are flagged as such.

## 3b. Upload-mode invariance

Unaltered readings under noise, phone band, room reverb, MP3 and gain change, upload mode: **2.240 false flags per minute**, worst score shift **23.5 points**, mean shift by condition GAIN 1.38, MP3 1.41, N10 6.93, N20 5.71, PHN 14.37, RVB 4.6 (n = 60).

