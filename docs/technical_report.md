# Flawline: measuring spoken delivery against controlled flaws

Track C, contrastive speech analytics. Draft of 7 October 2026. Numbers come from `results/` in the repository and are reproduced by `make eval`; the held-out test split has not been run yet and is marked as such.

## 1. Problem

A speaker practising a talk wants to know *where* delivery goes wrong and *why*, not just a single score. Judging delivery needs a yardstick. Human ratings are expensive and inconsistent, and "good" differs between speakers, texts and rooms. Flawline builds the yardstick the other way round: start from clean recordings of real speakers reading a known text, inject one flaw at a known place and strength, and keep the injection as the label. A detector can then be scored exactly: did it find the flaw, where, what kind, and does its score fall as the flaw gets worse?

The system has four parts that agree through one label format (`schema/label.schema.json`, v1.1.0): a dataset, a generator that writes it, an engine that predicts the same format blind, and an evaluation harness that compares them. The generator and the engine never import each other, so the engine cannot see how the data was made.

## 2. Dataset

**Who.** Ten baselines: eight VCTK 0.92 voices (CC BY 4.0; ages 18–38; American, English, Indian ×2, Irish, Australian, American, South African; four women, four men; studio read speech of the elicitation paragraph and Rainbow Passage) and two 60 s excerpts of President Kennedy's 1962 Rice University address (public domain, open-air recording with crowd noise). We first tried Svarah, a corpus of Indian-English spontaneous speech, and dropped it: the speakers' own fillers and dropped plurals make a recording unusable as a clean yardstick. VCTK is clean but flat, which later matters for emphasis and monotone flaws.

**Where.** Each baseline also appears clean and under six synthetic conditions: reversed multi-talker babble at 20 dB, pink noise at 10 dB, room reverb (RT60 0.6 s), a 300–3400 Hz phone band, 64 kbps MP3 and ±6 dB gain. These test *invariance*: a noisy recording of clean delivery should not look flawed.

**What.** Fifteen flaws in seven categories, five levels each (L1 mild to L5 egregious):

| Category | Flaws |
|---|---|
| Pacing | PACE_FAST, PACE_SLOW |
| Pausing | PAUSE_BAD (silence inside a phrase), PAUSE_LOST (a needed pause removed) |
| Intonation | MONOTONE, UPTALK, EMPH_FLAT |
| Volume | FADE (trailing off), SHOUT |
| Fluency | FILLER, REPEAT (false start), RARE_HESIT (hesitation before a rare word) |
| Clarity | SLUR |
| Text fidelity | WORD_SKIP, WORD_SWAP |

564 clips, 9.97 hours: 444 single-flaw, 50 multi-flaw, 70 clean or condition-only. Splits hold out whole speakers: test = B05, B08 (not used until the final run), dev = B03, train = B01, B02, B04, B06, B07, extra = the two JFK excerpts (never used to fit anything).

**How the flaws are made natural.** The first version sounded wrong: fillers were foreign audio and placement was random. The generator now works from the transcript. Word times come from faster-whisper matched to the known text, then snapped to energy valleys and glottal epochs, so cuts land at quiet points. spaCy types every word boundary (sentence end, clause punctuation, before a conjunction, inside a phrase …) and places pauses and hesitations where speakers actually do, never inside a tight phrase. Edits use the speaker's own audio only: pitch-synchronous time and pitch changes (PSOLA through Praat) for pace, drags and intonation; the speaker's own room tone for gaps; the speaker's own schwa for fillers. Pace edits spend the change on gaps first, then vowels, consonants least, as real rushing does. UPTALK replaces the tail contour from the sentence nucleus instead of adding to the natural fall. Levels are nested: the L2 edit contains the L1 anchor, so level is the only thing that changes. An artifact gate rejects any edit whose splice click exceeds 6 dB over the noise floor more than 5% of the time (never above 12 dB). Everything is seeded; the Praat random generator is seeded too.

**Checks.** Flaw strength grows with level on all eight VCTK speakers for 12 of the 15 flaws; UPTALK on 7 of 8, REPEAT on 4 of 8 (tied levels) and EMPH_FLAT on 2 of 8 do not. Clean-machine reproduction (fresh Linux container, pinned packages, shipped baseline takes): 75 of 80 clips were bit-identical to the macOS build and the other 5 differed by one 16-bit step. Re-aligning the baselines on another machine moves word times by up to 0.37 s, so the released package ships `takes/`.

**Leakage audit.** Can a classifier tell the edited clip from the clean one using only editing artifacts (click energy at the join, noise-floor step, spectral flux spike, exact repetition)? The control is paired: for every edit, the same place in the clean baseline. For flaws that *are* a silence the control is a natural silence edge. Held-out test windows give AUC 0.598 (pass mark 0.60); leave-one-speaker-out over the six train and dev speakers gives 0.616. Four flaws sit near 0.7 (RARE_HESIT, FILLER, PAUSE_BAD, EMPH_FLAT). We changed the audit definition once during the project (v1 compared noise floors across a pause edge, which measured the pause itself); the v1 value was 0.606. We report both and do not claim a clean pass.

## 3. Engine

The engine answers one question per clip: given the audio and the text, which flaws are present, where, how strong, and why. It has three modes that differ only in how "expected delivery" is obtained.

**Reference modes.** Compare the clip with a clean recording of the same text.
1. *Condition matching.* The reference is degraded to the participant's measured bandwidth, noise (taken from the participant's own quiet frames) and reverb, so the recording condition cannot masquerade as delivery.
2. *Alignment.* Dynamic time warping on log-mel features with deltas, normalised per utterance. The lag between participant and reference time is read two ways: sharp jumps are events (a rise is an insertion, a fall a deletion) and the slope over about 1.5 s is tempo.
3. *Detectors.* One per flaw, each a threshold on a speaker-normalised measurement. Pauses and fillers come from lag-curve events; word skips from deletions; word swaps from a per-word path cost that is high only locally; pace from tempo; monotone from F0 range; uptalk from the sentence-final rise; fade, shout and slur from level and 2–7.5 kHz energy. A match meter (the DTW path cost) scales the sensitive thresholds, so a worse match loosens sensitivity instead of creating false flags.
4. *Same speaker* uses the speaker's own clean take (an upper bound). *Another speaker* compares with several clean voices that read the same sentences; the sentences are stitched from other speakers in the participant's order, and a detection is kept only if at least two references agree.

**General mode (reference-free).** No reference recording. Expectations come from (a) the clip's own statistics (level, F0 range, a word-duration residual window), (b) transcript rules (pause norms by boundary type, ASR against the text for skipped and misread words, sentence-final rise) and (c) population norms (spread of each measurement across the clean train speakers, and a duration model fitted on them). Recognition is faster-whisper base.en, run twice (plain and disfluency-prompted) and cached.

**Shared stages.** Cross-detector arbitration drops known secondary symptoms inside a primary region (pauses inside a rushed stretch, prominence inside a shouted one) and resolves overlaps in favour of the more reliable detector (reliability = precision on train). Severity is a monotone (isotonic) map from the measured deviation to the injected level, fitted on train, so a bigger deviation can never score as milder. Each region carries an explanation filled from the measurements ("12.7 dB louder than your own typical level") and a coaching tip; there is no free text generation.

## 4. Scoring

Per region the penalty is `p = w · c · r · s²·m`: flaw weight `w` (1), recording-quality confidence `c`, detector reliability `r`, severity `s` in [0, 5], and `m` = 1 for events or region seconds/2 capped at 3 for spans. Per category `S = 100·exp(−P/35)` where `P` is the penalty per clip minute; the overall score is 0.6 × the genre-weighted mean (interpretive reading, declamation, extemporaneous, persuasive oratory weights in `engine/rubric.yaml`) plus 0.4 × the worst counted area, so one badly hurt area cannot hide behind an average; the band is also capped by the worst area (below 50 rules out polished and strong, below 30 means needs work). Users may supply their own rubric or turn fluency scoring off. Every point lost traces to a timestamped region with a measured cause.

We tried fitting a weight per flaw so that an L5 flaw lands near 55 and an L3 near 77. (The worst-area blend later brought a level-5 flaw to roughly 55-85 overall without any per-flaw weights.) It made dose-response worse (eight flaws failing) because noisy detectors were scaled up, so weights stay at 1 and tau/exponent were tuned instead (tau 35, exponent 2). The L5 target is not reached: the detectors' severity saturates at L4–L5, which scoring cannot repair.

## 5. Results

Event F1 at IoU ≥ 0.5, thresholds fitted on train only.

| Mode | Train | Dev | JFK 1962 | Flaws counted |
|---|---|---|---|---|
| Same speaker | 0.59 | 0.71 | 0.59 | 11 |
| General (no reference) | 0.31 | 0.20 | 0.07 | 8 |
| Another speaker | 0.23 | 0.22 | n/a | 8 |

*Test split (B05, B08): not run yet; it will be run once, after the feature freeze.*

Per flaw, same speaker, train: PACE_SLOW .87, PAUSE_LOST .86, PACE_FAST .84, WORD_SWAP .64, RARE_HESIT .65, FILLER .58, MONOTONE .57, SHOUT .57, WORD_SKIP .59, PAUSE_BAD .54, FADE .51, SLUR .34, UPTALK .25, REPEAT .13, EMPH_FLAT .09. Headline flaws are the eleven not marked experimental. EMPH_FLAT, REPEAT, UPTALK and RARE_HESIT are experimental and hidden by default: VCTK speakers are too flat for emphasis flattening to scale with level, and the others fail on detection or dose-response.

Acceptance tests (same speaker): score falls with level for 12 of 13 scored flaws (RARE_HESIT fails); under noise, phone, room and codec changes the engine raises 0.35 false flags per minute (target ≤ 0.5); the clean-speech score shift under those conditions averages 3.5-3.7 points for babble and reverb and at worst 29.8 (target < 3, **fails**; the worst-area blend makes a single false flag cost more); a flaw costs 3.7 points in categories other than its own (target < 5 met, < 2 not).

What the numbers say. With the speaker's own clean reading as the yardstick, delivery flaws are found and located reasonably well. Without it the same measurements are confounded by natural variation between speakers and texts, and results drop to 0.2–0.3. In general mode, SHOUT (0.74), PAUSE_BAD (0.36), FILLER (0.31) and WORD_SWAP (0.36) work; SLUR and MONOTONE cannot be detected without a reference (a slur changes consonant energy by 0.9 dB against 3.6 dB of natural spread). Pauses that were removed at commas cannot be seen, because half of natural readers do not pause there at all. On the JFK recording general mode fails (precision 0.04): recognition errors on noisy oratory and norms taken from studio speakers create false flags. A general mode needs far more varied clean speakers and a better recogniser before it can carry a headline.

## 6. Limitations and ethics

- **Nobody has listened.** Whether the injected flaws sound natural has not been rated by humans; the by-ear pilot gate and any human panel are still to do. Objective checks (dose-response, artifact gate, leakage audit) are not a substitute.
- **Few and narrow speakers.** Eight studio read-aloud voices aged 18–38 plus one 1962 voice. No spontaneous speech, slang, children or speakers over 45; one or two speakers per accent, so nothing here supports claims about accents or demographics.
- **Leakage.** The audit is borderline and four flaws are detectable from editing traces. The audit's definition changed once; both values are reported.
- **Held-out data.** Thresholds, classifiers and severity maps were fitted on five speakers. Dev is one speaker (20 clips for event metrics), so dev numbers are noisy. The test split has not been touched.
- **Scoring is a rubric, not a verdict.** It measures departure from a chosen yardstick. A flat voice, an accent, a stammer or a deliberate dramatic pause is not a fault, and the system should not be used to rate or rank people. The fluency switch and custom rubric exist for that reason; recordings flagged as low quality score nothing.
- **Voices.** The JFK clips are public-domain historical recordings, altered mechanically and labelled as altered; the VCTK corpus is released by its authors under CC BY 4.0 and cited in `LICENSES.md`.
- **Packaging.** Bit-exact rebuilds need the shipped baseline takes; on another platform a handful of time-stretched clips differ by one 16-bit step.

## Appendix: reproducing

`make setup`, `make baselines dataset` (or unzip the released dataset), `make eval`, `make app`. Calibration: `python -m engine.calibrate --mode same|cross`, `python -m engine.calibrate_free`. History and every failed attempt: `PROGRESS.md`.
