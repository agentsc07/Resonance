# Flawline v1.1 Build Spec

Oct 6, 2026 · @Sai Charan

## Summary

v1.1 makes the flaws sound like real people, adds the **scoring logic** the brief requires ("rubric-based scores"), and builds the engine, evaluation and dashboard **in parallel with** the dataset. That works because all of them share one contract: the label JSON. The generator writes it, the engine predicts it in the same shape, and the harness compares the two. Freeze that schema first, and every workstream can start on 6 Oct.

This spec is based on a review of the current generator (`flaws.py`, `linguistics.py`, `conditions.py`), the 314-clip manifest and the pilot checks. Main findings:

- Placement is already linguistically smart for pauses, fillers, repeats and skips. The weak spots are **how** several flaws are rendered (pure gain, uniform stretch, exact-copy repeats, random-word swaps) and **where** a few of them land (random windows for pace, shout and emphasis).
- Two flaws are broken against their own dose-response check on all 8 speakers. REPEAT fails because of its level design; EMPH\_FLAT because of how its effect is measured and applied. UPTALK has a real bug: the rise is multiplied onto the speaker's natural final fall, so the two cancel out.
- 32–36% of joins for the splice-based flaws click above 6 dB. A detector could learn "click = flaw". This is the biggest threat to credible results.
- Gain-based flaws also move the background noise floor, which is a second artifact a detector could learn.
- Scoring was missing from the plan, and it is a core deliverable. The Scoring logic section defines it.

## Workstreams

&#91;embedded content: Workstreams to submission · 6–14 Oct\]

The harness and engine start on day one against the frozen label schema and the current 314 clips. They switch to v1.1 data when it freezes on 9 Oct. Scoring calibration needs both the engine and frozen data, so it starts on the 9th.

## Flaw placement and rendering review

Each flaw is judged on two questions. **Where** does it land, and is that where real speakers make this mistake? **How** is it rendered, and does it sound like a person or like an edit? Priority: **P0** = fix before the dataset freeze, **P1** = do if time allows, **P2** = later.

### Pacing

**PACE\_FAST** (P0)

- Now: a random clause-aligned window of 2–4 s, sped up uniformly with ffmpeg atempo.
- Problems:
  - The rate jumps instantly at both edges.
  - Words and pauses are compressed equally, but real rushing shortens pauses first, vowels next and consonants least.
  - Placement ignores where people actually rush.
- Change, where: weight windows toward lists ("six spoons…, five thick slabs…, and…"), long sentences of 15+ words, and the last third of the clip, where time pressure bites.
- Change, how: a ramped tempo profile (rises over the first 30% of the region, holds, eases out at the clause end). Inside the region, gaps shrink by rate^1.6 and voiced nuclei by rate^0.9. Store the measured syllable rate before and after in `evidence`.

**PACE\_SLOW** (P0)

- Now: the same uniform stretch, down to 0.5x. At L4–L5 it sounds like slow-motion tape.
- Change, where: sentence openings, and before hard or rare words, where uncertainty slows people.
- Change, how: stretch vowels (PSOLA on voiced runs) and inter-word gaps; leave consonants near 1.0x. Real dragging lives in vowels and gaps.

### Pausing

**PAUSE\_BAD** (P1: placement is already right)

- Now: own-voice room tone inserted inside tight phrases (`the | people`), weighted to short function words. Good.
- Problem: real speakers who stop after "the" lengthen it ("thee…") and let it decay. A clean cut after a coarticulated /ðə/ sounds spliced.
- Change, how:
  - Lengthen the pre-pause word's final vowel 1.3–1.6x with a short decay.
  - From L4, optionally add an audible in-breath harvested from the speaker's own pauses, to model a breath in the wrong place.

**PAUSE\_LOST** (P0)

- Now: shortens any clause or sentence pause longer than 0.3 s.
- Problems:
  - VCTK pauses are editing artifacts. Utterances were joined with 0.3 s margins, so on B01 the "lost" pause is only 0.08–0.38 s.
  - At L5 the next sentence collides with the previous one's final fall, which sounds like a splice.
- Change, where: remove the pauses the champion chose to make, ranked by length: before a key word, between list items, after a point lands.
- Change, how: keep a 40–60 ms floor gap at L5 (a speaker never truly fuses sentences).
- Prerequisite: rebuild baselines that keep natural inter-utterance pauses (see Dataset sourcing).

### Intonation

**MONOTONE** (P1)

- Now: compresses F0 around the speaker's global median, in the window with the highest F0 IQR. The window choice is smart.
- Problem: compressing toward the global median also shifts the region's pitch level. An excited passage sitting above the median gets pulled down, which adds a second cue.
- Change, how: fit a declination line per phrase and compress only the excursions around it, so the level and slope survive and only liveliness is lost. Extend to 4–8 s at L4–L5, since real monotone is long.

**UPTALK** (P0: a real bug)

- Now: multiplies F0 by a ramp over the last 300 ms.
- Bug: the ramp is multiplied onto the natural final fall. A −4 st fall plus a +4 st ramp comes out flat, not rising. That is why B02 is non-monotonic (L3 measures +1.8 st while L1 measures +4.1 st).
- Change, how: replace the tail contour. Anchor at the nucleus, the last stressed vowel of the sentence, and set F0 = F0(nucleus) × 2^(st·ramp/12), keeping only the original micro-jitter.
- Change, where: prefer mid-paragraph statements and list items over the final sentence. Uptalk shows up while explaining, not at conclusions.
- Measure: F0 of the last 50 ms of voicing minus F0 at the nucleus.

**EMPH\_FLAT** (P0: fails dose-response on all 8)

- Now: a random window. The top 25% of words by acoustic prominence get F0 pulled toward the window mean and energy reduced.
- Problems:
  - The words chosen are whatever is loud, not the words that carry meaning.
  - Duration, the third stress cue, is not touched.
  - The check uses the max F0 peak, which is noisy (octave jumps). Measured loss at L2–L5 is 3.3, 3.3, 3.4, 2.1 st.
- Change, where: choose focus words that are both linguistically focal and prominent in the champion's reading:
  - numbers, negations (not, never), contrast words, superlatives
  - the last content word of each phrase (the nuclear accent)
  - first mentions of a noun
- Change, how: reduce all three cues on those words (F0 excursion, intensity, stressed-vowel lengthening). Measure with the same per-word prominence z-score the engine uses.
- New variant, P1: **EMPH\_WRONG**, emphasis moved to the wrong word. Lower the focal word and raise an adjacent function word ("I said *the* red bags"). It's a classic interpretive-reading coaching note.

### Volume

**FADE** (P1)

- Now: a linear dB ramp over the last \~1.4 s of sentences with 5+ words. Placement is right.
- Change, where: weight by breath budget, meaning seconds since the last pause over 0.25 s. People fade when they run out of air. Always include the final sentence of the clip as a candidate, since weak endings are a classic flaw.
- Change, how: the gain must not lower the room noise (see the noise-floor fix in Cross-cutting).

**SHOUT** (P0)

- Now: a pure gain step on a random 1–3 s window with a soft limiter. It sounds like a volume knob.
- Change, how: real vocal effort also flattens spectral tilt and raises pitch. Add a high-shelf boost (+1.5 dB per 4 dB of gain above 1 kHz) and F0 +0.25 st per dB, and keep the noise floor constant.
- Change, where: start at a phrase onset, or cover a single content word (over-emphasis) at L1–L2.

### Fluency

**FILLER** (P1: the best-placed flaw)

- Now: sentence, comma and clause-opener boundaries, plus "the… um… people". Own-voice vowel stretched with PSOLA. Very good.
- Changes:
  - Build "uh" only from schwa nuclei ("the" before a consonant, "a", reduced "of"). Nuclei from "and" or "to" give "aaa" or "ooo", not "uh".
  - Start fillers near the speaker's low pitch (10th percentile F0), not the previous word's end pitch.
  - Add lexical fillers ("so", "like", "actually", "you know") spliced from the speaker's own instances when the text contains them.
  - Remove the macOS TTS fallback from dataset builds entirely: 29 filler regions currently carry no `filler_source`.

**REPEAT** (P0: fails dose-response on all 8)

- Now: the words after a clause opener are copied exactly, 1–2 times.
- Problems:
  - The levels are not ordered: L3 (2 words × 1) inserts less than L2 (1 word × 2).
  - The candidate set depends on word count, so levels land in different places.
  - Exact copies are sample-identical, which is unnatural and trivially detectable.
- Change, levels: one anchor per take, chosen where a 4-word restart fits, used by every level:
  - L1 "the the"
  - L2 "the the the"
  - L3 a 2-word restart
  - L4 a 4-word restart
  - L5 a 4-word restart twice
- Change, how: the first attempt is cut off at 60–80% of its length, with +0.5–1 st pitch and −1 dB, then the clean restart follows.
- New variant, P1: a content-word false start with repair ("the bl— the red bags"): the content word is truncated at 40%, then the phrase restarts.

**RARE\_HESIT** (P1)

- Now: words longer than 4 letters with Zipf frequency below 3.5 get a pause, an "uh" from L3, a slowed word and a drawl of the previous word.
- Problem: Zipf alone picks proper nouns such as "Stella", which aren't hard words.
- Change, where: exclude proper nouns, and score difficulty as low Zipf + syllable count + spelling irregularity.
- New behaviour at L5: a partial attempt before the word ("spec— spectrum").

### Clarity

**SLUR** (P1)

- Now: attenuates 2–8 kHz in the crispest window, with temporal smoothing. The spec promised shortened stop bursts, but the code doesn't do it.
- Problem: band attenuation alone is a local low-pass filter, close to a muffled mic.
- Change, how: attenuate transients (burst onsets) more than steady fricatives, add −2 dB overall level and a slight 1.05x speed-up. Real mumbling is quieter and faster as well as duller.
- Change, where: weight sentence-final stretches and function-word runs.

### Text fidelity

**WORD\_SKIP** (P0, for clicks)

- Now: deletes function words, never sentence-first. Sensible.
- Problem: 32% of joins click above 6 dB.
- Change, how: pitch-synchronous splicing (see Cross-cutting).
- New variant, P1, at L4–L5: **eye-skip**, the classic reading error. The reader jumps from one occurrence of a word to its next occurrence and drops the phrase between (e.g. "these things … these things").

**WORD\_SWAP** (P0)

- Now: swaps a content word for a random same-part-of-speech word from elsewhere in the speech ("peas" becomes "frog"). That isn't a believable misread.
- Change, which word: choose donors by spelling or sound similarity (edit distance on letters or CMUdict phonemes, same first letter, similar length): "form" → "from", "though" → "through". Add morphological misreads (dropped or added -s, -ed). Fall back to same part of speech only if no similar word exists.
- Change, how: prefer donors from a similar prosodic position, and match pitch slope as well as median pitch.

## Cross-cutting generator fixes

These seven changes matter more than any single flaw, because they decide whether a detector learns delivery or learns our editing.

1. **Clean joins (P0).** Target: under 5% of joins above 6 dB click, from 15–36% now.
   - In voiced audio, splice pitch-synchronously at glottal closure instants (Parselmouth PointProcess) with a 15–25 ms equal-power crossfade.
   - Cut inside silent gaps whenever one is within 40 ms.
   - Make `artifact_check` a gate inside the generator: if a clip fails, re-place with the next candidate instead of shipping it.
2. **Constant noise floor (P0).** FADE, SHOUT and EMPH\_FLAT scale the room noise along with the voice, so the floor dips or jumps inside the region.
   - Fix: y = g·x + (1 − g)·tone, using the clip's own room tone (`ctx.tone`).
   - Verify with the existing `floor_db` column: change under 1 dB.
3. **Same anchor at every level (P0).** Severity must be the only thing that changes between L1 and L5. Pick each flaw's anchor once, using the most demanding level's needs (REPEAT's 4-word restart, enough rare words), then apply every level there.
4. **Closed-loop level calibration (P1).** Fixed parameters give different real effects per speaker: PAUSE\_LOST removes 0.38 s on B01 but 1.42 s on B03. Instead, define each level as a target measured effect relative to the speaker's own variability (e.g. MONOTONE L3 = lose 40% of the speaker's phrase-level F0 IQR). Then generate, measure, and bisect the parameter until it hits the target. This makes the gradient speaker-agnostic by construction.
5. **Natural flaw bundles (P0 decision).** Real flaws come with linked cues: a shout raises pitch and brightens tone, dragging stretches vowels and gaps, mumbling is quieter and faster. Make these linked cues the default rendering, all labelled under one flaw code. Keep a small "pure" slice (one cue only, 1 take × 15 flaws × 5 levels) for ablation.
6. **Scenario multi-flaw sets (P1).** Equal time zones spread flaws evenly, but real speeches cluster them. Replace the 5 fixed sets with root-cause scenarios. Each becomes ground truth for the "likely root cause" layer.
   - nervous start: fast + fillers + raised pitch in the first third
   - running out of breath: fade + lost pauses + rushing toward sentence ends
   - reading, not speaking: monotone + flat emphasis
   - under-rehearsed: repeat + rare-word hesitation + skip
   - fading finish: slow + fade + monotone in the last third
7. **Leakage audit (P0, cheap, impressive).** Train a classifier on artifact-only features: click energy at joins, floor discontinuity, spectral flux at edits, sample-exact repetition (autocorrelation). Report how well it finds flaws. Near chance means the dataset tests delivery, not editing. Few teams will show this.

## Dataset sourcing

The brief asks for baselines from "highly effective public speakers (e.g., prominent politicians, champion debaters, TED speakers)". VCTK's lab readers and LibriVox volunteers don't meet that bar. v1.1 therefore adds a real champion recording, and takes diversity from public corpora where **many different people read the same text**.

### Brief compliance check

| Brief requirement | Now | v1.1 |
| --- | --- | --- |
| Good baseline = highly effective public speakers | Not met: VCTK lab reading | JFK Rice University speech as the champion; sponsor debate clip if permission comes |
| Bad spectrum on the exact same transcripts | Met | Met |
| Labels temporally bounded, paired good/bad | Met | Met |
| Dataset public (GitHub or Drive) | At risk: depends on licences | Only sources that allow sharing modified audio; licence per subset in `LICENSES.md` |
| Speaker-agnostic normalisation | In the engine plan | Built into the engine; tested cross-speaker |
| Quality over quantity | Met | Met |

TED talks are out: their CC BY-NC-ND licence forbids sharing modified versions, and flawed mirrors are modified versions.

### Champion baselines

- **JFK, Rice University, 12 Sep 1962.** The JFK Library [marks this recording public domain](https://www.jfklibrary.org/asset-viewer/archives/jfkwha-127-002) as an official US government work.
  - Take two 60 s excerpts: the "We choose to go to the Moon" peak, and a calmer explanatory passage.
  - The archival audio also brings a real "Where" condition.
  - Readers (team, friends, family) read these exact excerpts, which gives true champion-vs-participant pairs.
- **Sponsor debate clip (Augli / Indian Debating League), only with written permission.** A champion debater is exactly the brief's example. Use it as a showcase baseline and in the video. Ask today, because permission takes time.
- **Gettysburg Address, LibriVox, 15 readers.** No original audio exists, so it can't be a champion. It's a natural quality spectrum instead: rank the readers, and use the best as that text's reference.

### Diversity: many people, same text

| Source | Same text? | Diversity | Licence | Use in v1.1 |
| --- | --- | --- | --- | --- |
| VCTK (have) | Yes: the "Please call Stella" paragraph | 8 speakers, 7 accents, ages 18–38 | CC BY 4.0 | Neutral read set; cross-speaker pairs |
| [Speech Accent Archive](https://accent.gmu.edu/about/) (GMU) | **Yes: the same Stella paragraph as VCTK** | Over 2,000 speakers, hundreds of native languages including many Indian ones, wide age range, varied mics | Non-commercial; mirrors list CC BY-NC-SA, so confirm on the site | **P0.** About 30 speakers chosen for Indian L1s, ages 50+ and gender balance. Pairs directly with our VCTK text. |
| [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) | Yes: the CMU ARCTIC prompts | 24 speakers, 6 first languages including Hindi | CC BY-NC 4.0 | P1. 150 utterances per speaker hand-annotated for mispronunciation: real ground truth for "accent is not penalised" |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (AI4Bharat) | No | Indian accents from many districts | CC BY 4.0 | P1. Natural-disfluency test set (its speakers' own fillers) and reference-free mode |
| [LibriVox Gettysburg](https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/) | Yes | 15 readers; ages, accents, mics | Public domain in the USA | P0. Natural quality spectrum |

The Speech Accent Archive is the best find: it's the same paragraph VCTK speakers read, so every flaw already built for VCTK works on it unchanged. It adds the older speakers, Indian first languages and messy recording conditions VCTK lacks.

**Licence handling.** Keep each source in its own folder with its own licence file. Non-commercial licences fit a non-commercial research dataset, and ShareAlike means our altered versions of those clips carry the same licence. State this in `LICENSES.md`. If unsure, ask the organisers on Discord. This isn't legal advice; when a licence is unclear, leave that source out.

### Later, not in v1.1

- Nehru's "Tryst with Destiny" (an Indian government work, likely past its copyright term; check first)
- Speakers over 65 and a speaker who stammers, recorded in person with consent
- Spontaneous speech for the extemporaneous genre
- Skip PAUSE\_LOST on VCTK joins between utterances (artificial pauses); apply it only at commas inside an utterance

## Scoring logic

Every point lost traces back to a timestamped flaw region with a measured cause. Scores are deterministic, configurable per genre, and calibrated on our own dataset. The calibration is the "trained model" part, and it stays explainable.

### Step 1: severity of each region (learned)

The engine measures each region's deviation *d* in that flaw's own unit: semitones of F0 range lost, the speed ratio, inserted seconds, dB. A per-flaw **isotonic regression**, fit on the train split, maps *d* to a severity *s* between 0 and 5 against the injected level. Isotonic regression is a monotone curve, so a bigger deviation can never score as less severe. That gives a trained model whose behaviour can still be explained in one sentence.

### Step 2: penalty of each region

```latex
p = w_f \cdot c \cdot s^{1.5} \cdot m
```

- *w\_f* = flaw weight from the rubric file (default 1)
- *c* = confidence from 0 to 1 (audio quality × alignment confidence). Below 0.4 the region is shown but costs nothing.
- *s*^1.5 makes one egregious flaw cost more than several mild ones
- *m* = 1 for event flaws (filler, repeat, rare-word hesitation, bad pause, lost pause, skip, swap, uptalk). For span flaws (pace, monotone, flat emphasis, fade, shout, slur), *m* = region seconds ÷ 2, capped at 3.

### Step 3: dimension and overall scores

Penalties are summed per category and divided by clip minutes, giving *P*. Each category score is then:

```latex
S_{dim} = 100 \cdot e^{-P_{dim} / \tau}, \quad \tau = 23
```

With τ = 23, one L3 flaw per minute costs about 20 points in its category. The overall score is the genre-weighted mean of the 7 category scores:

| Genre | Pacing | Pausing | Intonation | Volume | Fluency | Clarity | Text fidelity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Interpretive reading | 0.15 | 0.20 | 0.25 | 0.10 | 0.10 | 0.10 | 0.10 |
| Declamation | 0.15 | 0.20 | 0.20 | 0.20 | 0.10 | 0.10 | 0.05 |
| Extemporaneous | 0.15 | 0.15 | 0.15 | 0.10 | 0.30 | 0.15 | 0 |
| Persuasive oratory | 0.15 | 0.20 | 0.20 | 0.20 | 0.15 | 0.10 | 0 |

The weights live in `rubric.yaml`; a user can supply their own, which is the brief's "custom evaluative rubric". With the "don't score fluency" switch on, fluency's weight is shared out across the other categories.

Bands:

- 90+ polished
- 75–89 strong, minor issues
- 60–74 noticeable issues
- below 60 needs work

### What the score must prove (acceptance tests)

- **Dose-response:** for every flaw type, the overall score falls monotonically from L0 to L5 (Spearman ≤ −0.9).
- **Invariance:** on clean takes, Where conditions move the score by under 3 points, and a different reference speaker moves it by under 5.
- **Locality:** a flaw only costs points in its own category. Leakage into other categories stays under 2 points.
- **Reproducibility:** the same input gives byte-identical score JSON.
- **Human agreement (if the panel happens):** Spearman with panel mean of 0.6 or higher.

Every score ships with a breakdown table: region, category, severity, points lost, explanation.

## Detection engine

The engine reads an audio file, an optional transcript and an optional reference take. It writes a prediction JSON in **exactly the label schema**, plus scores. It's a new package, `engine/`, separate from the generator, and must not import from `flaws.py`: the engine may not know how the flaws were made.

### Pipeline

1. **Ingest:** convert to 16 kHz mono and loudness-normalise to −23 LUFS.
2. **Quality gate:**
   - SNR from pause frames
   - clipped fraction
   - bandwidth (the frequency below which 95% of energy sits)
   - reverb estimate

   The gate outputs a quality badge and a per-frame confidence.
3. **Two transcripts:**
   - *Reference alignment:* forced-align the reference text. Reuse the generator's aligner, or torchaudio's MMS forced aligner.
   - *Recognised words:* faster-whisper with word timestamps, prompted with "Um, uh, so, like…" so fillers survive.
4. **Speaker normalisation:**
   - F0 in semitones relative to the speaker's own median
   - intensity in dB relative to the speaker's median speech level
   - speaking rate relative to the speaker's own clip mean

   This is what makes comparisons speaker-agnostic.
5. **Features, per 10 ms frame:** F0, intensity, 2–8 kHz energy, spectral tilt, voicing.
6. **Features, per word:**
   - duration
   - syllables per second
   - pause after
   - F0 peak and range
   - mean intensity
   - prominence z-score
   - Zipf frequency
   - boundary type (reuse `linguistics.py`)
7. **Compare:** align participant words to reference words by text (Needleman–Wunsch on word strings, which also exposes skips and swaps). For every word and every 3-word window, compute participant minus reference on the normalised features.
8. **Detect:** one detector per flaw, each a threshold on a feature delta, tuned on train and dev only:
   - pace: windowed syllable rate
   - pause: inserted or missing gap, given boundary type
   - monotone: F0 IQR ratio
   - uptalk: final rise
   - emphasis: focal-word prominence
   - fade and shout: intensity ramp or step
   - filler: recognised tokens absent from the reference
   - repeat: duplicated word n-grams
   - rare-word hesitation: a gap or filler before a low-Zipf word
   - slur: 2–8 kHz drop
   - skip and swap: from the word alignment
9. **Regions:** merge flagged words separated by gaps under 0.3 s, snap edges to word boundaries, drop regions under 0.25 s.
10. **Explain:** one template per flaw, filled with the numbers and the context words:

    > *Words 58–69 "three red bags and we can go": 6.1 syllables/s against 4.5 in the reference (+35%), pauses inside the span shortened from 0.42 s to 0.11 s. Rushed delivery (Pacing, severity 3.2).*

### Three reference modes

All three are reported in the evaluation.

| Mode | Reference | Use |
| --- | --- | --- |
| Same speaker | The clip's own clean take | Upper bound and sanity check only. Too easy to be the headline. |
| Cross-speaker | Another speaker's clean reading of the same text | **Headline result.** The speaker-agnostic test the brief asks for. |
| Reference-free | Norms from all clean baselines (rate range, pause rules by boundary type, F0 IQR range) | Dashboard uploads with no matching champion |

## Evaluation harness

One command, `python eval/run.py --split test --mode cross`, writes `results/*.csv` and the figures for the technical doc. The harness can be built **today, before the engine exists**, by testing it with two fake predictors. An *oracle* that returns the labels must score F1 = 1.0. A *null* predictor that returns nothing must score 0.

| Experiment | Metric | Pass mark |
| --- | --- | --- |
| Grounding | Event F1 at temporal IoU 0.3 and 0.5, per flaw and per category | F1@0.5 ≥ 0.7 on synthetic |
| Timing precision | Median onset and offset error (ms) | ≤ 150 ms |
| Category accuracy | Confusion matrix of predicted vs true category | ≥ 85% on matched regions |
| Dose-response | Spearman of predicted severity vs injected level; score vs level | ≥ 0.85 |
| Invariance | False flags per minute on clean + 6 Where conditions; score shift | ≤ 0.5 per minute; < 3 points |
| Cross-speaker | The same metrics with another speaker as reference | Within 0.1 F1 of same-speaker |
| Natural flaws | F1 on LibriVox and reader takes with hand labels (if collected) | Report honestly, no threshold |
| Leakage audit | AUC of the artifact-only classifier | ≤ 0.6 |
| Reproducibility | SHA-256 of output JSON across 2 runs and Docker | Identical |

Split hygiene: thresholds and isotonic curves are fit on train, tuned on dev, and test is touched once, at the end. Test holds whole speakers (B05, B08) and, from LibriVox, 2 whole readers.

## Dashboard

Keep Streamlit and reuse the existing waveform and script views. Add a new **Analyse** page as the default, and keep the current review pages under a **Dataset** section. The brief requires three things to be visible: audio and transcript upload, a time-series overlay of baseline vs participant, and highlighted flaw regions with causal explanations.

**Analyse page, top to bottom**

1. **Inputs (sidebar):**
   - audio upload (wav, mp3, m4a)
   - transcript, pasted or uploaded (optional)
   - reference: one of the baselines, or "no reference"
   - genre preset, or upload `rubric.yaml`
   - "don't score fluency" switch
   - "Analyse" button

   Also offer a "Try a dataset clip" picker so the demo works without uploading anything.
2. **Quality badge:** SNR, clipping and bandwidth, with a plain warning if confidence is low.
3. **Scorecard:**
   - overall score and band
   - 7 category bars with points lost
   - the reference mode used
4. **Timeline,** the centrepiece. One shared time axis in participant time; the reference is time-warped to it by word alignment so words line up. Stacked lanes:
   1. waveform with flaw regions shaded by category colour
   2. pitch (semitones): participant line over the reference line
   3. loudness (dB): participant over reference
   4. speaking rate (syllables/s)
   5. transcript with flagged words highlighted

   Hover shows the word and values. Clicking a region opens its card.
5. **Flaw card:**
   - category and severity
   - timestamps
   - the causal explanation sentence
   - points lost
   - A/B players ("You" and "Reference"), each cut to the same words
   - one coaching tip
6. **Flaw table:** every region, sortable by time, severity or points lost.
7. **Export:** the score JSON in label format, plus a one-page PDF or HTML report.

**Dataset section (existing pages, light polish)**

- Add a dataset overview: counts by Who, Where and What, plus the dose-response and invariance figures from `results/`. This gives the video a single screen that shows the 30% criterion.
- Keep Alterations review and Pilot gate as they are.

**Demo must-haves for the video:**

- one cross-speaker example (an Indian-accent speaker against a British reference)
- one noisy-room example where nothing is flagged
- one egregious L5 clip
- one LibriVox reader's natural flaw, plus the sponsor debate clip if permission comes through

## Other pending tasks

| Task | Detail | By |
| --- | --- | --- |
| Git + GitHub | `git init`, `.gitignore` (`generator/.cache/`, `variants/`, `__pycache__`, `*.flac`), push a public repo | 6 Oct |
| Schema freeze | `schema/label.schema.json` + one example; generator, engine and harness all validate against it | 6 Oct |
| Licences | Fix `LICENSES.md` (B05–B08 are VCTK, not Svarah); add the LibriVox entry; drop the TTS fillers from the dataset | 7 Oct |
| Pinned environment | `requirements.lock` from `pip freeze`; `Dockerfile` with ffmpeg; `make dataset`, `make eval`, `make app` | 12 Oct |
| Dataset release | Zip with manifest, checksums and `DATASHEET.md` to a public Drive folder; link in the README | 13 Oct |
| README | Setup in 3 commands, results table, dataset link, video link, licences | 13 Oct |
| Technical doc (6 pages max) | 1 problem and approach · 2 dataset (Who/Where/What, generator, pilot gate, leakage audit) · 3 features and engine · 4 scoring rubric · 5 results · 6 limitations and ethics (accent, stammer, consent) | 13 Oct |
| Video (3–10 min) | 0:00 hook · 0:30 dataset build and a pilot listen · 2:30 stress tests (invariance, cross-speaker) · 4:30 live dashboard catching flaws · 7:00 results and close | 13 Oct |
| Submission | Devpost (repo + video), then tick the Unstop box | 14 Oct, by 20:00 |

## Your tasks (Sai)

The code can be built for you. These are the parts only a person can do: listening, judging, deciding, recruiting, presenting. Must-dos total about 9–11 hours across 9 days.

### Must do

- [ ] **Tue 6:** create the GitHub repo and the Devpost project and team; make a public Google Drive folder for the dataset; email Augli / IDL asking for one debate clip with written permission (30 min)
- [ ] **Tue 6:** decide on natural flaw bundles vs pure single-cue flaws (see Cross-cutting). This changes every flaw. (10 min)
- [ ] **Tue 6:** approve the genre weights in the Scoring table, or adjust them (15 min)
- [ ] **Wed 7:** listen through the 15 LibriVox Gettysburg readings and pick the 2 references and 2 test readers; choose the two 60 s JFK excerpts; approve the \~30 Speech Accent Archive speakers (1 h 15 min)
- [ ] **Wed 7 to Thu 8:** pilot gate by ear on the fixed generator: about 75 clips in the dashboard's Pilot gate page (1.5 h, plus 30 min re-listening after tweaks)
- [ ] **Thu 8:** hand-label natural flaws in 4 LibriVox readings: start, end and category per region. Use the dashboard or Praat. (1.5 h)
- [ ] **Fri 9:** sign off the dataset freeze after spot-checking 10% of the batch (45 min)
- [ ] **Sat 10 to Sun 11:** try the Analyse page on your own voice: read the Gettysburg text twice, once well and once badly. Report anything that feels wrong. (1 h)
- [ ] **Mon 12:** read the results and decide the 4 demo clips for the video (30 min)
- [ ] **Tue 13:** review the 6-page doc; write the hook and story for the video; record the voice-over and screen capture (3 h)
- [ ] **Wed 14:** final check, then submit on Devpost and tick Unstop by 20:00, not midnight (30 min)

### Nice to have, in order of value

- [ ] Human panel (not required by the brief, but the strongest proof that our score matches people): 4–5 friends rate 20 clips on a Google Form, 1–5 per category plus "when did you hear the flaw" (1 h of your time, spread over days)
- [ ] Record 2–3 friends or family of different ages reading the two JFK excerpts on their phones: real champion-vs-participant pairs, extra Who, real Where
- [ ] One reader who stammers, with consent, to show the fluency switch working
- [ ] Ask on the hackathon Discord whether a Drive dataset link is acceptable and whether judges run the code
- [ ] If the sponsor debate clip arrives: add it as a showcase baseline and feature it in the video
