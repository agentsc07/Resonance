# Track C Game Plan — Contrastive Speech Analytics

Oct 4, 2026 · @Sai Charan

## Goal and scoring

We win Track C by building the most rigorous dataset in the field and a grounding engine that proves, with numbers, that it finds flaws where they really are. Submission closes 15 Oct, 00:15 IST, so real working time runs 4 to 14 Oct.

| Criterion | Weight | What wins it |
| --- | --- | --- |
| Data engineering & stress testing | 30% | Clever baseline sourcing, clean temporal labels, a robust bad-mirror gradient |
| Causal explainability & temporal grounding | 25% | Accurate flaw timestamps, a mathematical rationale tied to context |
| Feature extraction | 20% | Rigorous acoustic processing, stress points, energy contours |
| Visualization & dashboard | 15% | Clear time-series overlay and causal explanations |
| Reproducibility & code quality | 10% | Same output every run, clean deploy instructions |

The app is only 15%. Our effort should follow the weights: roughly half on the dataset and evaluation, a third on features and grounding, the rest on the dashboard and packaging.

## What we detect, and what we never penalise

We score only the delivery choices a speaker controls. Who the speaker is and how the audio was recorded get measured as well, but they never cost points. The brief backs this up: it asks for delivery flaws (tone, cadence, volume, pauses) and for normalisation of "natural biological differences between speakers".

| Bucket | Examples | Treatment | How the system tells it apart |
| --- | --- | --- | --- |
| Delivery choices | Rushing, dead air, monotone, trailing off, fillers | Detected, timestamped, scored | A local deviation from the champion's delivery of the same words, after normalisation |
| Speaker identity | Voice pitch and timbre, accent, gender, age, a stammer or other speech condition | Normalised away, never scored | Stable across the whole clip; semitone and z-score normalisation against the speaker's own median removes it |
| Recording conditions | Background noise, reverb, phone mic, clipping, low input level | Shown as an audio-quality report and lowers confidence, never scored | Present across the whole clip, including in the pauses, not only while speaking |

These three buckets are the dataset's What, Who and Where axes. Rule of thumb: delivery flaws are local and tied to particular words. Identity and recording traits are global and present everywhere. The INV set exists to prove the system respects this line.

### Flaw categories: the "cause" a judge reads

Every flagged region gets one category, the feature that deviated, and the numbers behind it. That is the causal explanation the brief asks for.

| Category | Flaw codes | Measured signature | Example explanation (illustrative) |
| --- | --- | --- | --- |
| Pacing | PACE\_FAST, PACE\_SLOW | Syllables per second vs the champion on the same words | "Words 58–69: 6.1 syll/s vs baseline 4.5 (+35%), rushed" |
| Pausing | PAUSE\_BAD, PAUSE\_LOST | Pause length plus where it falls in the sentence | "1.0 s pause inside the phrase 'the … people'; champion paused 0 s" |
| Intonation | MONOTONE, UPTALK, EMPH\_FLAT | F0 range in semitones, final-word pitch slope, per-word prominence | "Pitch range 0.8 st vs 4.2 st in baseline (z = −3.1): monotone" |
| Volume and energy | FADE, SHOUT | RMS energy in dB relative to the speaker's median | "Last 4 words 10 dB below speaker median: trailing off" |
| Fluency | FILLER, REPEAT, RARE\_HESIT | Spoken tokens absent from the transcript; repeated word spans; pause or filler right before a rare word | "'um' at 31.2 s and 'uh' at 33.8 s, not in the transcript" |
| Clarity | SLUR | Local drop in consonant energy and spectral tilt vs the rest of the same clip | "Words 40–46: 2–8 kHz energy 9 dB below this clip's average" |
| Text fidelity | WORD\_SKIP, WORD\_SWAP | Recognised words aligned against the transcript | "'nation' read as 'notion' at 44.1 s" |

### Grey areas: decisions

- **Accent and "wrong" sounds** (v for w, d for th): not penalised. A substitution that happens consistently across the speech is accent, which counts as identity. We skip phoneme-level pronunciation scoring because those models are trained mostly on native accents and would mark Indian English as wrong.
- **Misread words**: these are different from accent: a wrong word, a skipped word, an added word. They're detected as text fidelity by aligning recognised words to the transcript. Reported with a low weight, since the brief centres on delivery.
- **Stammer and other speech conditions**: never diagnosed. Repetitions and blocks look acoustically like nervous false starts, so the dashboard gets a "don't score fluency" switch: fluency events stay visible but cost nothing. No stammer-like flaws are injected into the dataset.
- **Noisy audio**: handled in three steps. A quality gate up front measures SNR, clipping, reverb and bandwidth. Regions too noisy to measure are left unflagged rather than guessed at. The INV set proves noise never turns into a delivery flaw. No denoising by default, because denoisers reshape the very energy and pitch we measure.
- **Vocabulary complexity**: not scored, neither rewarded nor penalised. The dataset is same-text, so vocabulary never varies and a vocabulary score would have no evidence behind it. Complex words aren't better delivery, and scoring them would penalise non-native speakers. What we do score is the delivery signal around hard words: hesitation, slowing or stumbling on a rare word (RARE\_HESIT). Slang in place of the scripted word shows up as WORD\_SWAP; in extemporaneous speech it's the speaker's choice and isn't scored.

### Stretch: likely root causes

Flaws that show up together usually share a human cause. The dashboard can name it, labelled "likely" because it's inferred from co-occurring flaws rather than measured. About half a day of rules.

| Flaws seen together | Likely cause | Coaching tip |
| --- | --- | --- |
| FADE at sentence ends + rushing into the next sentence + short pauses | Running out of breath | Breathe at clause boundaries |
| Fast pace + fillers + raised pitch early in the speech | Nerves | Deliberately slow the first 30 s |
| MONOTONE + even rhythm + EMPH\_FLAT | Reading, not speaking | Mark 3 key words per sentence |
| REPEAT + WORD\_SKIP + irregular pauses | Under-rehearsed passage | Rehearse the flagged lines |

## Random-word seed ideation

Twelve words were drawn by a random word generator, not picked by hand. Each was chained by association until it hit a speech-evaluation idea. Ten chains produced something usable; the verdict column says whether we build it.

| Seed | Chain | Idea it produced | Verdict |
| --- | --- | --- | --- |
| paddle | rowing, stroke, back-and-forth, round trip, undo | Counterfactual repair: undo only the flagged feature in that region and show the score recovers. That is a causal test, not a correlation. | Build (core) |
| burial | buried, buried the lede, key word not stressed | Buried-emphasis map: find the words the champion stressed and flag where the participant failed to stress them | Build (core) |
| nothing | silence, pause, meaning of silence | Pause grammar: the same 1.2 s silence is a rhetorical pause between clauses but a flaw inside a phrase. Judge pauses by their position in the sentence. | Build (core) |
| mule | hybrid of two animals, chimera | Chimera clips: splice flawed takes into clean audio at word boundaries, so every flaw has an exact start and end | Build (core) |
| elf | Santa's workshop, toy factory | Flaw factory: a library of parameterised flaw injectors with a severity knob from 1 to 5 | Build (core) |
| dare | challenge, stress test, attack | Invariance suite: transforms that must NOT change the score (pitch shift, gain, noise, codec, different speaker) | Build (core) |
| parliament | debate, many judges, judges disagree | Human panel: 5+ people rate the same clips; we report their agreement and show our scores track the consensus more steadily than any one judge | Build |
| shoe | walking, gait, rhythm, footsteps | Rhythm metrics from phonetics research (nPVI, %V, varcoV) computed from phone-level alignment | Build |
| boulevard | road, route, GPS, "off route" | Delivery GPS: the baseline is a route through feature space; show where the participant's trajectory leaves it | Build (dashboard) |
| spot | spot the difference | Synced A/B player plus a blind "hear the flaw?" game that also collects human labels | Build (dashboard) |
| calcification | hardening, stiffening over time | Drift detector: delivery that degrades across a whole speech, such as energy fading in the final minute | Stretch |
| care | caregiver, speech therapy | Fairness note: a stammer is not a delivery flaw. Document this and exclude it from scoring. | Doc only |

## Chosen differentiators

We pitch four headline ideas. Each targets one judged criterion and all of them fit in 10 days.

1. **Ground truth by construction** (data, 30%). The flaw factory plus chimera splicing gives every flawed clip exact flaw timestamps. Most teams will hand-label; we can measure our accuracy against labels that cannot be wrong.
2. **Causal, not correlational** (grounding, 25%). For each flagged region we apply the reverse of the detected deviation to that region alone, re-score, and show the score recovers. That is an intervention test of the explanation. It doubles as the "hear it fixed" demo.
3. **Context-aware flaws** (features + grounding). Pause grammar and the buried-emphasis map judge acoustics against the text: where in the sentence the pause sits, which word carried the stress in the champion's version.
4. **Proven trustworthy** (reproducibility + stress testing). The invariance suite shows the evaluator is speaker-agnostic and stable. The human panel shows it agrees with people more consistently than people agree with each other.

| Differentiator | Criterion | Effort | Risk | Main tools |
| --- | --- | --- | --- | --- |
| Flaw factory (15 flaw types, 5 levels) | Data 30% | 2 days | Low | librosa, pyworld, Parselmouth PSOLA, numpy |
| Chimera splicing | Data 30% | 0.5 day | Low | forced alignment, crossfades |
| Counterfactual repair | Grounding 25% | 1.5 days | Medium | Parselmouth, time-stretch, re-scoring |
| Pause grammar | Grounding 25% | 0.5 day | Low | spaCy dependency parse + word timings |
| Buried-emphasis map | Features 20% | 1 day | Medium | per-word prominence from F0, energy, duration |
| Rhythm metrics (nPVI, %V) | Features 20% | 0.5 day | Low | phone-level alignment |
| Invariance suite | Data 30%, Repro 10% | 0.5 day | Low | audiomentations-style transforms |
| Human panel (5+ raters, 30 clips) | Data 30% | 1 day, spread out | Medium: depends on recruiting raters | Google Form, Krippendorff's alpha |
| Delivery GPS + A/B player | Dashboard 15% | 2 days | Low | Plotly / wavesurfer.js |
| Drift detector | Grounding | 0.5 day | Low | Stretch goal only |

## System architecture

One pipeline serves both the dashboard and the evaluation, so every number in the report comes from the same code the judges see running.

&#91;embedded content: Flawline pipeline · dataset, engine, evaluation\]

The dataset feeds the engine as input and feeds the evaluation harness as ground truth. All scoring is deterministic signal processing; an LLM, if used at all, only rewords explanations from fixed numbers.

## Evaluation plan

The report leads with five results, all computed on the held-out test speeches. Test speeches never appear in tuning, so the numbers are honest.

| Experiment | Metric | Question it answers |
| --- | --- | --- |
| Grounding accuracy | Event F1 at temporal IoU 0.5; median onset and offset error (ms) | Do we find the flaw where it really is? |
| Dose-response | Spearman correlation between injected severity and score drop, per flaw type | Does the score fall steadily as the flaw worsens, and only on the matching rubric dimension? |
| Counterfactual repair | Share of flagged regions whose score recovers after the targeted repair | Is the stated cause the real cause? |
| Invariance | False-flag rate on transforms that should change nothing | Is the evaluator speaker-agnostic and robust to mic, noise and codec? |
| Human agreement | Krippendorff's alpha among raters; Spearman between our score and the panel mean | Do we agree with people, and more steadily than people agree with each other? |

Reproducibility check: run the full pipeline twice, on a laptop and in Docker, and compare the SHA-256 of the output JSON. Identical hashes go in the README.

Report self-recorded (natural) flaws separately from synthetic ones. Synthetic flaws will score better; saying so openly is part of the rigor judges reward.

## Schedule and deliverables

The dataset is frozen by 8 Oct so everything after it is measured on fixed data. The submission is done by 14 Oct evening, leaving a buffer before the 00:15 cutoff.

1. **Sat 4 Oct**: pick 8 baseline speeches and confirm licences. Trim 60 to 90 s excerpts and get transcripts. Set up the repo and the alignment step.
2. **Sun 5 Oct**: build the flaw factory (first 5 flaw types) and chimera splicing. Send the panel invite to raters.
3. **Mon 6 Oct**: remaining 10 flaw types, the invariance transforms and the generator script. Record self-mirrors (3 takes per speech).
4. **Tue 7 Oct**: label self-recorded takes (auto-proposed, then hand-checked). Double-annotate 20% and measure boundary agreement.
5. **Wed 8 Oct**: freeze dataset v1.0 with splits, manifest, checksums and datasheet. Release the panel form.
6. **Thu 9 Oct**: feature extraction (F0, energy, MFCC, spectral tilt, rate, pauses, prominence, rhythm), speaker normalisation and baseline-vs-participant DTW.
7. **Fri 10 Oct**: flaw detectors, temporal grounding, pause grammar, buried-emphasis map, causal explanation templates.
8. **Sat 11 Oct**: counterfactual repair and the full evaluation run. First results table.
9. **Sun 12 Oct**: dashboard (upload, overlay, flaw lanes, A/B player, Delivery GPS).
10. **Mon 13 Oct**: panel results in. Polish the dashboard. Write the 6-page doc. Run the reproducibility check in Docker.
11. **Tue 14 Oct**: record and upload the 3 to 10 min YouTube video. Finalise README and dataset link. Submit on Devpost and tick the Unstop box.

Deliverables checklist:

- [ ] GitHub repo with code and README
- [ ] Dataset on GitHub or a public Drive link in the README, with datasheet and licences
- [ ] Dashboard: upload, feature extraction, visual and text output
- [ ] Technical doc, max 6 pages: dataset construction, rubric, model, scoring
- [ ] YouTube video showing dataset creation, stress tests, and live flaw catching
- [ ] Devpost submission plus the Unstop confirmation checkbox

## Dataset spec (v1, collection)

Every clip is described on three axes: **Who** is speaking, **Where** it was recorded, and **What** delivery flaws it contains. Only What is scored. Who and Where must never move the score, and the dataset exists to prove both.

**Why build a dataset at all.** The rubric is a claim ("this score means rushing"). The dataset is the evidence: labelled clips where we know exactly which flaw sits where, so we can measure whether the rubric finds it. The brief also makes it a deliverable worth 30% of the marks.

### Collection order: baselines, then a pilot gate, then scale-up

1. Collect all clean takes: 8 champion baselines, then reader takes.
2. Run every What variant on **one pilot clip** and check by ear that it sounds realistic (pilot gate, below).
3. Only after the pilot passes, batch-generate the same variants across all clean takes.

### Who: the speakers

Baselines are 8 champion excerpts, 60 to 90 s each, two per genre: interpretive reading, declamation, extemporaneous, persuasive oratory. They're the yardstick, so they stay clean and effective. They should still vary: at least 2 genders, at least 2 accents including Indian English, different eras.

Readers are 8 everyday people who each read 3 baseline transcripts, so every baseline gets 3 readers. Coverage target:

| Dimension | Target |
| --- | --- |
| Age band | At least one reader each in 18–25, 26–45, 46–65, 65+ |
| Accent | At least 4 Indian regional accents (e.g. Hindi-belt, Tamil, Telugu, Bengali), plus one non-Indian if possible |
| Gender | At least 3 women and 3 men |
| Fluency | 1 reader who stammers, with consent. Fallback: public stuttering data as a labelled supplement, after a licence check |
| Synthetic voices | Not used as baselines. Optional TTS slice for accent gaps, at most 10% of reader takes, reported separately |

Each reader records, per assigned transcript:

- a **best take**: practised, after listening to the champion twice
- one **natural-flaw take** (for one transcript only): deliberately flawed, following a planning sheet that marks where each flaw goes

### Where: recording conditions

| Code | Condition | Real or synthetic | Parameters |
| --- | --- | --- | --- |
| C0 | Studio or archival master | Real: champions | none |
| C1 | Quiet room, phone mic | Real: readers | none |
| C2 | Everyday room (fan, traffic, TV) | Real: readers, one optional extra take | none |
| N20, N10 | Babble or pink noise | Synthetic | 20 dB and 10 dB SNR |
| RVB | Room reverb | Synthetic | RT60 0.6 s |
| PHN | Phone band | Synthetic | 300–3,400 Hz |
| MP3 | Compression | Synthetic | 64 kbps |
| GAIN | Level change | Synthetic | ±6 dB |

Synthetic conditions are applied two ways. On clean takes they must produce no flags; that's the invariance set. On a sample of flawed clips they must leave the same flags in place.

### What: the delivery flaws

Each injected region is 1 to 6 s, starts and ends on word boundaries, and uses a 10 ms crossfade. Each clip gets 1 to 3 regions, placed by a seeded random generator.

Severity levels:

- **L0**: clean
- **L1**: barely audible to a trained listener
- **L2**: noticeable on close listening
- **L3**: a judge would deduct points
- **L4**: clearly poor
- **L5**: egregious

| Code | Category | Flaw | How it is injected | L1 / L2 / L3 / L4 / L5 |
| --- | --- | --- | --- | --- |
| PACE\_FAST | Pacing | Rushing | Time-stretch, pitch preserved | 1.10 / 1.20 / 1.35 / 1.50 / 1.75 x speed |
| PACE\_SLOW | Pacing | Dragging | Time-stretch, pitch preserved | 0.90 / 0.80 / 0.70 / 0.60 / 0.50 x speed |
| PAUSE\_BAD | Pausing | Pause inside a phrase | Insert room tone, never digital silence | 0.3 / 0.6 / 1.0 / 1.6 / 2.5 s |
| PAUSE\_LOST | Pausing | Rhetorical pause removed | Shorten a pause at a clause boundary | 20 / 40 / 60 / 80 / 100% removed |
| MONOTONE | Intonation | Flat pitch | Compress F0 range around the speaker's median (WORLD or PSOLA) | range x 0.8 / 0.6 / 0.4 / 0.2 / 0.0 |
| UPTALK | Intonation | Rising ending on a statement | Pitch ramp on the sentence's last word | +1 / +2 / +4 / +6 / +8 semitones |
| EMPH\_FLAT | Intonation | Buried emphasis | Pull pitch peak and energy of the most stressed words toward the local mean | 25 / 40 / 60 / 80 / 100% |
| FADE | Volume | Trailing off | Gain ramp over the sentence's final words | -3 / -6 / -10 / -15 / -20 dB |
| SHOUT | Volume | Sudden volume jump | Gain step on one phrase | +2 / +4 / +6 / +9 / +12 dB |
| FILLER | Fluency | Fillers (um, uh, like) | Splice from a filler bank, matched to speaker pitch and loudness | 1 / 2 / 3 / 5 / 8 per 30 s |
| REPEAT | Fluency | False start or repetition | Duplicate a word or phrase | 1 word x1 / 1 word x2 / 2 words / phrase / phrase x2 |
| RARE\_HESIT | Fluency | Hesitation on a rare word | Pause, plus a filler from L3 up, before a word with Zipf frequency below 3.5, and slow that word | pause 0.2 / 0.4 / 0.7 / 1.0 / 1.5 s |
| SLUR | Clarity | Slurred articulation, local only | Inside the region only: attenuate 2–8 kHz consonant energy, shorten stop bursts | -2 / -4 / -6 / -9 / -12 dB |
| WORD\_SKIP | Text fidelity | Skipped word | Cut a word at its aligned boundaries, crossfade the gap | 1 / 2 / 3 / 5 / 8 words per clip |
| WORD\_SWAP | Text fidelity | Misread word | Replace a word with one spliced from elsewhere in the same speech | 1 / 2 / 3 / 5 / 8 words per clip |

Never injected, because they're identity rather than delivery: accent changes, stammer-like blocks, voice changes.

### Pilot gate (realism check before scaling)

1. Pick one pilot clip: a clean 60 s champion excerpt.
2. Generate all 75 What variants on it (15 flaws x 5 levels) plus the 6 synthetic Where conditions.
3. One listener (Sai) rates each clip in `pilot/realism.csv`: sounds natural / slight artifact / robotic; level feels too weak / right / too strong; notes.
4. Tune `generator/config.yaml`, regenerate only the failing flaws, and listen again.
5. Pass when each flaw type has at least 80% of clips rated natural or slight artifact, and its 5 levels sound in order. A flaw that can't pass is dropped from v1, not shipped.
6. Freeze the config as v1.0 and batch it across all clean takes.
7. Repeat steps 2–5 once on a reader's phone take before batching readers, since artifacts behave differently on noisy audio.

### Clip counts

| Set | Formula | Clips |
| --- | --- | --- |
| Clean takes | 8 champions + 24 reader best takes | 32 |
| Natural-flaw takes | 1 per reader, hand-labelled | 8 |
| Chimeras | 2 per natural-flaw take: flawed phrases spliced into the same reader's best take | 16 |
| Champion full grid | 8 x (15 flaws x 5 levels + 5 multi-flaw) | 640 |
| Reader grid | 24 x (15 flaws at L3 + 5 multi-flaw) | 480 |
| Invariance (clean + condition) | 32 x 6 synthetic conditions | 192 |
| Flawed + condition | 120 sampled flawed clips x 1 random condition | 120 |
| **Total** |  | **about 1,490 (about 30 h, about 2 GB as FLAC)** |

At that size the audio goes on a public Drive link, with the generator and seeds in the repo so anyone can rebuild every synthetic clip bit for bit.

### Transcripts

- **Reference text** = what should be said. For baselines: an official transcript where one exists, otherwise a Whisper draft corrected by a person. Readers read this exact text.
- **Recognised text** = what was actually said, from verbatim speech recognition. Text fidelity is the difference between the two, so the reference is never taken from transcribing flawed audio.
- **Dashboard uploads**: the transcript is optional. Without it we transcribe the audio and switch off skipped- and misread-word detection, with a visible note.

### Folder layout

```
flawline-dataset/
  README.md  DATASHEET.md  LICENSES.md  manifest.csv  checksums.sha256
  baselines/B01/
    source.json        URL, speaker, date, licence, genre
    reference.txt      the reference text
  takes/               clean and natural-flaw recordings
    B01-CHAMP_C0.flac  B01-CHAMP_C0.align.json
    B01-R03_C1.flac    B01-R03_C1.align.json
    B01-R03_C1_NAT.flac
  variants/            generated clips + one label JSON each
    B01-R03_C1__PACE_FAST_L3_s0412.flac
    B01-R03_C1__PACE_FAST_L3_s0412.json
  readers/
    readers.csv        id, age band, accent, gender, fluency profile (self-reported)
    consent/           signed one-line release per reader
  fillers/             filler bank + manifest
  pilot/realism.csv
  panel/clips.csv  panel/ratings.csv
  generator/config.yaml  generator/make_dataset.py
```

File name: `<take>_<where>[__<what>_<level>_s<seed>]`, where a take is `<baseline>-<speaker>`, e.g. `B01-R03`.

### Label file (one JSON per clip)

```json
{
  "clip_id": "B01-R03_N20__PACE_FAST_L3_s0412",
  "take_id": "B01-R03",
  "baseline_id": "B01",
  "genre": "declamation",
  "who": {"speaker_id": "R03", "age_band": "46-65", "accent": "Tamil English",
          "gender": "F", "fluency_profile": "typical", "synthetic_voice": false},
  "where": {"base_condition": "C1", "added_condition": "N20", "params": {"snr_db": 20}},
  "what": [
    {"flaw": "PACE_FAST", "category": "Pacing", "level": 3, "params": {"rate": 1.35},
     "start_s": 23.61, "end_s": 27.08, "word_start": 58, "word_end": 69,
     "baseline_start_s": 23.61, "baseline_end_s": 28.30}
  ],
  "duration_s": 71.42,
  "seed": 412,
  "generator_version": "1.0.0",
  "labels_from": "generator"
}
```

`what` is empty for clean and invariance clips. `labels_from` is one of generator, splice, human or human-double. Each region keeps both clip time and baseline time, because time-stretching shifts everything after it.

### manifest.csv columns

`clip_id, take_id, baseline_id, genre, speaker_id, age_band, accent, gender, fluency_profile, base_condition, added_condition, flaw_codes, categories, max_level, n_regions, duration_s, split, labels_from, licence, sha256`

### Splits

Hold out 2 baselines and 2 readers entirely. Any clip involving either goes to test; the rest splits into train and dev by baseline. The test set therefore checks unseen speeches and unseen voices.

### Reader recording protocol

- Phone at about 30 cm, recorded in the phone's voice-memo app, original file sent unedited (no WhatsApp: it compresses audio).
- 3 s of room silence at the start, used for the noise estimate.
- Listen to the champion twice, then record the best take; re-record freely.
- Natural-flaw take: follow the planning sheet, which marks the flaw and word span for each region.
- A one-line consent to public release, signed before recording.

### Quality checks

- Automated on every build: no clipping; no digital-zero runs; every region boundary within 20 ms of a word boundary; alignment confidence above threshold; regenerating from seeds reproduces the stored SHA-256.
- By ear: the pilot gate, then a random 10% spot-check of the batch, logged in `qa_log.csv`.
- Natural-flaw labels: 20% double-annotated. Report the median boundary difference between annotators; aim for under 100 ms.

## Risks and open questions

The biggest risk is that synthetic flaws are too easy to detect. The fix is reporting self-recorded results separately and leading the pitch with them where they hold up.

| Risk | Fallback |
| --- | --- |
| Synthetic flaws leave obvious artifacts, so detection is trivially easy | Use high-quality PSOLA and WORLD resynthesis; report SELF and CHI results separately; add an artifact check that the INV set catches |
| Forced alignment fails on noisy archival audio | Choose clean baselines; fall back to WhisperX word timings and drop phone-level rhythm metrics for that file |
| Not enough panel raters in time | Shrink to 20 clips and 4 raters; recruit from class and team group chats on 5 Oct |
| Licence of a famous speech is unclear | Drop it; publish only a link plus our derived features for that file |
| Counterfactual repair sounds unnatural | Report score recovery as the metric and use repaired audio only where it sounds clean |
| Dashboard eats the schedule | Plain Streamlit fallback; the overlay plot is what matters, not polish |

Open questions:

- Team size and who owns what: dataset, signal processing, dashboard.
- Ask on Discord whether self-recorded speakers may differ from the baseline speaker. We assume yes, given the speaker-agnostic requirement.
- Ask on Discord whether a public Drive link is fine for the audio if the dataset exceeds GitHub's size limits.
