# Flawline — Track C dataset + review dashboard

> Current status, known failures and next steps: see [PROGRESS.md](PROGRESS.md).

Spec: `spec-v1.md`. Everything runs from the repo root.

```bash
pip install -r requirements.txt
cd flawline-dataset/generator
python build_baselines.py                                   # 8 placeholder baselines + word alignments
python make_dataset.py --pilot B01-CHAMP --l3 --invariance --flawed-cond 40
python verify_flaws.py B01-CHAMP B05-CHAMP                  # objective dose-response check -> pilot/objective_check.csv
python qa.py                                                # clipping / zero-runs / boundary checks
python make_dataset.py --check-repro B01-CHAMP              # regenerate one clip, compare SHA-256
cd ../.. && streamlit run dashboard/app.py                  # review dashboard
```

## Status (honest)
- **Baselines**: 8 real VCTK 0.92 recordings (CC BY 4.0), studio, read aloud, 60-65 s each, ages 18-38 (US F 32, English M 38, Indian M 22 and F 23,
  Irish F 24, Australian M 26, US M 22, South African F 26). 4 women, 4 men. `generator/audit_baselines.py`: 7 PASS, 1 minor pause (B05).
- **Svarah was dropped as a baseline source**: its recordings contain the speakers' own fillers/dropped plurals (human transcripts omit them),
  so it cannot be a clean yardstick. Code remains in `ingest_corpus.py`; the clean-speech/slang gap is covered by reader recordings
  (`readers/RECORDING_SHEET.md`: slang scripts + protocol).
- **Gaps**: no slang, no speakers over 38, no champion speeches, until readers record.
- **Contract**: `schema/label.schema.json` (v1.1.0, frozen; optional prediction fields `severity/confidence/points_lost/explanation`). The generator
  validates every label; `qa.py` re-validates. `eval/run.py --predictor oracle|null` proves the harness (oracle F1 = 1.0, null = 0, `eval/test_harness.py`).
- **Flaw factory v1.1** (`generator/`): placement from the transcript's grammar (`linguistics.py`); natural-bundle rendering (`render.py`): ramped PACE with
  gap/vowel/consonant proportions, UPTALK anchored at the nucleus (bug fixed), SHOUT = louder + brighter + higher, partial false starts for REPEAT,
  EMPH_FLAT on focal words with three cues, believable WORD_SWAP (similar word / dropped -s). `--pure TAKE` renders the single-cue ablation slice.
  Cross-cutting: pitch-synchronous joins (glottal epochs, SOLA), floor-preserving gain, same anchor at every level (nested), an artifact gate
  (<=5% of joins above 6 dB, none above 12) that re-places until it passes, seeded Praat RNG (bit-identical reruns). Fillers: own-voice schwa only; TTS removed.
- **Measured (8 baselines)**: all flaws scale with level on 8/8 speakers except EMPH_FLAT (2/8) and REPEAT (4/8, tied levels); join clicks above 6 dB: 0% PAUSE_BAD/REPEAT,
  ~5-10% for splice flaws; reproducibility: identical SHA-256. **Leakage audit** (paired, `eval/leakage_audit.py`): overall AUC 0.61 (pass mark 0.60);
  PAUSE_BAD 0.88, UPTALK 0.91, REPEAT 0.76, RARE_HESIT 0.74 still leak, others near chance. Open work.
- Not built yet: readers, natural-flaw takes, chimeras, human panel, the detection engine.

## Engine, scoring and evaluation (spec v1.1)
```bash
python -m engine.calibrate --mode same     # fit thresholds, REPEAT/FILLER classifier, isotonic severity maps on the TRAIN split only
python -m engine.calibrate --mode cross
python eval/run.py --split dev --mode cross --predictor engine       # grounding / timing / category / dose-response / invariance
python eval/acceptance.py --mode cross                                # score dose-response, invariance, locality
python eval/leakage_audit.py                                          # artifact-only classifier (pass mark AUC <= 0.60)
python eval/test_harness.py                                           # oracle F1 = 1.0, null = 0 (proves the harness)
streamlit run dashboard/app.py                                        # Analyse page is the default view
```
- **engine/** (never imports the generator; reads only dataset assets): `audio` (16 kHz ingest, quality gate) -> `conditions` (degrade the clean reference to
  the participant's measured bandwidth / noise / reverb so Where cannot masquerade as delivery) -> `dtw` + `compare` (baseline-vs-participant DTW; the lag
  curve is read as EVENTS = insertions/deletions and a robust TEMPO slope) -> `detect` (one detector per flaw) -> `explain` (templates filled with measured
  numbers) -> `score` (isotonic severity, penalty = w * confidence * s^1.5 * m, S = 100 exp(-P/tau), genre-weighted, `rubric.yaml`).
- **Match-adaptive sensitivity**: the DTW path cost is a "how well do we match the reference" meter; sensitive thresholds scale with it, so noise or a different speaker
  loosen sensitivity instead of creating false flags (condition clips: 7.4 -> ~0.6 false flags per clip).
- Split hygiene: thresholds / classifiers / isotonic maps are fit on `train`, checked on `dev`, and `test` (B05, B08) is touched once at the end.

## Current engine status (6 Oct 2026, honest numbers)
- Same-speaker reference (upper bound): event F1@0.5 = 0.50 on train (235 clips), 0.62 on dev. Strong: PACE, PAUSE_LOST, WORD_SWAP; weak: EMPH_FLAT, REPEAT, UPTALK, SLUR.
- Cross-speaker reference: F1@0.5 = 0.03 (natural speaker differences swamp the detectors). **Experimental**; the Analyse page defaults to same-speaker.
- Acceptance (same mode): dose-response median Spearman -0.94 (fails PAUSE_LOST, EMPH_FLAT, SLUR); invariance 0.54 false flags/min, max shift 16.7 pts (fail vs <=0.5 / <3); locality 26 pts other-category loss (fail vs <2).
- Leakage audit AUC 0.606 (mark 0.60); PAUSE_BAD, UPTALK, REPEAT, RARE_HESIT still leak editing cues.
- Test split (B05, B08) has not been used for tuning.
