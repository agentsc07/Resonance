"""DEV ONLY: render TTS placeholder baselines from placeholders_tts.yaml (real baselines: ingest_corpus.py) and write their word alignments.

Usage: python build_baselines.py [B01 B02 ...]

Alignment method (placeholder TTS only, `align_source: tts-prefix`): macOS `say` gives no word
timestamps, so we render every word-prefix of the script, take the trimmed duration of each as an
estimate of that word's end, rescale to the full render, then snap each boundary to the nearest
energy valley. Real recordings must be aligned with a forced aligner instead (spec: WhisperX / MFA).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from common import (BASELINES, GEN, SR, TAKES, db, dump_json, frame_rms_db, load_yaml, pink_noise,
                    s2n, tokenize, undb, write_audio)

HOP = s2n(0.005)
COMMA_GAP_S, SENT_GAP_S = 0.45, 0.80       # minimum pause after , ; : and . ? !
ROOM_FLOOR_DB = -62.0


def render(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.wav"
        subprocess.run(["say", "-v", voice, "-o", str(p), "--file-format=WAVE",
                        f"--data-format=LEI16@{SR}", text], check=True)
        x, sr = sf.read(str(p), dtype="float32")
    assert sr == SR
    return x if x.ndim == 1 else x.mean(axis=1)


def active_end(x: np.ndarray) -> int:
    fr = frame_rms_db(x, hop=HOP, win=HOP * 2)
    thr = np.percentile(fr, 95) - 40
    idx = np.where(fr > thr)[0]
    return int(min(len(x), (idx[-1] + 2) * HOP)) if len(idx) else len(x)


_MODEL = None


def _norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha() or c == "'")


def whisper_words(x: np.ndarray, prompt: str):
    global _MODEL
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    if _MODEL is None:
        _MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")
    x16 = resample_poly(x, 320, 441).astype(np.float32)
    segs, _ = _MODEL.transcribe(x16, word_timestamps=True, language="en", beam_size=5,
                                condition_on_previous_text=False)
    return [(_norm(w.word), w.start, w.end) for s in segs for w in s.words if _norm(w.word)]


def align(text: str, full: np.ndarray) -> list[dict]:
    """Reference tokens -> times: whisper word timestamps matched to the known text, then
    unmatched words interpolated and every boundary snapped to the nearest energy valley."""
    from difflib import SequenceMatcher
    toks = tokenize(text)
    n = len(toks)
    hyp = whisper_words(full, text)
    ref = [t["clean"] for t in toks]
    t0 = [None] * n
    t1 = [None] * n
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=ref, b=[h[0] for h in hyp], autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for d in range(i2 - i1):
                t0[i1 + d], t1[i1 + d] = hyp[j1 + d][1], hyp[j1 + d][2]
    matched = sum(v is not None for v in t0)
    # interpolate unmatched runs evenly between the neighbouring matched words
    i = 0
    while i < n:
        if t0[i] is None:
            j = i
            while j < n and t0[j] is None:
                j += 1
            a = t1[i - 1] if i > 0 else 0.0
            b = t0[j] if j < n else len(full) / SR
            step = (b - a) / (j - i)
            for d in range(j - i):
                t0[i + d], t1[i + d] = a + d * step, a + (d + 1) * step
            i = j
        else:
            i += 1
    env = frame_rms_db(full, hop=HOP, win=HOP * 2)
    thr = np.percentile(env, 95) - 32
    words = []
    for k, t in enumerate(toks):
        words.append({**t, "start": s2n(t0[k]), "end": s2n(t1[k])})
    for k in range(n - 1):                      # snap touching boundaries to the energy valley
        e, s_ = words[k]["end"], words[k + 1]["start"]
        if s_ - e < s2n(0.06):
            c = (e + s_) // 2
            lo, hi = max(0, (c - s2n(0.03)) // HOP), min(len(env) - 1, (c + s2n(0.03)) // HOP)
            v = (lo + int(np.argmin(env[lo:hi + 1]))) * HOP
            words[k]["end"] = words[k + 1]["start"] = v
    for w in words:                             # tighten word edges into silence
        while w["start"] + s2n(0.04) < w["end"] and env[min(w["start"] // HOP, len(env) - 1)] < thr:
            w["start"] += HOP
        while w["end"] - s2n(0.04) > w["start"] and env[min((w["end"] - 1) // HOP, len(env) - 1)] < thr:
            w["end"] -= HOP
    print(f"   whisper matched {matched}/{n} reference words")
    return words


def pad_pauses(full: np.ndarray, words: list[dict], rng) -> tuple[np.ndarray, list[dict]]:
    """Insert room tone mid-gap so every clause boundary pauses >= COMMA/SENT_GAP_S."""
    pieces, shift, cursor = [], 0, 0
    out_words = [dict(w) for w in words]
    for k, w in enumerate(words[:-1]):
        target = SENT_GAP_S if w["sent_end"] else COMMA_GAP_S if w["clause_end"] else 0.0
        gap = words[k + 1]["start"] - w["end"]
        need = s2n(target) - gap
        if need > 0:
            mid = w["end"] + max(gap, 0) // 2
            pieces.append(full[cursor:mid])
            pieces.append(np.zeros(need, dtype=np.float32))
            cursor = mid
            shift += need
        out_words[k + 1]["start"] += shift
        out_words[k + 1]["end"] += shift
    pieces.append(full[cursor:])
    return np.concatenate(pieces).astype(np.float32), out_words


def build(b: dict):
    take = f"{b['id']}-CHAMP"
    voice, text = b["voice"], " ".join(b["text"].split())
    full = render(voice, text)
    words = align(text, full)
    # silence outside the speech gets trimmed to 0.5 s lead/lead-out of room tone
    lead = words[0]["start"]
    full = full[lead:]
    for w in words:
        w["start"] -= lead
        w["end"] -= lead
    full = full[: words[-1]["end"] + s2n(0.1)]
    full, words = pad_pauses(full, words, None)
    full = np.concatenate([np.zeros(s2n(0.5), np.float32), full, np.zeros(s2n(0.5), np.float32)])
    for w in words:
        w["start"] += s2n(0.5)
        w["end"] += s2n(0.5)
    full = full / (np.max(np.abs(full)) + 1e-9) * undb(-3.0)
    rng = np.random.default_rng(1000 + int(b["id"][1:]))
    full = full + pink_noise(len(full), rng) * undb(ROOM_FLOOR_DB)     # never digital silence
    write_audio(TAKES / f"{take}_C0.flac", full)
    dur = len(full) / SR
    dump_json(TAKES / f"{take}_C0.align.json", {
        "take_id": take, "sr": SR, "duration_s": round(dur, 4), "align_source": "tts-prefix",
        "words": [{"i": w["i"], "w": w["text"], "clean": w["clean"], "punct": w["punct"],
                   "sent_end": w["sent_end"], "clause_end": w["clause_end"],
                   "start_s": round(w["start"] / SR, 4), "end_s": round(w["end"] / SR, 4)} for w in words]})
    d = BASELINES / b["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "reference.txt").write_text(text + "\n")
    meta = {k: b[k] for k in ("id", "title", "genre", "gender", "age_band", "origin", "accent", "register", "slang_terms")}
    dump_json(d / "source.json", {**meta, "status": "placeholder_tts", "voice": voice, "take_id": take,
                                  "licence": "CC0 (original script, synthetic voice)", "url": None,
                                  "note": "Replace with a real champion recording; see generator/candidates.yaml"})
    print(f"{b['id']} {voice:9s} {len(words):3d} words {dur:5.1f}s")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for b in load_yaml(GEN / 'placeholders_tts.yaml')['baselines']:
        if not want or b["id"] in want:
            build(b)
