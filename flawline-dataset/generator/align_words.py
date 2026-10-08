"""Word alignment of a recorded baseline against its known text: recogniser word timestamps matched to the text, unmatched words interpolated,
every boundary snapped to the nearest energy valley. Used by ingest_corpus.py."""
from __future__ import annotations

import numpy as np

from common import SR, frame_rms_db, s2n, tokenize

HOP = s2n(0.005)

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
