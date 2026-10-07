""""Try that part again": score ONLY the flagged phrase on a retake, using the one feature that caused the flag.
The same measurement is applied to the first attempt (cut from the original recording) and to the retake, against an expected range taken from the
rest of the same reading (windows with no flag). Points for the part: before = what the flag cost; after = before x (how far the retake is still outside
the range / how far the first attempt was), 0 when inside. No engine detector is re-run: this is a direct feature check, so it is quick and local."""
from __future__ import annotations

import numpy as np

from engine import asr, audio as A, features
from engine.align import needleman_wunsch, sim
from engine.render_facts import syllables

# flaw -> (feature key, plain name, unit, direction text)
SPEC = {
    "PACE_FAST": ("rate", "Speaking rate", "syllables per second"),
    "PACE_SLOW": ("rate", "Speaking rate", "syllables per second"),
    "PAUSE_BAD": ("pause_in", "Longest pause inside the phrase", "seconds"),
    "PAUSE_LOST": ("pause_end", "Pause after the last word", "seconds"),
    "FADE": ("fade", "Drop in loudness from start to end", "dB"),
    "SHOUT": ("spread", "Loudest part above the typical level", "dB"),
    "SLUR": ("slur", "Consonant brightness", "dB"),
    "MONOTONE": ("pitch", "Pitch range", "semitones"),
    "FILLER": ("fillers", "Filler sounds heard", "count"),
    "RARE_HESIT": ("fillers", "Filler sounds heard", "count"),
    "REPEAT": ("words", "Words matching the text", "share"),
    "WORD_SKIP": ("words", "Words matching the text", "share"),
    "WORD_SWAP": ("words", "Words matching the text", "share"),
}
CAN = set(SPEC)
FILLER_TOKENS = {"uh", "um", "er", "erm", "ah", "hmm", "mm", "uhh", "umm"}


# ---------------------------------------------------------------- measurements (x: 16 kHz mono)
def _mask(x):
    d = A.frame_db(x)
    m = d > np.percentile(d, 95) - 28
    return d, m, np.where(m)[0]


def _runs(b):
    out, i = [], None
    for k, v in enumerate(list(b) + [False]):
        if v and i is None:
            i = k
        if not v and i is not None:
            out.append((i, k - 1))
            i = None
    return out


def m_rate(x, text, ctx=None):
    d, m, idx = _mask(x)
    if len(idx) < 20:
        return float("nan")
    seg = m[idx[0]: idx[-1] + 1]
    gaps = [b - a + 1 for a, b in _runs(~seg) if b - a + 1 > 30]            # silences over 0.3 s are pauses, not speed
    active = (len(seg) - sum(gaps)) * 0.01
    syl = sum(syllables(w.strip(",.;:!?\"'").lower()) for w in text.split())
    return syl / max(active, 0.2)


def m_pause_in(x, text="", ctx=None):
    d, m, idx = _mask(x)
    if len(idx) < 20:
        return float("nan")
    seg = m[idx[0]: idx[-1] + 1]
    g = [b - a + 1 for a, b in _runs(~seg)]
    return (max(g) * 0.01) if g else 0.0


def m_pause_end(x, text="", ctx=None):
    d, m, idx = _mask(x)
    return float((len(m) - 1 - idx[-1]) * 0.01) if len(idx) else float("nan")


def m_fade(x, text="", ctx=None):
    d, m, idx = _mask(x)
    if len(idx) < 30:
        return float("nan")
    v = d[idx]
    k = len(v) // 3
    return float(np.percentile(v[:k], 80) - np.percentile(v[-k:], 80))      # vowel-level peaks, so consonant dips do not count


def word_level(x, times, r0, r1):
    """Mean level of words r0..r1 above the median level of the other words of the phrase (dB). times: (start_s, end_s) per phrase word, in x's own clock."""
    d = A.frame_db(x)
    lv = [float(np.mean(d[int(s * 100): max(int(s * 100) + 1, int(e * 100))])) if e > s else np.nan for s, e in times]
    inside = [lv[i] for i in range(r0, r1 + 1) if np.isfinite(lv[i])]
    rest = [lv[i] for i in range(len(lv)) if not r0 <= i <= r1 and np.isfinite(lv[i])]
    return float(np.mean(inside) - np.median(rest)) if inside and len(rest) >= 3 else float("nan")


def m_spread(x, text="", ctx=None):
    """Retake: align the recording to the phrase with the recogniser, then compare the level of the flagged words with the rest."""
    ref = [w.strip(",.;:!?\"'").lower() for w in text.split()]
    hyp = asr.words(x, False)
    times = [(0.0, 0.0)] * len(ref)
    for i, j in needleman_wunsch(ref, [h["w"] for h in hyp]):
        if i is not None and j is not None:
            times[i] = (hyp[j]["start"], hyp[j]["end"])
    return word_level(x, times, ctx["r0"], ctx["r1"])


def m_slur(x, text="", ctx=None):
    fr = features.compute(x)
    d, m, idx = _mask(x)
    n = min(len(fr.hf), len(m))
    sel = m[:n]
    return float(np.median(fr.hf[:n][sel])) if sel.any() else float("nan")


def m_pitch(x, text="", ctx=None):
    fr = features.compute(x)
    v = fr.f0_st[~np.isnan(fr.f0_st)]
    return float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 20 else float("nan")


def m_fillers(x, text="", ctx=None):
    return float(sum(1 for e in asr.words(x, True) if e["w"] in FILLER_TOKENS))


def m_words(x, text="", ctx=None):
    ref = [w.strip(",.;:!?\"'").lower() for w in text.split()]
    hyp = [h["w"] for h in asr.words(x, False)]
    path = needleman_wunsch(ref, hyp)
    ok = sum(1 for i, j in path if i is not None and j is not None and sim(ref[i], hyp[j]) >= 0.8)
    return ok / max(len(ref), 1)


MEASURE = {"rate": m_rate, "pause_in": m_pause_in, "pause_end": m_pause_end, "fade": m_fade, "spread": m_spread, "slur": m_slur, "pitch": m_pitch, "fillers": m_fillers, "words": m_words}


# ---------------------------------------------------------------- expected ranges from the rest of the same reading
def _windows(words, flagged: set[int], k: int = 8):
    out, cur = [], []
    for w in words:
        if w["i"] in flagged or w.get("missed"):
            cur = []
            continue
        cur.append(w)
        if len(cur) == k:
            out.append(cur)
            cur = []
    return out


def expected(key, x_full, words, flagged, facts):
    """(lo, hi) of the expected range for feature `key`, from clean windows of the reading itself."""
    vals = []
    if key in ("rate", "pause_in", "fade", "slur", "pitch"):
        for win in _windows(words, flagged)[:14]:
            a, b = int(max(0, win[0]["s"] - 0.05) * A.SR), int((win[-1]["e"] + 0.05) * A.SR)
            v = MEASURE[key](x_full[a:b], " ".join(w["w"] for w in win))
            if np.isfinite(v):
                vals.append(v)
    med = float(np.median(vals)) if vals else None
    if key == "rate":
        med = med or facts.get("clip_median") or 4.0
        return 0.85 * med, 1.20 * med
    if key == "pause_in":
        return 0.0, max(0.30, float(np.percentile(vals, 90)) if vals else 0.0)
    if key == "pause_end":
        return 0.8 * min(0.5, float(facts.get("typical_s", 0.5))), float("inf")
    if key == "fade":
        return float("-inf"), float(np.clip(np.percentile(vals, 90) + 1.0, 5.5, 7.0)) if vals else 6.0
    if key == "spread":
        return float("-inf"), 5.0
    if key == "slur":
        return (float(np.percentile(vals, 25)) - 1.5 if vals else float("-inf")), float("inf")
    if key == "pitch":
        return (0.8 * med if med else 0.0), float("inf")
    if key == "fillers":
        return 0.0, 0.0
    return 0.85, 1.0                                   # words: nearly every word of the text heard (the recogniser itself misses about one in ten)


def _out(v, lo, hi):
    return float(max(0.0, lo - v, v - hi)) if np.isfinite(v) else float("nan")


def _phrase(R, r):
    w, n = R["words"], len(R["words"])
    w0, w1, flaw = r["w0"], r["w1"], r["flaw"]
    if flaw in ("PAUSE_LOST", "FADE"):
        a, b = max(0, w0 - 5), min(n - 1, w1)
    else:
        a, b = max(0, w0 - 3), min(n - 1, w1 + 3)
        if b - a > 24:
            b = a + 24
    return a, b


def prepare(R, rid: int, x_orig):
    """What to read, what is measured, the expected range, and the first attempt's value. x_orig: the original recording (16 kHz mono)."""
    r = R["regions"][rid]
    flaw = r["flaw"]
    if flaw not in CAN:
        raise ValueError("This kind of moment cannot be re-scored on its own.")
    key, name, unit = SPEC[flaw]
    a, b = _phrase(R, r)
    words = R["words"]
    text = " ".join(w["w"] for w in words[a: b + 1] if any(c.isalpha() for c in w["w"]))
    flagged = {w["i"] for w in words if w.get("flag") is not None}
    facts = R["pred"]["what"][rid].get("params", {})
    ctx = {"r0": max(0, r["w0"] - a), "r1": min(b, r["w1"]) - a}
    lo, hi = expected(key, x_orig, words, flagged, facts)
    if key == "pause_end":
        nxt = words[b + 1]["s"] if b + 1 < len(words) else words[b]["e"] + 1.0
        first = max(0.0, float(nxt - words[b]["e"]))
    else:
        s, e = max(0.0, words[a]["s"] - 0.1), words[b]["e"] + 0.1
        seg = x_orig[int(s * A.SR): int(e * A.SR)]
        if key == "spread":
            first = word_level(seg, [(words[i]["s"] - s, words[i]["e"] - s) for i in range(a, b + 1)], ctx["r0"], ctx["r1"])
        else:
            first = MEASURE[key](seg, text)
            if key == "fillers":
                first = max(first, 1.0)                       # the flag itself is a filler sound, even if the recogniser's second listen merges it
    return {"region": rid, "flaw": flaw, "text": text, "feature": key, "name": name, "unit": unit, "first": _num(first), "lo": _num(lo), "hi": _num(hi),
            "ctx": ctx, "points_before": r["points"], "span": [round(words[a]["s"], 2), round(words[b]["e"], 2)],
            "hint": "Say the words above in one go, the way you want them to sound." + (" Then stay quiet for a second before you stop." if key == "pause_end" else "")}


def _num(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


def score(R, rid: int, x_orig, x_retake):
    p = prepare(R, rid, x_orig)
    key = p["feature"]
    lo = p["lo"] if p["lo"] is not None else float("-inf")
    hi = p["hi"] if p["hi"] is not None else float("inf")
    x_retake = np.asarray(x_retake, dtype=np.float32)
    v = MEASURE[key](x_retake, p["text"], p["ctx"])
    if not np.isfinite(v):
        raise ValueError("Could not hear speech in that recording. Try again a little closer to the microphone.")
    out_first = _out(p["first"], lo, hi) if p["first"] is not None else 0.0
    out_new = _out(v, lo, hi)
    inside = out_new == 0.0
    ratio = 0.0 if inside else min(1.0, out_new / out_first) if out_first and out_first > 1e-9 else 1.0
    return {**p, "value": _num(v), "inside": bool(inside), "points_after": round(p["points_before"] * ratio, 2)}
