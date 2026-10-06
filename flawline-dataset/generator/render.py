"""Rendering primitives for v1.1 flaws: how real speakers rush/drag, restart, shout and de-emphasise.
Everything works on the speaker's OWN audio (PSOLA on voiced runs, own room tone for gaps)."""
from __future__ import annotations

import re

import numpy as np

from common import SR, crossfade_concat, db, ffmpeg_filter, high_shelf, rms, s2n, soft_limit, undb
from psola import psola_pitch, psola_stretch

# tempo exponents: how much of a rate change each kind of speech material absorbs (v1.1: real rushing shortens pauses first,
# then vowels, consonants least; real dragging lives in vowels and gaps)
FAST_EXP = {"gap": 1.6, "voiced": 0.9, "cons": 0.35}
SLOW_EXP = {"gap": 1.3, "voiced": 1.0, "cons": 0.15}
JOIN = s2n(0.012)


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def _stretch_plain(seg: np.ndarray, S: float) -> np.ndarray:
    """Consonant material: WSOLA tempo change only when it matters and the piece is long enough."""
    if abs(S - 1.0) < 0.015 or len(seg) < s2n(0.045):
        return seg
    return ffmpeg_filter(seg, f"atempo={1.0 / S:.4f}")


def _gap(ctx, gap: np.ndarray, S: float) -> np.ndarray:
    n = len(gap)
    if n < s2n(0.025) or abs(S - 1.0) < 0.01:
        return gap
    new = int(n * S)
    if new < n:                                          # shorten: drop the middle, keep both soft edges
        h = new // 2
        return crossfade_concat([gap[:h], gap[n - (new - h):]], s2n(0.008))
    extra = ctx.tone(new - n)                            # lengthen: more of the speaker's own room tone in the middle
    h = n // 2
    return crossfade_concat([gap[:h], extra, gap[h:]], s2n(0.008))


def _warp_once(ctx, a: int, b: int, i0: int, i1: int, rate: float, profile=None):
    """Re-time words i0..i1 so the region lasts (b-a)/rate, spending the change on gaps, vowels and consonants in the v1.1 proportions
    and following a tempo profile over the region (ramp in, hold, ease out). Returns (audio, evidence)."""
    W = ctx.words
    fast = rate > 1.0
    E = FAST_EXP if fast else SLOW_EXP
    sgn = -1.0 if fast else 1.0
    pieces = []                                          # (kind, start, end) in original samples, in order
    prev_end = a
    for j in range(i0, i1 + 1):
        w = W[j]
        if w["start"] > prev_end:
            pieces.append(("gap", prev_end, w["start"], j))
        run = ctx.voiced_run(w["start"], w["end"], 0.04)
        if run:
            s0, e0, _ = run
            if s0 > w["start"]:
                pieces.append(("cons", w["start"], s0, j))
            pieces.append(("voiced", s0, e0, j))
            if e0 < w["end"]:
                pieces.append(("cons", e0, w["end"], j))
        else:
            pieces.append(("cons", w["start"], w["end"], j))
        prev_end = w["end"]
    if prev_end < b:
        pieces.append(("gap", prev_end, b, i1))
    total = b - a
    pos = lambda s, e: ((s + e) / 2 - a) / max(total, 1)
    prof = profile or (lambda p: 1.0)

    target = total / rate
    lo, hi = 0.0, 6.0
    for _ in range(40):                                  # bisection on the intensity u (monotone in u)
        mid = (lo + hi) / 2
        L = sum((e - s) * np.exp((-1.0 if fast else 1.0) * E[k] * mid * prof(pos(s, e))) for k, s, e, _ in pieces)
        if (L > target) == fast:
            lo = mid
        else:
            hi = mid
    u = (lo + hi) / 2
    out, changed = [], []
    for k, s, e, j in pieces:
        S = float(np.exp((-1.0 if fast else 1.0) * E[k] * u * prof(pos(s, e))))
        seg = ctx.x[s:e]
        if k == "gap":
            o = _gap(ctx, seg, S)
        elif k == "voiced":
            o = psola_stretch(seg, S, float(np.exp(ctx.f0_median_log))) if abs(S - 1.0) > 0.004 else seg
        else:
            o = _stretch_plain(seg, S)
        out.append(o)
        changed.append(o is not seg)
    audio = out[0] if out else np.zeros(0, np.float32)
    for n_, o in enumerate(out[1:], 1):                  # crossfade only where a modified piece meets its neighbour
        if len(o):
            audio = crossfade_concat([audio, o], JOIN, True) if (changed[n_] or changed[n_ - 1]) else np.concatenate([audio, o])
    syl = sum(syllables(W[j]["clean"]) for j in range(i0, i1 + 1))
    gaps_before = sum(e - s for k, s, e, _ in pieces if k == "gap") / SR
    gaps_after = sum(len(o) for (k, *_), o in zip(pieces, out) if k == "gap") / SR
    ev = {"syll_per_s_before": round(syl / (total / SR), 2), "syll_per_s_after": round(syl / (len(audio) / SR), 2),
          "gaps_s_before": round(gaps_before, 2), "gaps_s_after": round(gaps_after, 2), "u": round(u, 3)}
    return audio.astype(np.float32), ev


def warp_region(ctx, a: int, b: int, i0: int, i1: int, rate: float, profile=None):
    """_warp_once plus a correction pass: crossfades and PSOLA rounding make the achieved duration ratio drift from the target."""
    audio, ev = _warp_once(ctx, a, b, i0, i1, rate, profile)
    for _ in range(3):
        got = (b - a) / max(len(audio), 1)
        if abs(got - rate) / rate < 0.02:
            break
        audio, ev = _warp_once(ctx, a, b, i0, i1, rate * rate / got, profile)
    ev["achieved_rate"] = round((b - a) / len(audio), 3)
    return audio, ev


def tempo_profile(fast: bool):
    """Position p in [0,1] -> tempo weight: rises over the first 30%, holds, eases out toward the clause end."""
    def f(p):
        up = np.clip(p / 0.30, 0, 1)
        up = up * up * (3 - 2 * up)                      # smoothstep
        down = 0.6 + 0.4 * np.clip((1 - p) / 0.20, 0, 1)
        return float(up * down) if fast else float(0.35 + 0.65 * up)
    return f


def partial_attempt(ctx, a: int, b: int, frac: float, up_st: float, gain_db: float) -> np.ndarray:
    """A false start: the span cut off at `frac` of its length, slightly higher (+0.5-1 st) and quieter (-1 dB), fading out."""
    seg = ctx.x[a:b]
    piece = seg[: max(int(frac * len(seg)), s2n(0.05))].copy()
    if len(piece) > s2n(0.08):
        med = float(np.exp(ctx.f0_median_log))
        piece = psola_pitch(piece, lambda f, t: f * 2 ** (up_st / 12), med)
    piece = piece * undb(gain_db)
    n = min(s2n(0.015), len(piece) // 2)
    piece[-n:] *= np.linspace(1, 0, n).astype(np.float32)
    return piece.astype(np.float32)


def shout_bundle(ctx, a: int, seg: np.ndarray, gain_db: float, pure: bool) -> np.ndarray:
    """Vocal effort = louder + brighter + higher. Bundle: high-shelf +1.5 dB per 4 dB of gain above 1 kHz, F0 +0.25 st per dB;
    gain goes on speech only so the room noise stays put. pure=True renders gain only (ablation slice)."""
    y = seg
    if not pure:
        med = float(np.exp(ctx.f0_median_log))
        y = psola_pitch(y, lambda f, t: f * 2 ** (0.25 * gain_db / 12), med)
        y = high_shelf(y, 1000.0, 1.5 * gain_db / 4)
        y = y * (rms(seg) / (rms(y) + 1e-9))             # shelf/pitch must not change loudness: gain is the only loudness cue
    m = s2n(0.03)
    env = np.ones(len(y), np.float32) * undb(gain_db)
    env[:m] = undb(np.linspace(0, gain_db, m))
    env[-m:] = undb(np.linspace(gain_db, 0, m))
    return soft_limit(ctx.shaped_gain(a, y, env))


# ----------------------------------------------------------------------------- focal words (for EMPH_FLAT)
NEG = {"not", "never", "no", "nobody", "nothing", "none", "neither", "nor", "cannot"}
CONTRAST = {"but", "only", "even", "just", "still", "yet", "however", "although", "instead", "rather", "except"}
PHRASE_END = ("sentence", "clause_punct", "before_coord", "before_subord", "before_relative")


def focal_words(ctx) -> dict[int, str]:
    """Words that carry meaning in an interpretive reading: numbers, negations, contrast, superlatives, the nuclear (last content) word of each
    phrase, and the first mention of a noun. {word index: why}"""
    W, L = ctx.words, ctx.ling
    out, seen = {}, set()
    for i, w in enumerate(W):
        a = L["w"][i]
        c = w["clean"]
        if not c and not any(ch.isdigit() for ch in w["w"]):
            continue
        if a["pos"] == "NUM" or any(ch.isdigit() for ch in w["w"]):
            out[i] = "number"
        elif c in NEG or c.endswith("n't"):
            out[i] = "negation"
        elif c in CONTRAST:
            out[i] = "contrast"
        elif a["tag"] in ("JJS", "RBS") or c.endswith("est") and a["pos"] in ("ADJ", "ADV"):
            out[i] = "superlative"
        elif a["pos"] in ("NOUN", "PROPN") and a["lemma"] not in seen:
            out[i] = "first mention"
        elif a["content"] and i < len(W) - 1 and L["b"][i] in PHRASE_END:
            out[i] = "nuclear accent"
        elif a["content"] and i == len(W) - 1:
            out[i] = "nuclear accent"
        if a["pos"] in ("NOUN", "PROPN"):
            seen.add(a["lemma"])
    return out
