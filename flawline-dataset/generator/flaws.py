"""Flaw factory: 15 parameterised injectors, 5 severity levels each (spec "What").

Every injector returns a list of `Edit`s against the ORIGINAL take. `assemble` applies them left to
right with a 10 ms crossfade and reports each region in both clip time and baseline time, because
time-stretching and insertions shift everything after them.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from render import focal_words, partial_attempt, shout_bundle, syllables, tempo_profile, warp_region
from linguistics import HESITATION_WEIGHT, annotate, context, describe
from common import (join_clicks, pink_noise, BASELINES, FILLERS, SR, TAKES, crossfade_concat, db, ffmpeg_filter, frame_rms_db,
                    load_config, read_audio, rms, room_tone, s2n, soft_limit, undb)

CFG = load_config()
FADE = s2n(CFG["crossfade_ms"] / 1000)


# --------------------------------------------------------------------------- data structures
@dataclass
class Edit:
    a: int                      # baseline sample span replaced (a == b: insertion)
    b: int
    seg: np.ndarray             # replacement audio, any length
    label: dict                 # flaw, category, level, params, word_start, word_end, kind, ...
    pre: int = 0                # labelled region extends this far before a / after b in unchanged audio
    post: int = 0


class Ctx:
    """One clean take, loaded once, shared by every injector."""

    def __init__(self, take_id: str, base_condition: str = "C0"):
        self.take_id = take_id
        self.x, sr = read_audio(TAKES / f"{take_id}_{base_condition}.flac")
        assert sr == SR
        al = json.loads((TAKES / f"{take_id}_{base_condition}.align.json").read_text())
        self.words = [{**w, "start": s2n(w["start_s"]), "end": s2n(w["end_s"])} for w in al["words"]]
        self._pool = None
        self.refined = al.get("refined") == 2
        if not self.refined:
            self._refine_bounds()
        self._align_path = TAKES / f"{take_id}_{base_condition}.align.json"
        self.align_source = al["align_source"]
        src = BASELINES / take_id.split("-")[0] / "source.json"
        self.source = json.loads(src.read_text()) if src.exists() else {}
        self.voice = self.source.get("voice")
        self.taken: list[tuple[int, int]] = []
        self._f0_med = None
        self.filler_source = None
        self.pure = False                          # True = single-cue rendering (ablation slice); default = natural bundles
        self.last_join = None
        self.nrng = np.random.default_rng(0)       # content noise (room tone, filler choice); placement uses its own rng

    # -- placement bookkeeping
    def free(self, a: int, b: int, margin_s: float = 0.2) -> bool:
        m = s2n(margin_s)
        return all(b + m <= ta or a - m >= tb for ta, tb in self.taken)

    def claim(self, a: int, b: int, margin_s: float = 0.2) -> bool:
        if not self.free(a, b, margin_s):
            return False
        self.taken.append((a, b))
        return True

    @property
    def dur(self) -> float:
        return len(self.x) / SR

    @property
    def f0_median_log(self) -> float:
        """Speaker's own median log-F0: the anchor MONOTONE compresses toward (identity is normalised away)."""
        if getattr(self, "_f0_med", None) is None:
            import pyworld as pw
            x64 = self.x.astype(np.float64)
            f0, t = pw.dio(x64, SR, f0_floor=71.0, f0_ceil=500.0, frame_period=5.0)
            f0 = pw.stonemask(x64, f0, t, SR)
            self._f0_med = float(np.median(np.log(f0[f0 > 0])))
        return self._f0_med

    @property
    def epochs(self) -> np.ndarray:
        """Glottal closure instants (samples) from Praat's periodic point process: where voiced audio should be cut and joined."""
        if getattr(self, "_ep", None) is None:
            import parselmouth
            from parselmouth.praat import call
            med = float(np.exp(self.f0_median_log))
            snd = parselmouth.Sound(self.x.astype(np.float64), SR)
            pp = call(snd, "To PointProcess (periodic, cc)", max(60.0, med * 0.55), min(600.0, med * 2.4))
            n = call(pp, "Get number of points")
            self._ep = np.array([int(call(pp, "Get time from index", i) * SR) for i in range(1, n + 1)], dtype=np.int64)
        return self._ep

    def snap_cut(self, t: int, rad_s: float = 0.006) -> int:
        """Nearest glottal epoch within rad_s (pitch-synchronous splice point); in unvoiced/silent audio, the nearest zero crossing."""
        ep = self.epochs
        if len(ep):
            i = int(np.searchsorted(ep, t))
            cand = [ep[j] for j in (i - 1, i) if 0 <= j < len(ep)]
            best = min(cand, key=lambda e: abs(e - t))
            if abs(best - t) <= s2n(rad_s):
                return int(best)
        rad = s2n(0.0015)
        lo, hi = max(1, t - rad), min(len(self.x) - 1, t + rad)
        return lo + int(np.argmin(np.abs(self.x[lo:hi]))) if hi > lo else t

    @property
    def floor_db(self) -> float:
        if getattr(self, "_floor", None) is None:
            h = s2n(0.01)
            env = np.array([db(rms(self.x[i:i + 2 * h])) for i in range(0, len(self.x) - 2 * h, h)])
            self._floor = float(np.percentile(env, 5))
        return self._floor

    def shaped_gain(self, a: int, seg: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Apply a gain envelope g (linear, per sample) to SPEECH only: frames near the clip's noise floor keep gain 1, so the room noise
        does not breathe with the voice (a detector must not learn 'floor moved = flaw')."""
        h = s2n(0.01)
        n = len(seg)
        e = np.array([db(rms(seg[i:i + 2 * h])) for i in range(0, max(1, n - 2 * h), h)])
        w = np.clip((e - (self.floor_db + 8.0)) / 10.0, 0.0, 1.0)
        w = np.convolve(w, np.ones(3) / 3, mode="same")
        wfull = np.interp(np.arange(n), (np.arange(len(w)) * h + h), w)
        return (seg * (1.0 + (g - 1.0) * wfull)).astype(np.float32)

    # -- cut-point quality: ASR word times are only +-40 ms, so cut where the signal really is quiet, at a glottal epoch
    def _refine_bounds(self):
        x, W = self.x, self.words
        h = s2n(0.005)
        env = np.array([db(rms(x[i:i + 2 * h])) for i in range(0, len(x) - 2 * h, h)])
        env = np.convolve(env, np.ones(3) / 3, mode="same")

        def zc(t, rad=s2n(0.0015)):
            return self.snap_cut(t)

        for k in range(len(W) - 1):
            e, st = W[k]["end"], W[k + 1]["start"]
            if st - e < s2n(0.06):                                   # touching words share one cut point: the deepest valley nearby
                c = (e + st) // 2
                lo = max(W[k]["start"] + s2n(0.03), c - s2n(0.05)) // h
                hi = min(W[k + 1]["end"] - s2n(0.03), c + s2n(0.05)) // h
                if hi > lo:
                    c = (lo + int(np.argmin(env[lo:min(hi + 1, len(env))]))) * h + h
                W[k]["end"] = W[k + 1]["start"] = zc(c)
            else:                                                    # real pause: keep the edges, just land them on zero crossings
                W[k]["end"], W[k + 1]["start"] = zc(e), zc(st)

    # -- transcript intelligence + measured headroom
    @property
    def ling(self) -> dict:
        if getattr(self, "_ling", None) is None:
            self._ling = annotate(self.words)
        return self._ling

    def f0_track(self):
        if getattr(self, "_f0t", None) is None:
            import parselmouth
            med = float(np.exp(self.f0_median_log))
            p = parselmouth.Sound(self.x.astype(np.float64), SR).to_pitch(0.01, max(60.0, med * 0.55), min(600.0, med * 2.4))
            f = p.selected_array["frequency"]
            self._f0t = (p.xs(), np.where(f > 0, 12 * np.log2(np.maximum(f, 1.0)), np.nan))
        return self._f0t

    def f0_iqr(self, a: int, b: int) -> float:
        """Semitone IQR of voiced F0 in [a, b): how much expressive pitch movement there is to lose."""
        t, st = self.f0_track()
        v = st[(t >= a / SR) & (t <= b / SR)]
        v = v[~np.isnan(v)]
        return float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 10 else 0.0

    def hf_stat(self, a: int, b: int, x: np.ndarray | None = None) -> float:
        """Consonant strength: mean level (dB) of the loudest 30% of 2-8 kHz frames in [a, b). Slurring lowers it."""
        from scipy.signal import stft
        seg = (self.x if x is None else x)[a:b].astype(np.float64)
        if len(seg) < 1024:
            return -120.0
        f, _, Z = stft(seg, SR, nperseg=512, noverlap=256)
        band = (f >= 2000) & (f <= 8000)
        e = 10 * np.log10(np.mean(np.abs(Z[band]) ** 2, axis=0) + 1e-14)
        return float(np.mean(np.sort(e)[-max(1, int(0.3 * len(e))):]))

    def voiced_run(self, a: int, b: int, min_s: float = 0.05):
        """Longest voiced stretch inside [a, b): (start, end, median f0 Hz) in samples, or None. Cached (hot in PACE/drawl)."""
        cache = self.__dict__.setdefault("_vr", {})
        key = (a, b, round(min_s, 3))
        if key not in cache:
            cache[key] = self._voiced_run(a, b, min_s)
        return cache[key]

    def _voiced_run(self, a: int, b: int, min_s: float = 0.05):
        import parselmouth
        if b - a < s2n(0.05):
            return None
        med = float(np.exp(self.f0_median_log))
        p = parselmouth.Sound(self.x[a:b].astype(np.float64), SR).to_pitch(0.005, max(60.0, med * 0.55), min(600.0, med * 2.4))
        f, t = p.selected_array["frequency"], p.xs()
        v = f > 0
        best, i = (0, 0), 0
        while i < len(v):
            if v[i]:
                j = i
                while j + 1 < len(v) and (v[j + 1] or (j + 2 < len(v) and v[j + 2])):
                    j += 1
                if j - i > best[1] - best[0]:
                    best = (i, j)
                i = j + 1
            else:
                i += 1
        s0, e0 = a + int(max(0, t[best[0]] - 0.0025) * SR), a + int((t[best[1]] + 0.0025) * SR)
        if (e0 - s0) / SR < min_s:
            return None
        return s0, min(e0, b), float(np.median(f[best[0]: best[1] + 1]))

    def save_refined(self):
        """Persist the snapped boundaries so labels, QA and the dashboard all use the SAME word times."""
        if self.refined:
            return
        al = json.loads(self._align_path.read_text())
        for w, r in zip(al["words"], self.words):
            w["start_s"], w["end_s"] = round(r["start"] / SR, 4), round(r["end"] / SR, 4)
        if "snap" not in al["align_source"]:
            al["align_source"] += " + energy-valley/glottal-epoch snap"
        al["refined"] = 2
        self._align_path.write_text(json.dumps(al, indent=2) + "\n")

    def tone(self, n: int) -> np.ndarray:
        """Room tone of length n built from THIS clip's own quiet frames (same noise colour, level and mic), never synthetic."""
        if self._pool is None:
            x, W, h = self.x, self.words, s2n(0.01)
            env = np.array([db(rms(x[i:i + 2 * h])) for i in range(0, len(x) - 2 * h, h)])
            floor = np.percentile(env, 5)
            speech = np.zeros(len(env), bool)
            for w in W:
                speech[max(0, (w["start"] - s2n(0.08)) // h): (w["end"] + s2n(0.08)) // h + 1] = True
            quiet = (env < floor + 3.5) & ~speech
            runs, i = [], 0
            while i < len(quiet):
                if quiet[i]:
                    j = i
                    while j < len(quiet) and quiet[j]:
                        j += 1
                    if (j - i) * h >= s2n(0.15):
                        runs.append(x[i * h: j * h + h])
                    i = j
                else:
                    i += 1
            self._pool = runs if sum(len(r) for r in runs) >= s2n(0.3) else []
        if not self._pool:
            return room_tone(n, self.x, self.nrng)
        out, f, piece = np.zeros(0, np.float32), s2n(0.02), s2n(0.12)
        while len(out) < n + f:
            r = self._pool[int(self.nrng.integers(0, len(self._pool)))]
            if len(r) <= piece:
                seg = r
            else:
                o = int(self.nrng.integers(0, len(r) - piece))
                seg = r[o:o + piece]
            out = crossfade_concat([out, seg], f) if len(out) else seg.copy()
        return out[:n].astype(np.float32)

    def sentences(self) -> list[tuple[int, int]]:
        out, s = [], 0
        for k, w in enumerate(self.words):
            nxt = self.words[k + 1]["start"] if k + 1 < len(self.words) else None
            pause_end = nxt is not None and (nxt - w["end"]) > s2n(0.45)    # spontaneous speech: a long pause ends a phrase
            if w["sent_end"] or pause_end:
                out.append((s, w["i"]))
                s = w["i"] + 1
        if s < len(self.words):
            out.append((s, len(self.words) - 1))
        return out

    def span(self, i0: int, i1: int) -> tuple[int, int]:
        return self.words[i0]["start"], self.words[i1]["end"]


# --------------------------------------------------------------------------- DSP primitives
def time_stretch(seg: np.ndarray, speed: float) -> np.ndarray:
    """Tempo change with pitch preserved (ffmpeg atempo, WSOLA-style)."""
    if abs(speed - 1.0) < 1e-3:
        return seg
    return ffmpeg_filter(seg, f"atempo={speed:.4f}")


def pitch_edit(ctx: "Ctx", seg: np.ndarray, f0_fn, a: int | None = None, pad_s: float = 0.06) -> np.ndarray:
    """F0 edit with PSOLA: original periods are re-used, so the speaker's own timbre/noise survive (WORLD adds vocoder buzz).
    With `a` (the segment's start in the clip) PSOLA runs on [a-pad, b+pad] of the original and is cropped back: the splice then falls
    where the pitch is unchanged and both sides are glottally in phase, instead of at PSOLA's re-synthesised first/last period."""
    from psola import psola_pitch
    med = float(np.exp(ctx.f0_median_log))
    if a is None:
        return psola_pitch(seg, f0_fn, med)
    pad = s2n(pad_s)
    lo, hi = max(0, a - pad), min(len(ctx.x), a + len(seg) + pad)
    off = a - lo
    dur = len(seg) / SR

    def wrapped(f, t):
        tr = t - off / SR
        out = f.copy()
        m = (tr >= 0) & (tr <= dur)
        if m.sum() >= 3:
            out[m] = f0_fn(f[m], tr[m])
        return out

    y = psola_pitch(ctx.x[lo:hi], wrapped, med)
    return y[off: off + len(seg)]


def world_edit(seg: np.ndarray, f0_fn, gain_fn=None) -> np.ndarray:
    """Analyse with WORLD, edit F0 via f0_fn(f0, t_s, sp) -> f0', resynthesise at the original loudness."""
    import pyworld as pw
    x64 = seg.astype(np.float64)
    f0, t = pw.harvest(x64, SR, f0_floor=71.0, f0_ceil=500.0, frame_period=5.0)
    sp = pw.cheaptrick(x64, f0, t, SR)
    ap = pw.d4c(x64, f0, t, SR)
    y = pw.synthesize(f0_fn(f0, t), sp, ap, SR, 5.0).astype(np.float32)
    y = np.pad(y, (0, max(0, len(seg) - len(y))))[: len(seg)]
    y *= rms(seg) / (rms(y) + 1e-9)
    return y


def stress_scores(seg: np.ndarray, words: list[dict], a0: int) -> list[float]:
    """Per-word prominence (z-sum of F0 peak, energy, duration): the buried-emphasis score."""
    import pyworld as pw
    x64 = seg.astype(np.float64)
    f0, t = pw.harvest(x64, SR, f0_floor=71.0, f0_ceil=500.0, frame_period=5.0)
    lf = np.where(f0 > 0, np.log2(np.maximum(f0, 1)) * 12, np.nan)
    peak, en, du = [], [], []
    for w in words:
        s, e = (w["start"] - a0) / SR, (w["end"] - a0) / SR
        m = (t >= s) & (t <= e)
        v = lf[m]
        v = v[np.isfinite(v)]
        if len(v) >= 5:
            from scipy.signal import medfilt
            v = medfilt(v, 5)
        peak.append(float(np.percentile(v, 90)) if len(v) else np.nan)
        en.append(db(rms(seg[w["start"] - a0: w["end"] - a0])))
        du.append(e - s)
    def z(v):
        v = np.array(v, float)
        v = np.where(np.isnan(v), np.nanmean(v), v)
        return (v - v.mean()) / (v.std() + 1e-9)
    return list(z(peak) + z(en) + z(du))


# --------------------------------------------------------------------------- fillers
def _drift(y, f_start, fall=0.10):
    from psola import psola_pitch
    return psola_pitch(y, lambda f, t: np.full_like(f, f_start) * (1.0 - fall * (t - t.min()) / max(float(np.ptp(t)), 1e-3)),
                       f_start)


def own_filler(ctx: "Ctx", k: int, token: str = "uh", rng=None) -> tuple[np.ndarray, str]:
    """A filler in THE SPEAKER'S OWN VOICE built only from SCHWA nuclei ('the' before a consonant, 'a', reduced 'of'): nuclei from 'and' or
    'to' give 'aaa'/'ooo', not 'uh'. Voiced nucleus only, PSOLA-stretched to 0.3-0.55 s, started near the speaker's low pitch (10th
    percentile F0; hesitations drop in pitch) and falling ~10%, at ~ -2 dB of the previous word; 'um' closes into a low-passed hum.
    No synthetic or foreign audio: returns (None, None) if the speaker has no usable schwa nearby."""
    from scipy.signal import butter, sosfilt
    from psola import psola_stretch
    W = ctx.words
    t_ins = W[k]["end"]

    def schwa(i):
        w = W[i]
        if w["punct"] or not s2n(0.05) <= w["end"] - w["start"] <= s2n(0.30):
            return False
        if w["clean"] in ("a", "of"):
            return True
        return w["clean"] == "the" and i + 1 < len(W) and W[i + 1]["clean"][:1] not in ("a", "e", "i", "o", "u", "")

    near = sorted((i for i in range(len(W)) if schwa(i)), key=lambda i: abs(W[i]["start"] - t_ins))[:14]
    pick = []
    for i in near:
        run = ctx.voiced_run(W[i]["start"], W[i]["end"], 0.040)
        if run:
            pick.append((W[i], run))
    if not pick:
        return None, None
    w, (s0, e0, f_src) = pick[int(ctx.nrng.integers(0, len(pick)))]
    core = ctx.x[s0 + int(0.1 * (e0 - s0)): e0 - int(0.1 * (e0 - s0))]
    _, st_track = ctx.f0_track()
    v = st_track[~np.isnan(st_track)]
    f_low = float(2 ** (np.percentile(v, 10) / 12) * 1.05) if len(v) > 20 else f_src
    target = (0.30 if token == "uh" else 0.34) + 0.2 * float(ctx.nrng.random())
    y = psola_stretch(core, min(6.0, max(1.0, target * SR / len(core))), f_low)
    y = _drift(y, f_low)
    if token == "um":
        hum = sosfilt(butter(2, 700, btype="low", fs=SR, output="sos"), y[-s2n(0.14):]) * 0.6
        y = crossfade_concat([y, hum.astype(np.float32)], s2n(0.03))
    lvl = rms(ctx.x[W[k]["start"]: W[k]["end"]]) * undb(-2.0)
    y = y * (lvl / (rms(y) + 1e-9))
    n_in, n_out = s2n(0.015), s2n(0.03)
    y = y.astype(np.float32).copy()
    y[:n_in] *= np.linspace(0, 1, n_in)
    y[-n_out:] *= np.linspace(1, 0, n_out)
    return y, "own-voice-schwa" + ("+hum" if token == "um" else "")


def drag_word(ctx: "Ctx", k: int, factor: float):
    """Drawl: lengthen the voiced vowel of word k in place ('and' -> 'aaand'), keeping its own onset and coda. Returns the
    replacement audio for the word span, or None if it has no voiced vowel long enough."""
    from psola import psola_stretch
    W = ctx.words
    a, b = W[k]["start"], W[k]["end"]
    run = ctx.voiced_run(a, b, 0.045)
    if not run:
        return None
    s0, e0, f0 = run
    core = psola_stretch(ctx.x[s0:e0], factor, f0)
    core = _drift(core, f0, 0.08)
    return crossfade_concat([ctx.x[a:s0], core, ctx.x[e0:b]], s2n(0.008), True)


def windows(ctx: Ctx, lo_s: float, hi_s: float, zone: tuple[int, int], prefer_clause=True):
    W, out = ctx.words, []
    for i0 in range(len(W)):
        if W[i0]["start"] < zone[0]:
            continue
        for i1 in range(i0, len(W)):
            if W[i1]["end"] > zone[1]:
                break
            d = (W[i1]["end"] - W[i0]["start"]) / SR
            if d > hi_s:
                break
            if d >= lo_s:
                out.append((i0, i1))
    if prefer_clause:
        nice = [(i0, i1) for i0, i1 in out if (i0 == 0 or W[i0 - 1]["punct"]) and W[i1]["punct"]]
        if len(nice) >= 3:
            return nice
    return out


def pick_windows(ctx, rng, n, lo_s, hi_s, zone, prefer_clause=True):
    cands = windows(ctx, lo_s, hi_s, zone, prefer_clause) or windows(ctx, lo_s * 0.6, hi_s * 1.3, zone, False)
    picked = []
    for k in rng.permutation(len(cands)):
        i0, i1 = cands[k]
        a, b = ctx.span(i0, i1)
        if ctx.claim(a, b):
            picked.append((i0, i1))
            if len(picked) == n:
                break
    return sorted(picked)


def boundaries(ctx, zone, pred):
    """Word boundaries k|k+1 inside the zone satisfying pred(k)."""
    W = ctx.words
    return [k for k in range(2, len(W) - 3)
            if zone[0] <= W[k]["end"] and W[k + 1]["start"] <= zone[1] and pred(k)]


def lab(code, level, params, i0, i1, kind="modify", **extra):
    c = CFG["flaws"][code]
    return {"flaw": code, "category": c["category"], "level": level, "params": params,
            "word_start": i0, "word_end": i1, "kind": kind, **extra}


# --------------------------------------------------------------------------- injectors
def weighted_order(rng, items, weights):
    """Weighted random order without replacement (Efraimidis-Spirakis); zero weights never appear."""
    keyed = [(float(rng.random()) ** (1.0 / w), it) for it, w in zip(items, weights) if w > 0]
    return [it for _, it in sorted(keyed, key=lambda kv: -kv[0])]


def explain(ctx, k):
    """Why this boundary: boundary type, readable context, plain-English reason (goes into the label)."""
    W, bt = ctx.words, ctx.ling["b"][k]
    strip = lambda t: t.strip(",.;:!?")
    return {"boundary": bt, "context": context(W, k), "why": describe("", bt, strip(W[k]["w"]), strip(W[k + 1]["w"]))}


def n_regions(rng) -> int:
    return int(rng.integers(1, CFG["max_regions"] + 1))


def inj_pace(ctx, code, level, rng, zone, n):
    """Pacing the way people actually rush / drag. FAST: tempo ramps up over the first 30% of the region, gaps shrink most, vowels next,
    consonants least; placed in lists, long sentences and the last third (time pressure). SLOW: vowels and gaps stretch, consonants
    barely; placed at sentence openings and before hard words (uncertainty). pure=True gives the old uniform stretch (ablation)."""
    from wordfreq import zipf_frequency
    c = CFG["flaws"][code]
    rate = c["levels"][level - 1]
    lo, hi = c["region_s"]
    fast = rate > 1.0
    W = ctx.words
    cands = windows(ctx, lo, hi, zone, True) or windows(ctx, lo * 0.7, hi * 1.3, zone, False)
    sent_of = {}
    for s_, e_ in ctx.sentences():
        for i in range(s_, e_ + 1):
            sent_of[i] = (s_, e_)
    n_total = len(ctx.x)

    def weight(win):
        i0, i1 = win
        s_, e_ = sent_of[i0]
        w = 1.0
        if fast:
            if sum(1 for j in range(i0, i1) if W[j]["clause_end"]) >= 2:
                w += 1.5                                                    # a list: "six spoons..., five slabs..., and..."
            if e_ - s_ + 1 >= 15:
                w += 1.0
            if W[i0]["start"] > 2 * n_total / 3:
                w += 1.2                                                    # last third of the clip
        else:
            if i0 - s_ <= 2:
                w += 2.0                                                    # sentence opening
            if any(len(W[j]["clean"]) > 4 and ctx.ling["w"][j]["content"] and zipf_frequency(W[j]["clean"], "en") < 3.5
                   for j in range(i0, i1 + 1)):
                w += 1.5                                                    # a hard word in the window
        return w

    edits = []
    for i0, i1 in weighted_order(rng, cands, [weight(w) for w in cands]):
        a, b = ctx.span(i0, i1)
        if not ctx.claim(a, b):
            continue
        if ctx.pure:
            audio, ev = time_stretch(ctx.x[a:b], rate), {"uniform_atempo": rate}
        else:
            audio, ev = warp_region(ctx, a, b, i0, i1, rate, tempo_profile(fast))
        why = ("rushing a list / long sentence" if fast else "dragging at a sentence opening / hard word")
        edits.append(Edit(a, b, audio, lab(code, level, {"rate": rate, "bundle": not ctx.pure, "evidence": ev}, i0, i1,
                                            context=context(W, i0, i1), why=why)))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_pause_bad(ctx, code, level, rng, zone, n):
    """A silent pause INSIDE a tight phrase (the | people). Real speakers who stop after 'the' lengthen it ('thee...') and let it decay:
    the pre-pause function word's vowel is drawn out 1.3-1.6x. Claims are level-independent so every level uses the same anchors."""
    pause = CFG["flaws"][code]["levels"][level - 1]
    W, L = ctx.words, ctx.ling
    ks = boundaries(ctx, zone, lambda k: L["b"][k] == "phrase_tight" and not W[k]["punct"] and not W[k + 1]["punct"]
                    and W[k + 1]["start"] - W[k]["end"] < s2n(0.08))
    wts = [3.0 if (L["w"][k]["reducible"] and len(W[k]["clean"]) <= 3) else 1.5 if L["w"][k]["function"] else 0.8 for k in ks]
    edits = []
    for k in weighted_order(rng, ks, wts):
        k = int(k)
        a = (W[k]["end"] + W[k + 1]["start"]) // 2
        drag = None
        if not ctx.pure and L["w"][k]["reducible"] and len(W[k]["clean"]) <= 4:
            drag = drag_word(ctx, k, 1.3 + 0.075 * (level - 1))
        lo = W[k]["start"] if drag is not None else a
        if not ctx.claim(lo, a + s2n(1.2)):
            continue
        if drag is not None:
            edits.append(Edit(W[k]["start"], W[k]["end"], drag, lab(code, level, {"token": "drawl-before-pause"}, k, k, "modify",
                                                                  why=f"'{W[k]['w']}' drawn out before the pause")))
        edits.append(Edit(a, a, ctx.tone(s2n(pause)), lab(code, level, {"pause_s": pause, "bundle": not ctx.pure}, k, k + 1, "insert",
                                                          inserted_s=pause, **explain(ctx, k))))
        if sum(1 for e in edits if e.label["kind"] == "insert") >= n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_pause_lost(ctx, code, level, rng, zone, n):
    """Remove a rhetorical pause the speaker CHOSE to make. VCTK utterance joins are editing artifacts (0.3 s margins), so only commas inside
    an utterance qualify there; candidates are ranked by pause length (longest = most deliberate). A 50 ms floor always remains: a speaker
    never truly fuses two clauses, and fusing them reads as a splice."""
    frac = CFG["flaws"][code]["levels"][level - 1]
    W, Lg = ctx.words, ctx.ling
    vctk = str(ctx.source.get("corpus", "")).startswith("VCTK")
    types = ("clause_punct",) if vctk else ("clause_punct", "sentence")
    floor = s2n(0.05)
    ks = boundaries(ctx, zone, lambda k: Lg["b"][k] in types and not W[k + 1].get("utt_start")
                    and W[k + 1]["start"] - W[k]["end"] > s2n(0.12) + floor)
    wts = [((W[k + 1]["start"] - W[k]["end"]) / SR) ** 2 for k in ks]
    edits = []
    for k in weighted_order(rng, ks, wts):
        k = int(k)
        g0, g1 = W[k]["end"], W[k + 1]["start"]
        gap = g1 - g0
        r = int(min(frac * gap, gap - floor))
        a = g0 + ((gap - r) // 2)
        if ctx.claim(g0, g1):
            edits.append(Edit(a, a + r, np.zeros(0, np.float32),
                              lab(code, level, {"removed_frac": frac, "baseline_pause_s": round(gap / SR, 3), "floor_s": 0.05},
                                  k, k + 1, "delete", removed_s=round(r / SR, 3), **explain(ctx, k)),
                              pre=a - g0, post=g1 - (a + r)))
            if len(edits) == n:
                break
    return sorted(edits, key=lambda e: e.a)


def inj_monotone(ctx, code, level, rng, zone, n):
    """Flatten pitch where the speaker is MOST expressive (highest F0 IQR): flattening a flat stretch proves nothing."""
    c = CFG["flaws"][code]
    mult = c["levels"][level - 1]
    med = ctx.f0_median_log

    def f0_fn(f0, t):
        out = f0.copy()
        v = f0 > 0
        out[v] = np.exp(med + mult * (np.log(f0[v]) - med))
        return out

    cands = windows(ctx, *c["region_s"], zone, True) or windows(ctx, 2.0, 6.0, zone, False)
    scored = sorted(cands, key=lambda w: -(ctx.f0_iqr(*ctx.span(*w)) + 0.4 * float(rng.random())))
    edits = []
    for i0, i1 in scored:
        a, b = ctx.span(i0, i1)
        if not ctx.claim(a, b):
            continue
        before = ctx.f0_iqr(a, b)
        y = pitch_edit(ctx, ctx.x[a:b], f0_fn, a)
        import parselmouth
        pm = parselmouth.Sound(y.astype(np.float64), SR).to_pitch(0.01, 70.0, 500.0).selected_array["frequency"]
        pm = 12 * np.log2(pm[pm > 0]) if (pm > 0).sum() > 10 else np.array([0.0, 0.0])
        after = float(np.percentile(pm, 75) - np.percentile(pm, 25))
        edits.append(Edit(a, b, y, lab(code, level, {"range_mult": mult, "evidence": {"f0_iqr_before_st": round(before, 2), "f0_iqr_after_st": round(after, 2),
                                                                                    "audible": bool(before - after >= 1.0)}}, i0, i1,
                                      context=context(ctx.words, i0, i1), why=f"flattened the most expressive stretch (F0 IQR {before:.1f} st)")))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_uptalk(ctx, code, level, rng, zone, n):
    """Rising end on a statement. v1.1 fix: REPLACE the tail contour, anchored at the nucleus (loudest voiced point of the last word):
    F0 = F0(nucleus) * 2^(st*ramp/12) with the original micro-jitter, instead of multiplying a ramp onto the natural final fall (which
    cancelled out). Placed mid-paragraph / on list items rather than at the conclusion."""
    import parselmouth
    from scipy.ndimage import uniform_filter1d
    st = CFG["flaws"][code]["levels"][level - 1]
    W = ctx.words
    med = float(np.exp(ctx.f0_median_log))
    sents = [(s_, e_) for s_, e_ in ctx.sentences() if W[e_]["punct"] != "?" and e_ - s_ >= 5
             and W[s_]["start"] >= zone[0] and W[e_]["end"] <= zone[1]]
    last = max((e_ for _, e_ in ctx.sentences()), default=0)
    wts = []
    for s_, e_ in sents:
        wt = 1.0 + 0.6 * (sum(1 for j in range(s_, e_) if W[j]["clause_end"]) >= 2)       # list items
        wt *= 0.25 if e_ == last else (0.6 if s_ == 0 else 1.0)                           # not the conclusion / opener
        wts.append(wt)
    edits = []
    for k in weighted_order(rng, list(range(len(sents))), wts):
        s_, e_ = sents[k]
        i0 = e_ - 2
        while i0 > s_ + 1 and (W[e_]["end"] - W[i0]["start"]) / SR < 1.0:
            i0 -= 1
        a, b = ctx.span(i0, e_)
        if not ctx.claim(a, b):
            continue
        seg = ctx.x[a:b]
        info = {}

        def f0_fn(f0, t, seg=seg, a=a, e_=e_, info=info):
            h = s2n(0.01)
            inten = np.array([db(rms(seg[max(0, int(ti * SR) - h): int(ti * SR) + h])) for ti in t])
            lw = (W[e_]["start"] - a) / SR
            cand = np.where(t >= lw)[0]
            if len(cand) < 4:
                cand = np.arange(max(0, len(t) - 10), len(t))
            i_n = int(cand[np.argmax(inten[cand])])
            if len(t) - 1 - i_n < 3:
                i_n = max(0, len(t) - 12)
            t_n, f_n, t_end = t[i_n], f0[i_n], t[-1]
            jitter = f0 / np.maximum(uniform_filter1d(f0, 5), 1.0)
            out = f0.copy()
            idx = np.arange(len(f0)) > i_n
            ramp = np.clip((t[idx] - t_n) / max(t_end - t_n, 1e-3), 0, 1)
            out[idx] = f_n * 2 ** (st * ramp / 12) * jitter[idx]
            info.update(t_nucleus=float(t_n), f_nucleus=float(f_n))
            return out

        y = pitch_edit(ctx, seg, f0_fn, a)
        p = parselmouth.Sound(y.astype(np.float64), SR).to_pitch(0.005, max(60.0, med * 0.55), min(600.0, med * 2.4)).selected_array["frequency"]
        v = p[p > 0]
        rise = float(12 * np.log2(np.median(v[-10:]) / info["f_nucleus"])) if len(v) > 12 and info else 0.0
        edits.append(Edit(a, b, y, lab(code, level, {"semitones": st, "evidence": {"rise_after_nucleus_st": round(rise, 2),
                                                                               "nucleus_s": round(a / SR + info.get("t_nucleus", 0.0), 3)}},
                                      i0, e_, context=context(W, i0, e_), why="rising end anchored at the last stressed vowel")))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_emph_flat(ctx, code, level, rng, zone, n):
    """Buried emphasis on the words that CARRY MEANING (numbers, negations, contrast, superlatives, the nuclear word of each phrase, first
    mentions), not whichever word happens to be loud. All three stress cues are reduced on those words: F0 excursion (toward the region's
    mean pitch), intensity (toward the region's mean level, speech only) and stressed-vowel length (PSOLA, up to -25%). Evidence = the
    per-word prominence z-score (F0 peak + energy + duration) before and after."""
    from psola import psola_stretch
    c = CFG["flaws"][code]
    p = c["levels"][level - 1]
    W = ctx.words
    med = float(np.exp(ctx.f0_median_log))
    focal = focal_words(ctx)
    if not hasattr(ctx, "_prom"):
        ctx._prom = stress_scores(ctx.x, W, 0)
    prom = ctx._prom
    cands = windows(ctx, *c["region_s"], zone, True) or windows(ctx, 2.0, 6.0, zone, False)
    score = lambda w: sum(max(prom[j], 0.0) for j in range(w[0], w[1] + 1) if j in focal)
    scored = sorted(cands, key=lambda w: -(score(w) + 0.5 * float(rng.random())))
    edits = []
    for i0, i1 in scored:
        fw = sorted([j for j in range(i0, i1 + 1) if j in focal], key=lambda j: -prom[j])[:3]
        if not fw:
            continue
        a, b = ctx.span(i0, i1)
        if not ctx.claim(a, b):
            continue
        seg = ctx.x[a:b]
        lvl_region = db(rms(seg))
        pieces, cur, new_spans = [], a, {}
        for j in sorted(fw):
            ws, we = W[j]["start"], W[j]["end"]
            if ws < cur:
                continue
            pieces.append(ctx.x[cur:ws])
            word = ctx.x[ws:we]
            tt, st_track = ctx.f0_track()
            vv = st_track[(tt >= a / SR) & (tt <= b / SR)]
            vv = vv[~np.isnan(vv)]
            mean_hz = float(2 ** (np.mean(vv) / 12)) if len(vv) > 5 else med

            def f0_fn(f, t, mean_hz=mean_hz):
                return np.exp(np.log(mean_hz) + (1 - p) * (np.log(f) - np.log(mean_hz)))

            wd = pitch_edit(ctx, word, f0_fn, ws)
            over = max(0.0, db(rms(word)) - lvl_region)
            wd = ctx.shaped_gain(ws, wd, np.full(len(wd), undb(-p * over), np.float32))
            run = ctx.voiced_run(ws, we, 0.045)
            if run and not ctx.pure:
                s0, e0, _ = run
                core = psola_stretch(ctx.x[s0:e0], 1.0 - 0.25 * p, med)
                wd = crossfade_concat([wd[: s0 - ws], core, wd[e0 - ws:]], s2n(0.01), True)
            pieces.append(wd)
            new_spans[j] = (sum(len(q) for q in pieces) - len(wd), sum(len(q) for q in pieces))
            cur = we
        pieces.append(ctx.x[cur:b])
        y = crossfade_concat([q for q in pieces if len(q)], s2n(0.01), True) if len(pieces) > 1 else pieces[0]
        rw = [k_ for k_ in range(i0, i1 + 1)]
        before_w = [{"i": k_, "start": W[k_]["start"], "end": W[k_]["end"]} for k_ in rw]
        zb_all = stress_scores(seg, before_w, a)
        shift, after_w = 0, []
        for k_ in rw:                                          # new word spans: only edited words changed length
            st_ = W[k_]["start"] - a + shift
            if k_ in new_spans:
                after_w.append({"i": k_, "start": a + new_spans[k_][0], "end": a + new_spans[k_][1]})
                shift = new_spans[k_][1] - (W[k_]["end"] - a)
            else:
                after_w.append({"i": k_, "start": a + st_, "end": a + st_ + (W[k_]["end"] - W[k_]["start"])})
        za_all = stress_scores(y, after_w, a)
        z_b = [round(float(zb_all[k_ - i0]), 2) for k_ in fw]
        z_a = [round(float(za_all[k_ - i0]), 2) for k_ in fw]
        edits.append(Edit(a, b, y, lab(code, level, {"flatten": p, "bundle": not ctx.pure, "focal_words": [W[j]["clean"] for j in fw],
                                                    "evidence": {"prominence_z_before": z_b, "prominence_z_after": z_a,
                                                                 "prominence_drop": round(float(np.mean(z_b) - np.mean(z_a)), 2),
                                                                 "reasons": [focal[j] for j in fw]}}, i0, i1,
                                      context=context(W, i0, i1), why="flattened the focal words: " + ", ".join(f"'{W[j]['clean']}' ({focal[j]})" for j in fw))))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_fade(ctx, code, level, rng, zone, n):
    """Trailing off at the end of a sentence, weighted by the breath budget (seconds since the last real pause): people fade when they
    run out of air. The final sentence of the clip is always a candidate. Gain touches speech only, never the room noise."""
    end_db = CFG["flaws"][code]["levels"][level - 1]
    W = ctx.words
    zone = (zone[0], len(ctx.x) - s2n(0.3))
    sents = [(s_, e_) for s_, e_ in ctx.sentences() if e_ - s_ >= 5 and W[s_]["start"] >= zone[0] and W[e_]["end"] <= zone[1]]
    last = max((e_ for _, e_ in ctx.sentences()), default=0)

    def breath(s_, e_):
        t = 0.0
        for j in range(e_, s_, -1):
            if W[j]["start"] - W[j - 1]["end"] > s2n(0.25):
                break
            t += (W[j]["end"] - W[j]["start"]) / SR + max(0, W[j]["start"] - W[j - 1]["end"]) / SR
        return t

    wts = [1.0 + breath(s_, e_) / 2.5 + (2.0 if e_ == last else 0.0) for s_, e_ in sents]
    edits = []
    for k in weighted_order(rng, list(range(len(sents))), wts):
        s_, e_ = sents[k]
        i0 = e_
        while i0 > s_ + 2 and (W[e_]["end"] - W[i0 - 1]["start"]) / SR < 1.4:
            i0 -= 1
        a, b = ctx.span(i0, e_)
        if (b - a) / SR > 3.0 or not ctx.claim(a, b):
            continue
        g = undb(np.linspace(0, end_db, b - a)).astype(np.float32)
        edits.append(Edit(a, b, ctx.shaped_gain(a, ctx.x[a:b], g),
                          lab(code, level, {"end_db": end_db, "breath_s": round(breath(s_, e_), 2)}, i0, e_,
                              context=context(W, i0, e_), why=f"trails off after {breath(s_, e_):.1f} s without a breath")))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_shout(ctx, code, level, rng, zone, n):
    """Vocal effort, not a volume knob: louder + brighter (high shelf) + higher (F0 +0.25 st/dB), noise floor untouched.
    Starts at a phrase onset; at L1-L2 it covers just the most prominent content word (over-emphasis), from L3 the whole phrase."""
    c = CFG["flaws"][code]
    g_db = c["levels"][level - 1]
    W = ctx.words
    edits = []
    for i0, i1 in pick_windows(ctx, rng, n, *c["region_s"], zone):
        if level <= 2:
            content = [j for j in range(i0, i1 + 1) if ctx.ling["w"][j]["content"] and (W[j]["end"] - W[j]["start"]) > s2n(0.12)]
            if content:
                best = max(content, key=lambda j: W[j]["end"] - W[j]["start"])
                j0 = j1 = best
            else:
                j0, j1 = i0, i1
        else:
            j0, j1 = i0, i1
        a, b = ctx.span(j0, j1)
        y = shout_bundle(ctx, a, ctx.x[a:b], g_db, ctx.pure)
        lvl_b, lvl_a = db(rms(ctx.x[a:b])), db(rms(y))
        edits.append(Edit(a, b, y, lab(code, level, {"gain_db": g_db, "bundle": not ctx.pure,
                                                    "evidence": {"level_gain_db": round(lvl_a - lvl_b, 2)}}, j0, j1,
                                      context=context(W, j0, j1),
                                      why="over-emphasis on one word" if level <= 2 else "sudden vocal effort from a phrase onset")))
    return edits


def inj_filler(ctx, code, level, rng, zone, n):
    """Hesitations where people really hesitate: after full stops and commas, before and/or/but and clause openers, after
    discourse markers, and (as 'the... um... people') after a reducible function word. Never inside a content-word compound.
    Realised as a drawl of the previous function word, or an own-voice 'uh'/'um' (voiced vowel, local pitch and level)."""
    from wordfreq import zipf_frequency
    per30 = CFG["flaws"][code]["levels"][level - 1]
    count = max(1, round(per30 * ((zone[1] - zone[0]) / SR) / 30))
    W, Lg = ctx.words, ctx.ling
    ks = boundaries(ctx, zone, lambda k: True)
    wts = []
    for k in ks:
        bt = Lg["b"][k]
        w = HESITATION_WEIGHT[bt]
        if bt == "phrase_tight":
            w = 0.25 if (Lg["w"][k]["reducible"] and len(W[k]["clean"]) <= 3) else 0.0      # 'the ... um ... people'
        nxt = W[k + 1]["clean"]
        if Lg["w"][k + 1]["content"] and len(nxt) > 5 and zipf_frequency(nxt, "en") < 4.2:
            w += 1.2                                                                         # searching for a hard word
        wts.append(w)
    edits = []
    for k in weighted_order(rng, ks, wts):
        k = int(k)
        bt = Lg["b"][k]
        a = (W[k]["end"] + W[k + 1]["start"]) // 2
        token = "um" if bt in ("sentence", "clause_punct", "before_coord", "before_subord") and ctx.nrng.random() < 0.55 else "uh"
        can_drag = Lg["w"][k]["reducible"] and not W[k]["punct"] and len(W[k]["clean"]) <= 4
        if can_drag and ctx.nrng.random() < 0.4:                      # drawl the function word, then a short beat of own room tone
            dr = drag_word(ctx, k, 2.2 + 0.4 * (level - 1))
            if dr is not None and ctx.claim(W[k]["start"], W[k]["start"] + s2n(1.0), 0.4):
                edits.append(Edit(W[k]["start"], W[k]["end"], np.concatenate([dr, ctx.tone(s2n(0.10))]),
                                  lab(code, level, {"per_30s": per30, "token": "drawl", "filler_source": "own-voice-drawl"}, k, k,
                                      "modify", **explain(ctx, k))))
                if len(edits) == count:
                    break
                continue
        f, src = own_filler(ctx, k, token)
        if f is None:
            continue
        seg = np.concatenate([ctx.tone(s2n(0.05)), f, ctx.tone(s2n(0.07))])
        if ctx.claim(a, a + s2n(1.0), 0.4):
            edits.append(Edit(a, a, seg, lab(code, level, {"per_30s": per30, "token": token, "filler_source": src}, k, k + 1,
                                             "insert", inserted_s=round(len(seg) / SR, 3), **explain(ctx, k))))
            if len(edits) == count:
                break
    return sorted(edits, key=lambda e: e.a)


def inj_repeat(ctx, code, level, rng, zone, n):
    """False start at a clause opener. v1.1 redesign: ONE anchor per take, a clause opener where a 4-word restart fits, used by every
    level (L1 'the the', L2 'the the the', L3 2-word restart, L4 4-word restart, L5 4-word restart twice). The first attempt is cut off at
    60-80% of its length, +0.5-1 st higher and -1 dB, then the clean restart follows: never a sample-identical copy (pure=True keeps the old
    exact copy for ablation)."""
    nw, copies = CFG["flaws"][code]["levels"][level - 1]
    W, Lg = ctx.words, ctx.ling
    opener = ("sentence", "clause_punct", "before_coord", "before_subord", "before_relative", "after_discourse")
    cands, wts = [], []
    for k in range(len(W) - 5):
        i0, i1 = k + 1, k + 4
        if Lg["b"][k] not in opener or W[i0]["start"] < zone[0] or W[i1]["end"] > zone[1]:
            continue
        if any(W[j]["punct"] for j in range(i0, i1)) or not W[i0]["clean"] or not Lg["w"][i0]["function"]:
            continue
        cands.append(i0)
        wts.append(2.5 if Lg["w"][i0]["pos"] in ("DET", "PRON", "CCONJ", "ADP") else 1.5)
    edits = []
    for i0 in weighted_order(rng, cands, wts):
        i0 = int(i0)
        i1 = i0 + nw - 1
        a0, a1 = W[i0]["start"], W[i1]["end"]
        if not ctx.claim(a0, W[i0 + 3]["end"], 0.3):
            continue
        parts, fracs = [], []
        for _ in range(copies):
            frac, up = 0.6 + 0.2 * float(ctx.nrng.random()), 0.5 + 0.5 * float(ctx.nrng.random())
            piece = ctx.x[a0:a1].copy() if ctx.pure else partial_attempt(ctx, a0, a1, frac, up, -1.0)
            fracs.append(round(frac, 2))
            parts += [piece, ctx.tone(s2n(0.06 + 0.06 * float(ctx.nrng.random())))]
        seg = np.concatenate(parts)
        edits.append(Edit(a0, a0, seg, lab(code, level, {"words": nw, "restarts": copies, "bundle": not ctx.pure,
                                                       "evidence": {"cut_fracs": fracs, "inserted_s": round(len(seg) / SR, 2),
                                                                                   "restart_words": round(nw * float(np.sum(fracs)), 2)}},
                                          i0, i1, "insert", inserted_s=round(len(seg) / SR, 3), boundary=Lg["b"][i0 - 1],
                                          context=context(W, i0, i1),
                                          why=f"false start at a clause opener: '{' '.join(w['w'] for w in W[i0:i1 + 1])}' attempted, cut off, restarted")))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_rare_hesit(ctx, code, level, rng, zone, n):
    """Hesitation before a rare word: optional drawl of the function word before it, a pause of the speaker's own room tone,
    an own-voice 'uh' from L3, and the rare word itself slowed (PSOLA)."""
    from wordfreq import zipf_frequency
    from psola import psola_stretch
    c = CFG["flaws"][code]
    pause, slow = c["levels"][level - 1], c["word_slow"][level - 1]
    W, Lg = ctx.words, ctx.ling
    med = float(np.exp(ctx.f0_median_log))
    cands = [k for k in range(2, len(W) - 2) if W[k]["start"] >= zone[0] and W[k]["end"] <= zone[1]
             and len(W[k]["clean"]) > 4 and zipf_frequency(W[k]["clean"], "en") < c["zipf_below"]]
    edits = []
    for k in rng.permutation(cands):
        k = int(k)
        a, b = W[k]["start"], W[k]["end"]
        if not ctx.claim(a - s2n(0.3), b + s2n(0.1), 0.2):
            continue
        parts = [ctx.tone(s2n(pause))]
        src = None
        if level >= c["filler_from_level"]:
            f, src = own_filler(ctx, k - 1, "uh")
            if f is not None:
                parts = [ctx.tone(s2n(pause * 0.6)), f, ctx.tone(s2n(pause * 0.4))]
        seg = np.concatenate(parts + [psola_stretch(ctx.x[a:b], 1.0 / slow, med)])
        pv = W[k - 1]
        if Lg["w"][k - 1]["reducible"] and not pv["punct"] and level >= 2 and pv["end"] <= a:
            dr = drag_word(ctx, k - 1, 1.8 + 0.3 * level)
            if dr is not None:
                edits.append(Edit(pv["start"], pv["end"], dr, lab(code, level, {"token": "drawl"}, k - 1, k - 1, "modify",
                                                                  why=f"drawl on '{pv['w']}' before the hard word")))
        edits.append(Edit(a, b, seg, lab(code, level, {"pause_s": pause, "word_speed": slow, "filler": src, "word": W[k]["clean"],
                                                      "zipf": round(zipf_frequency(W[k]["clean"], "en"), 2)}, k, k,
                                         context=context(W, k, k), boundary=Lg["b"][k - 1],
                                         why=f"hesitation before the rare word '{W[k]['clean']}'")))
        if sum(1 for e in edits if e.label["flaw"] == "RARE_HESIT" and e.label["params"].get("word")) >= n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_slur(ctx, code, level, rng, zone, n):
    """Slur where articulation is crispest (strongest 2-8 kHz consonant energy), so the loss is real, and record the measured drop."""
    from scipy.signal import istft, stft
    c = CFG["flaws"][code]
    att = c["levels"][level - 1]
    cands = windows(ctx, *c["region_s"], zone, True) or windows(ctx, 1.0, 4.5, zone, False)
    scored = sorted(cands, key=lambda w: -(ctx.hf_stat(*ctx.span(*w)) + 1.5 * float(rng.random())))
    edits = []
    for i0, i1 in scored:
        a, b = ctx.span(i0, i1)
        if not ctx.claim(a, b):
            continue
        f, t, Z = stft(ctx.x[a:b].astype(np.float64), SR, nperseg=512, noverlap=384)
        mag, ph = np.abs(Z), np.angle(Z)
        band = (f >= 2000) & (f <= 8000)
        edge = np.clip(np.minimum(f - 1700, 8300 - f) / 300, 0, 1)
        g = 1 + (undb(att) - 1) * edge
        tt = np.ones(len(t))
        m = min(8, len(t) // 4)
        tt[:m] = np.linspace(0, 1, m)
        tt[-m:] = np.linspace(1, 0, m)
        gain = 1 + (g[:, None] - 1) * tt[None, :]
        mag2 = mag * gain
        k = 1 + 2 * min(3, level)
        sm = np.apply_along_axis(lambda r: np.convolve(r, np.ones(k) / k, mode="same"), 1, mag2[band])
        w = 0.15 * level * tt
        mag2[band] = (1 - w) * mag2[band] + w * sm
        _, y = istft(mag2 * np.exp(1j * ph), SR, nperseg=512, noverlap=384)
        y = np.pad(y, (0, max(0, b - a - len(y))))[: b - a].astype(np.float32)
        before, after = ctx.hf_stat(a, b), ctx.hf_stat(0, len(y), y)
        edits.append(Edit(a, b, y, lab(code, level, {"atten_db": att, "evidence": {"consonant_db_before": round(before, 1), "consonant_db_after": round(after, 1),
                                                                                 "drop_db": round(before - after, 1)}}, i0, i1,
                                      context=context(ctx.words, i0, i1), why=f"slurred the crispest-articulated stretch ({before:.0f} dB consonant energy)")))
        if len(edits) == n:
            break
    return sorted(edits, key=lambda e: e.a)


def inj_word_skip(ctx, code, level, rng, zone, n):
    """Skip a word the way readers do: small function words (the/of/to/is), never the first or last word of a sentence."""
    count = CFG["flaws"][code]["levels"][level - 1]
    W, Lg = ctx.words, ctx.ling
    wpos = {"DET": 3.0, "ADP": 2.5, "AUX": 1.5, "CCONJ": 1.0, "PRON": 0.7, "PART": 1.0}
    ks, wts = [], []
    for k in range(2, len(W) - 2):
        if W[k]["start"] < zone[0] or W[k]["end"] > zone[1] or W[k]["punct"] or W[k - 1]["punct"] and W[k - 1]["sent_end"]:
            continue
        if W[k + 1]["punct"] and W[k + 1]["sent_end"] and False:
            continue
        pos = Lg["w"][k]["pos"]
        if pos in wpos and (W[k]["end"] - W[k]["start"]) > s2n(0.06) and len(W[k]["clean"]) > 1 and not W[k - 1]["sent_end"]:
            ks.append(k)
            wts.append(wpos[pos])
    edits = []
    for k in weighted_order(rng, ks, wts):
        k = int(k)
        a, b = W[k]["start"], W[k]["end"]
        if ctx.claim(a, b, 0.25):
            edits.append(Edit(a, b, np.zeros(0, np.float32),
                              lab(code, level, {"n_words": count}, k, k, "delete", word=W[k]["clean"], pos=Lg["w"][k]["pos"],
                                  context=context(W, k, k), why=f"reader skips the {Lg['w'][k]['pos'].lower()} '{W[k]['clean']}'")))
            if len(edits) == count:
                break
    return sorted(edits, key=lambda e: e.a)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _drop_final_s(ctx, a: int, b: int):
    """Morphological misread: the final plural/3rd-person -s is not said. Cut at the start of the word's closing unvoiced sibilant."""
    from scipy.signal import butter, sosfilt
    seg = ctx.x[a:b]
    h = s2n(0.005)
    hf = sosfilt(butter(4, 3000, btype="high", fs=SR, output="sos"), seg)
    for i in range(len(seg) - h, s2n(0.05), -h):                  # walk back while the tail is HF-dominated
        e_hf, e_all = rms(hf[i:i + h]), rms(seg[i:i + h]) + 1e-9
        if e_hf / e_all < 0.55:
            cut = i + h
            break
    else:
        return None
    if len(seg) - cut < s2n(0.045) or cut < s2n(0.06):
        return None
    out = seg[:cut].copy()
    n = min(s2n(0.01), len(out) // 4)
    out[-n:] *= np.linspace(1, 0, n).astype(np.float32)
    return out


def inj_word_swap(ctx, code, level, rng, zone, n):
    """A believable misread, not a random word: (1) a word that LOOKS like the target (same first letter, edit distance <= 2, 'form'/'from',
    'thing'/'things') taken from the same speech, (2) a dropped plural -s, (3) fallback: another word of the same part of speech. The donor is
    re-timed and pitch-matched (median AND slope) with PSOLA, preferring a similar position in its sentence."""
    from psola import psola_match, psola_pitch
    import parselmouth
    count = CFG["flaws"][code]["levels"][level - 1]
    W, Lg = ctx.words, ctx.ling
    med = float(np.exp(ctx.f0_median_log))
    sent_of = {}
    for s_, e_ in ctx.sentences():
        for i in range(s_, e_ + 1):
            sent_of[i] = (s_, e_)
    relpos = lambda i: (i - sent_of[i][0]) / max(1, sent_of[i][1] - sent_of[i][0])

    def f0_slope(seg):
        p = parselmouth.Sound(seg.astype(np.float64), SR).to_pitch(0.005, max(60.0, med * 0.55), min(600.0, med * 2.4))
        f, t = p.selected_array["frequency"], p.xs()
        v = f > 0
        if v.sum() < 4:
            return med, 0.0
        st = 12 * np.log2(f[v])
        return float(np.median(f[v])), float(np.polyfit(t[v], st, 1)[0])

    ks = boundaries(ctx, zone, lambda k: not W[k]["punct"] and (W[k]["end"] - W[k]["start"]) > s2n(0.12)
                    and len(W[k]["clean"]) > 2 and Lg["w"][k]["content"])
    edits = []
    for k in rng.permutation(ks):
        k = int(k)
        a, b = W[k]["start"], W[k]["end"]
        dk = b - a
        cl = W[k]["clean"]
        kind, donor_i, seg = None, None, None
        sim = [j for j in range(len(W)) if abs(j - k) > 3 and not W[j]["punct"] and W[j]["clean"] != cl and len(W[j]["clean"]) > 2
               and W[j]["clean"][0] == cl[0] and abs(len(W[j]["clean"]) - len(cl)) <= 2 and _lev(W[j]["clean"], cl) <= 2
               and 0.6 * dk <= W[j]["end"] - W[j]["start"] <= 1.6 * dk]
        if sim:
            donor_i = min(sim, key=lambda j: (_lev(W[j]["clean"], cl), abs(relpos(j) - relpos(k))))
            kind = "similar-word"
        elif cl.endswith("s") and len(cl) > 3 and not cl.endswith("ss"):
            seg = _drop_final_s(ctx, a, b)
            kind = "dropped-s" if seg is not None else None
        if kind is None:
            same = [j for j in range(len(W)) if abs(j - k) > 3 and W[j]["clean"] != cl and not W[j]["punct"] and Lg["w"][j]["pos"] == Lg["w"][k]["pos"]
                    and Lg["w"][j]["lemma"] != Lg["w"][k]["lemma"] and 0.8 * dk <= W[j]["end"] - W[j]["start"] <= 1.25 * dk]
            if same:
                donor_i = min(same, key=lambda j: abs(relpos(j) - relpos(k)))
                kind = "same-pos"
        if kind is None or not ctx.claim(a, b, 0.25):
            continue
        if donor_i is not None:
            donor = ctx.x[W[donor_i]["start"]: W[donor_i]["end"]]
            ft, sl_t = f0_slope(ctx.x[a:b])
            fd, sl_d = f0_slope(donor)
            seg = psola_match(donor, dk, ft / fd, med)
            if abs(sl_t - sl_d) > 0.5 and len(seg) > s2n(0.08):
                seg = psola_pitch(seg, lambda f, t: f * 2 ** ((sl_t - sl_d) * (t - t.mean()) / 12), ft)
            seg = seg * (rms(ctx.x[a:b]) / (rms(seg) + 1e-9))
            seg = seg.astype(np.float32).copy()
            nf = min(s2n(0.008), len(seg) // 4)
            seg[:nf] *= np.linspace(0, 1, nf)
            seg[-nf:] *= np.linspace(1, 0, nf)
            rep = W[donor_i]["clean"]
        else:
            rep = cl[:-1]
        edits.append(Edit(a, b, seg, lab(code, level, {"n_words": count, "swap_kind": kind}, k, k, original=cl, replaced_with=rep,
                                         pos=Lg["w"][k]["pos"], context=context(W, k, k),
                                         why=f"misread '{cl}' as '{rep}' ({kind.replace('-', ' ')})")))
        if len(edits) == count:
            break
    return sorted(edits, key=lambda e: e.a)


INJECTORS = {
    "PACE_FAST": inj_pace, "PACE_SLOW": inj_pace, "PAUSE_BAD": inj_pause_bad, "PAUSE_LOST": inj_pause_lost,
    "MONOTONE": inj_monotone, "UPTALK": inj_uptalk, "EMPH_FLAT": inj_emph_flat, "FADE": inj_fade,
    "SHOUT": inj_shout, "FILLER": inj_filler, "REPEAT": inj_repeat, "RARE_HESIT": inj_rare_hesit,
    "SLUR": inj_slur, "WORD_SKIP": inj_word_skip, "WORD_SWAP": inj_word_swap,
}
FLAW_CODES = list(INJECTORS)


# --------------------------------------------------------------------------- assembly
def assemble(x: np.ndarray, edits: list[Edit]) -> tuple[np.ndarray, list[dict]]:
    edits = sorted(edits, key=lambda e: (e.a, e.b))
    out = np.zeros(0, np.float32)
    cursor, regions = 0, []
    for e in edits:
        assert e.a >= cursor, "overlapping edits"
        out = crossfade_concat([out, x[cursor:e.a]], FADE, True) if len(out) else x[cursor:e.a].copy()
        if len(e.seg):
            start = max(len(out) - FADE, 0)
            out = crossfade_concat([out, e.seg], FADE, True)
            end = len(out) - FADE
        else:                                          # deletion: the join is made by the NEXT crossfade, centred FADE/2 before here
            start = end = max(len(out) - FADE // 2, 0)
        regions.append({**e.label,
                        "start_s": round((start - e.pre) / SR, 4), "end_s": round((end + e.post) / SR, 4),
                        "baseline_start_s": round((e.a - e.pre) / SR, 4), "baseline_end_s": round((e.b + e.post) / SR, 4)})
        cursor = e.b
    out = crossfade_concat([out, x[cursor:]], FADE, True) if len(out) else x[cursor:].copy()
    return soft_limit(out.astype(np.float32)), regions


def seed_for(take_id: str, tag: str) -> int:
    """Per take+flaw, NOT per level: the five levels of a flaw hit the same regions, so the only thing that
    changes between L1 and L5 is severity (clean dose-response)."""
    return zlib.crc32(f"{take_id}|{tag}".encode()) % 10000


GATE_MAX = 12.0        # ... and no single join above this
GATE_DB = 6.0          # v1.1: a join must not click more than this over what the baseline does naturally
MAX_ATTEMPTS = 8
_ANCHOR: dict = {}


def _code_hash() -> str:
    """Anchors are only valid for the exact generator code that chose them."""
    import hashlib
    h = hashlib.sha256()
    for f in ("flaws.py", "render.py", "psola.py", "common.py", "linguistics.py", "config.yaml"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def _anchor_file(key) -> Path:
    import hashlib
    d = Path(__file__).parent / ".cache" / "anchors"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(repr(key).encode()).hexdigest()[:20] + "_" + _code_hash() + ".json")


def _with_rotation(ctx: Ctx, flaws: list[tuple[str, int]], seed: int):
    err = None
    for rot in range(len(flaws)):
        try:
            return _make_flawed(ctx, flaws[rot:] + flaws[:rot], seed)
        except RuntimeError as e:
            err = e
    raise err


def make_flawed(ctx: Ctx, flaws: list[tuple[str, int]], seed: int, tag: str | None = None, pure: bool = False):
    """flaws: [(code, level)]. One flaw: whole clip is the zone. Several: one equal time zone each.
    ANCHOR: the placement attempt is chosen ONCE per (take, flaws) at the most demanding level and reused for every level, so severity is the
    only thing that changes between L1 and L5. GATE: an attempt is accepted only if no join clicks more than GATE_DB over the baseline;
    otherwise the next candidate placement is tried (best attempt kept if none passes, and flagged in ctx.last_join)."""
    ctx.pure = pure
    key = (ctx.take_id, tuple(c for c, _ in flaws), pure, tuple(l for _, l in flaws) if len(flaws) > 1 else ())
    if key not in _ANCHOR:
        af = _anchor_file(key)
        if af.exists():
            _ANCHOR[key] = tuple(json.loads(af.read_text()))
    if key not in _ANCHOR:
        probes = [[(c, lv) for c, _ in flaws] for lv in (1, 5)] if len(flaws) == 1 else [flaws]
        best = None
        for att in range(MAX_ATTEMPTS):
            try:
                cks = []
                for pr in probes:
                    y, regs = _with_rotation(ctx, pr, seed + 7919 * att)
                    cks += join_clicks(ctx.x, y, regs)
            except RuntimeError:
                continue
            share = float(np.mean([c_ > GATE_DB for c_ in cks])) if cks else 0.0
            worst = max(cks, default=0.0)
            score = (share > 0.05 or worst > GATE_MAX, share, worst)
            if best is None or score < best[0]:
                best = (score, att)
            if not score[0]:
                break
        if best is None:
            raise RuntimeError(f"{ctx.take_id}: could not place {[c for c, _ in flaws]}")
        _ANCHOR[key] = (best[0][2], best[1])
        _anchor_file(key).write_text(json.dumps(list(_ANCHOR[key])))
    att = _ANCHOR[key][1]
    y, regs = _with_rotation(ctx, flaws, seed + 7919 * att)
    ck = join_clicks(ctx.x, y, regs)
    share = float(np.mean([c_ > GATE_DB for c_ in ck])) if ck else 0.0
    ctx.last_join = {"max_click_db": round(max(ck, default=0.0), 1), "n_joins": len(ck), "share_above_6db": round(share, 3),
                     "passed": bool(share <= 0.05 and max(ck, default=0.0) <= GATE_MAX), "attempt": att}
    return y, regs


def _make_flawed(ctx: Ctx, flaws: list[tuple[str, int]], seed: int):
    rng = np.random.default_rng(seed)           # placement: same regions at every level of a flaw
    ctx.nrng = np.random.default_rng((seed % 100000) * 10 + 1)    # content choices (tokens, tone draws): level-independent too
    ctx.taken = []
    lo, hi = s2n(0.8), len(ctx.x) - s2n(0.8)
    edits: list[Edit] = []
    for i, (code, level) in enumerate(flaws):
        zone = (lo, hi) if len(flaws) == 1 else (lo + i * (hi - lo) // len(flaws), lo + (i + 1) * (hi - lo) // len(flaws))
        n = n_regions(rng) if len(flaws) == 1 else 1
        got = INJECTORS[code](ctx, code, level, rng, zone, n)
        if not got:
            raise RuntimeError(f"{ctx.take_id}: could not place {code} L{level}")
        edits += got
    return assemble(ctx.x, edits)
