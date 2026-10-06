"""Frame and word features, speaker-normalised (semitones vs the speaker's own median, dB vs the speaker's own speech level)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
from scipy.signal import medfilt, stft

from .audio import SR, frame_db

HOP = 0.010


@dataclass
class Frames:
    t: np.ndarray            # frame centre times (s)
    f0_st: np.ndarray        # semitones re speaker median, NaN where unvoiced
    inten: np.ndarray        # dB re speaker median active level
    hf: np.ndarray           # 2-7.5 kHz energy dB re total (spectral tilt proxy)
    voiced: np.ndarray       # bool
    f0_median_hz: float
    level_median_db: float

    def sl(self, a: float, b: float) -> slice:
        return slice(max(0, int(a / HOP)), max(1, int(b / HOP)))


def compute(x: np.ndarray) -> Frames:
    snd = parselmouth.Sound(x.astype(np.float64), SR)
    p0 = snd.to_pitch(HOP, 70.0, 500.0)
    f = p0.selected_array["frequency"]
    med = float(np.median(f[f > 0])) if (f > 0).any() else 150.0
    p = snd.to_pitch(HOP, max(60.0, med * 0.55), min(600.0, med * 2.4))
    f = p.selected_array["frequency"].astype(float)
    t = p.xs()
    v = f > 0
    # octave/glitch cleanup
    if v.sum() > 7:
        ref = medfilt(np.where(v, f, med), 9)
        bad = v & (np.abs(f / np.maximum(ref, 1.0) - 1) > 0.2)
        f[bad] = ref[bad]
    st = np.where(v, 12 * np.log2(np.maximum(f, 1.0) / med), np.nan)
    d = frame_db(x)
    n = min(len(d), len(t))
    d, t, st, v = d[:n], t[:n], st[:n], v[:n]
    act = d > np.percentile(d, 95) - 25
    lvl = float(np.median(d[act])) if act.any() else float(np.median(d))
    fr, tt, Z = stft(x.astype(np.float64), SR, nperseg=400, noverlap=240, boundary=None)
    P = np.abs(Z) ** 2
    band = (fr >= 2000) & (fr <= 7500)
    hf_db = 10 * np.log10(P[band].sum(axis=0) + 1e-12) - 10 * np.log10(P.sum(axis=0) + 1e-12)
    hf = np.interp(t, tt, hf_db)
    return Frames(t=t, f0_st=st, inten=d - lvl, hf=hf, voiced=v, f0_median_hz=med, level_median_db=lvl)


def word_feats(fr: Frames, words: list[dict]) -> list[dict]:
    """Per word: duration, F0 peak/range (st), mean intensity (dB), hf, voiced fraction; plus a clip-level prominence z-score
    (F0 peak + energy + duration, the same recipe for every speaker)."""
    out = []
    for w in words:
        s = fr.sl(w["start"], w["end"])
        v = fr.f0_st[s]
        v = v[~np.isnan(v)]
        pk = float(np.percentile(v, 90)) if len(v) >= 4 else np.nan
        out.append({"dur": w["end"] - w["start"], "f0_peak": pk, "f0_range": float(np.ptp(v)) if len(v) >= 4 else np.nan,
                    "inten": float(np.mean(fr.inten[s])) if fr.inten[s].size else np.nan,
                    "hf": float(np.mean(fr.hf[s])) if fr.hf[s].size else np.nan,
                    "voiced": float(np.mean(fr.voiced[s])) if fr.voiced[s].size else 0.0})
    def z(key):
        a = np.array([o[key] for o in out], float)
        a = np.where(np.isnan(a), np.nanmean(a) if np.isfinite(a).any() else 0.0, a)
        return (a - a.mean()) / (a.std() + 1e-9)
    prom = z("f0_peak") + z("inten") + z("dur")
    for o, p in zip(out, prom):
        o["prom"] = float(p)
    return out
