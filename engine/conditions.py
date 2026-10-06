"""Recording-condition matching ("Where" is measured, never scored). Compare like with like: estimate the participant's bandwidth, noise
(level and colour, from its own pauses) and reverb, and apply the same degradation to the clean reference BEFORE comparing. A noisy phone
recording is then compared with a noisy, band-limited reference, so noise/phone/room cannot masquerade as delivery flaws."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .audio import SR, frame_db


def bandwidth(x: np.ndarray, frac: float = 0.995, win_s: float = 2.0) -> float:
    """Recording bandwidth: the 75th percentile of the per-window (2 s) bandwidths of the speech windows. A phone line or codec limits EVERY
    window, while a local slur (a few clauses with the top band dulled) limits a minority of them, so the recording's bandwidth is not
    mistaken for a delivery flaw and the slur survives condition matching."""
    seg = x[: SR * 60]
    w = int(win_s * SR)
    if len(seg) < 2 * w:
        w = len(seg)
    d = frame_db(seg, 0.025, 0.010)
    thr = np.percentile(d, 95) - 25
    out = []
    for a in range(0, max(1, len(seg) - w + 1), w // 2):
        part = seg[a:a + w]
        if np.mean(d[a // 160: (a + w) // 160] > thr) < 0.4:               # mostly pause: no band information
            continue
        spec = np.abs(np.fft.rfft(part * np.hanning(len(part)))) ** 2
        cum = np.cumsum(spec) / (spec.sum() + 1e-12)
        out.append(np.searchsorted(cum, frac) / len(spec) * (SR / 2))
    return float(np.percentile(out, 75)) if out else 0.0


def low_cut(x: np.ndarray) -> float:
    """Frequency below which only 0.3% of the energy lies (phone lines lose the lows)."""
    seg = x[: SR * 40]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    cum = np.cumsum(spec) / (spec.sum() + 1e-12)
    return float(np.searchsorted(cum, 0.003) / len(spec) * (SR / 2))


def quiet_pool(x: np.ndarray, frac: float = 0.12) -> np.ndarray:
    """Concatenated quietest frames of the clip: its own noise, in its own colour."""
    d = frame_db(x, 0.025, 0.010)
    thr = np.percentile(d, frac * 100)
    h, w = int(0.010 * SR), int(0.025 * SR)
    idx = np.where(d <= thr)[0]
    return np.concatenate([x[i * h: i * h + w] for i in idx]) if len(idx) else x[: SR // 4]


def decay_stat(x: np.ndarray) -> float:
    """Median level drop (dB) over the 120 ms after speech offsets: small drop = long reverberant tail."""
    d = frame_db(x, 0.025, 0.010)
    thr = np.percentile(d, 95) - 18
    drops = []
    for i in range(1, len(d) - 16):
        if d[i - 1] > thr and d[i] <= thr:                      # a speech offset
            drops.append(d[i - 1] - np.min(d[i: i + 14]))
    return float(np.median(drops)) if drops else 30.0


def reverb_ir(rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = int(rt60 * 1.2 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.008 * SR)] = 0
    tail = tail / np.sqrt(np.sum(tail ** 2)) * 0.6
    ir = np.zeros(n)
    ir[0] = 1.0
    return ir + tail


def match(ref: np.ndarray, par: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (reference degraded to the participant's conditions, what was applied)."""
    info = {}
    out = ref.astype(np.float64)
    bw_p, bw_r = bandwidth(par), bandwidth(ref)
    lc_p, lc_r = low_cut(par), low_cut(ref)
    if bw_p < 0.8 * bw_r and bw_p < 6500:                       # band-limited (phone / codec): same band for the reference
        hi = max(2500.0, bw_p * 1.02)
        lo = lc_p if lc_p > 150 else 0
        sos = butter(6, [max(lo, 60.0), min(hi, SR / 2 - 100)], btype="band", fs=SR, output="sos") if lo else butter(6, min(hi, SR / 2 - 100), btype="low", fs=SR, output="sos")
        out = sosfilt(sos, out)
        info["bandlimit_hz"] = [round(lo), round(hi)]
    # noise first (the participant's own quiet frames, tiled: same level and colour), THEN choose the reverb that explains the remaining decay
    # difference ('none' is a candidate), because noise alone also shortens the apparent decay and must not be mistaken for a room
    pool = quiet_pool(par)
    dp = frame_db(par)
    floor_p = 10 ** (np.percentile(dp, 8) / 10)
    floor_r = 10 ** (np.percentile(frame_db(out.astype(np.float32)), 8) / 10)
    tile = None
    if floor_p > 2.0 * floor_r:                                # participant is noisier than the reference
        tile = np.tile(pool, int(np.ceil(len(out) / len(pool))))[: len(out)].astype(np.float64)
        tile *= np.sqrt(max(floor_p - floor_r, 0.0) / (np.mean(tile ** 2) + 1e-12))
        info["noise_added_db_re_speech"] = round(float(10 * np.log10(floor_p)), 1)
    ds_p = decay_stat(par)
    base_rms = np.sqrt(np.mean(out ** 2))
    cand = {}
    for rt in (0.0, 0.3, 0.45, 0.6, 0.8, 1.0):
        y = out if rt == 0.0 else fftconvolve(out, reverb_ir(rt, np.random.default_rng(7)))[: len(out)]
        y = y * (base_rms / (np.sqrt(np.mean(y ** 2)) + 1e-12))
        y = y + tile if tile is not None else y
        cand[rt] = (abs(decay_stat(y.astype(np.float32)) - ds_p), y)
    rt_best = min(cand, key=lambda r: (cand[r][0] - (0.25 if r == 0.0 else 0.0)))      # mild preference for 'no reverb'
    out = cand[rt_best][1]
    if rt_best:
        info["reverb_rt60"] = rt_best
    return out.astype(np.float32), info
