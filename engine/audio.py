"""Ingest + quality gate. 16 kHz mono, level-normalised. Independent of the generator."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
TARGET_DBFS = -23.0           # speech level target (RMS of active frames), a LUFS-style normalisation


def load(path: str) -> np.ndarray:
    """Any format ffmpeg reads -> 16 kHz mono float32."""
    p = Path(path)
    try:
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ac", "1", "-ar", str(SR), str(wav)], check=True)
            x, sr = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def frame_db(x: np.ndarray, win: float = 0.025, hop: float = 0.010) -> np.ndarray:
    w, h = int(win * SR), int(hop * SR)
    n = max(1, 1 + (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    fr = x[np.minimum(idx, len(x) - 1)]
    return 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)


def normalise(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale so the active-speech RMS sits at TARGET_DBFS. Returns (audio, applied gain dB)."""
    d = frame_db(x)
    act = d > np.percentile(d, 95) - 25
    cur = 10 * np.log10(np.mean(10 ** (d[act] / 10)) + 1e-12) if act.any() else -40.0
    g = TARGET_DBFS - cur
    return (x * 10 ** (g / 20)).astype(np.float32), float(g)


def quality(x: np.ndarray) -> dict:
    """SNR from pause frames, clipped fraction, bandwidth (95% energy), crude reverb tail; plus a 0-1 confidence."""
    d = frame_db(x)
    floor = float(np.percentile(d, 8))
    speech = float(np.percentile(d, 90))
    snr = speech - floor
    clipped = float(np.mean(np.abs(x) > 0.99))
    spec = np.abs(np.fft.rfft(x[: SR * 30] * np.hanning(len(x[: SR * 30])))) ** 2
    cum = np.cumsum(spec) / (spec.sum() + 1e-12)
    bw = float(np.searchsorted(cum, 0.995) / len(spec) * (SR / 2))      # frequency below which 99.5% of the energy sits
    # reverb: how slowly level decays after loud frames (median fall time, frames) -> 0..1
    tail = float(np.mean(np.diff(d)[np.diff(d) < -0.5]) * -1) if (np.diff(d) < -0.5).any() else 3.0
    conf = float(np.clip((snr - 8) / 22, 0, 1) * (1 - min(1, clipped * 200)) * np.clip(bw / 3000, 0.3, 1.0))
    badge = "good" if conf > 0.7 else "fair" if conf > 0.4 else "poor"
    return {"snr_db": round(snr, 1), "clipped_frac": round(clipped, 5), "bandwidth_hz": round(bw), "decay_db_per_frame": round(tail, 2),
            "confidence": round(conf, 2), "badge": badge}
