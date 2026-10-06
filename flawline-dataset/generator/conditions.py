"""Synthetic recording conditions ("Where"): N20 N10 RVB PHN MP3 GAIN. All preserve timing (labels stay valid)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from common import SR, TAKES, db, frame_rms_db, load_config, pink_noise, read_audio, rms, soft_limit, undb

CFG = load_config()["conditions"]


def speech_rms(x: np.ndarray) -> float:
    fr = frame_rms_db(x)
    thr = np.percentile(fr, 95) - 25
    hop, win = 220, 441
    idx = [i for i, v in enumerate(fr) if v > thr]
    return rms(np.concatenate([x[i * hop: i * hop + win] for i in idx]))


def babble(n: int, exclude_take: str, rng) -> np.ndarray:
    """Four other talkers from the baseline pool, TIME-REVERSED, random offsets, summed. The baselines share their text, so forward babble would put
    the participant's own words into the noise; reversed speech keeps the multi-talker spectrum and modulation but carries no words."""
    pool = sorted(p for p in TAKES.glob("*_C0.flac") if not p.name.startswith(exclude_take))
    out = np.zeros(n, np.float32)
    for k in rng.permutation(len(pool))[:4]:
        y, _ = read_audio(pool[k])
        y = np.tile(y[::-1].copy(), int(np.ceil(n / len(y)) + 1))
        o = int(rng.integers(0, len(y) - n))
        out += y[o:o + n] / (rms(y[o:o + n]) + 1e-9)
    return out


def add_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise / (rms(noise) + 1e-9) * speech_rms(x) * undb(-snr_db)
    return x + noise.astype(np.float32)


def align_to(ref: np.ndarray, y: np.ndarray, max_lag: int = 4000) -> np.ndarray:
    """Remove codec delay so region labels stay valid."""
    n = min(len(ref), len(y), SR * 20)
    c = fftconvolve(y[:n], ref[:n][::-1], mode="full")
    lag = int(np.argmax(c[n - 1 - max_lag: n - 1 + max_lag])) - max_lag
    y = y[lag:] if lag > 0 else np.pad(y, (-lag, 0))
    return np.pad(y, (0, max(0, len(ref) - len(y))))[: len(ref)]


def apply(x: np.ndarray, code: str, rng, take_id: str = "") -> tuple[np.ndarray, dict]:
    c = CFG[code]
    k = c["kind"]
    if k == "babble":
        y, p = add_at_snr(x, babble(len(x), take_id, rng), c["snr_db"]), {"snr_db": c["snr_db"], "noise": "babble"}
    elif k == "pink":
        y, p = add_at_snr(x, pink_noise(len(x), rng), c["snr_db"]), {"snr_db": c["snr_db"], "noise": "pink"}
    elif k == "reverb":
        n = int(c["rt60_s"] * 1.2 * SR)
        t = np.arange(n) / SR
        tail = rng.standard_normal(n) * np.exp(-6.9 * t / c["rt60_s"])
        tail[: int(0.008 * SR)] = 0
        tail = tail / np.sqrt(np.sum(tail ** 2)) * 0.6
        ir = np.zeros(n)
        ir[0] = 1.0
        ir += tail
        y = fftconvolve(x, ir)[: len(x)].astype(np.float32)
        y *= rms(x) / (rms(y) + 1e-9)
        p = {"rt60_s": c["rt60_s"]}
    elif k == "phone":
        sos = butter(4, [c["low_hz"], c["high_hz"]], btype="band", fs=SR, output="sos")
        y, p = sosfilt(sos, x).astype(np.float32), {"band_hz": [c["low_hz"], c["high_hz"]]}
    elif k == "mp3":
        with tempfile.TemporaryDirectory() as td:
            a, m, b = Path(td) / "a.wav", Path(td) / "a.mp3", Path(td) / "b.wav"
            sf.write(str(a), x, SR, subtype="FLOAT")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a), "-c:a", "libmp3lame", "-b:a", f"{c['kbps']}k", str(m)], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(m), "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", str(b)], check=True)
            y, _ = sf.read(str(b), dtype="float32")
        y, p = align_to(x, y), {"kbps": c["kbps"]}
    elif k == "gain":
        sign = 1 if rng.random() < 0.5 else -1
        y, p = x * undb(sign * c["db"]), {"db": sign * c["db"]}
    else:
        raise ValueError(code)
    return soft_limit(np.asarray(y, np.float32)), p
