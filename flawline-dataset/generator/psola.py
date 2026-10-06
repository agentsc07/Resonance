"""PSOLA (Praat overlap-add) pitch and duration edits. Unlike WORLD resynthesis, PSOLA re-uses the original
waveform periods, so timbre, noise and voice quality stay the speaker's own (no vocoder buzz)."""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call, run

from common import SR, rms


def _seed(n: int = 0):
    """Praat's overlap-add draws random numbers (unvoiced duplication); without a fixed seed two runs differ and the dataset is not
    reproducible bit for bit. Seed before every Praat resynthesis."""
    run(f"random_initializeWithSeedUnsafelyButPredictably({12345 + int(n) % 9973})")


def _range(median_hz: float, n_samples: int | None = None) -> tuple[float, float]:
    fmin, fmax = max(60.0, median_hz * 0.55), min(600.0, median_hz * 2.4)
    if n_samples:                                   # Praat: minimum pitch must be >= 3 periods of the sound's duration
        fmin = max(fmin, 3.3 / (n_samples / SR))
    return fmin, max(fmax, fmin * 2.0)


def clean_octaves(f: np.ndarray) -> np.ndarray:
    """Pitch trackers make octave jumps and glitches (esp. creaky sentence ends). Replace points that stray >20% from the local median
    so an edit never amplifies a tracking error into an audible squeak."""
    from scipy.signal import medfilt
    if len(f) < 5:
        return f
    ref = medfilt(f, 7 if len(f) >= 7 else 5)
    bad = np.abs(f / np.maximum(ref, 1.0) - 1.0) > 0.20
    out = f.copy()
    out[bad] = ref[bad]
    return out


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def psola_pitch(seg: np.ndarray, f0_fn, median_hz: float) -> np.ndarray:
    """f0_fn(f0_array, times_s) -> new f0 array, evaluated at Praat's voiced pitch-tier points."""
    _seed(len(seg))
    fmin, fmax = _range(median_hz, len(seg))
    snd = parselmouth.Sound(seg.astype(np.float64), SR)
    m = call(snd, "To Manipulation", 0.01, fmin, fmax)
    pt = call(m, "Extract pitch tier")
    n = call(pt, "Get number of points")
    if n < 3:
        return seg
    t = np.array([call(pt, "Get time from index", i) for i in range(1, n + 1)])
    f = clean_octaves(np.array([call(pt, "Get value at index", i) for i in range(1, n + 1)]))
    f2 = np.clip(f0_fn(f, t), 40.0, 900.0)
    new = call("Create PitchTier", "new", 0.0, len(seg) / SR)
    for ti, fi in zip(t, f2):
        call(new, "Add point", float(ti), float(fi))
    call([new, m], "Replace pitch tier")
    y = call(m, "Get resynthesis (overlap-add)").values[0]
    y = _fit(y, len(seg))
    return y * (rms(seg) / (rms(y) + 1e-9))


def psola_stretch(seg: np.ndarray, factor: float, median_hz: float) -> np.ndarray:
    """Duration x factor (>1 slower), pitch preserved, original periods re-used."""
    _seed(len(seg))
    fmin, fmax = _range(median_hz, len(seg))
    snd = parselmouth.Sound(seg.astype(np.float64), SR)
    out = call(snd, "Lengthen (overlap-add)", fmin, fmax, float(factor))
    return out.values[0].astype(np.float32)


def psola_match(seg: np.ndarray, target_len: int, ratio_f0: float, median_hz: float) -> np.ndarray:
    """Re-time `seg` to target_len samples and scale its pitch by ratio_f0 (word-swap donor matching)."""
    _seed(len(seg))
    fmin, fmax = _range(median_hz, len(seg))
    snd = parselmouth.Sound(seg.astype(np.float64), SR)
    m = call(snd, "To Manipulation", 0.01, fmin, fmax)
    pt = call(m, "Extract pitch tier")
    if call(pt, "Get number of points") >= 3:
        call(pt, "Multiply frequencies", 0.0, len(seg) / SR, float(ratio_f0))
        call([pt, m], "Replace pitch tier")
    dt = call("Create DurationTier", "d", 0.0, len(seg) / SR)
    call(dt, "Add point", 0.0, target_len / len(seg))
    call([dt, m], "Replace duration tier")
    y = call(m, "Get resynthesis (overlap-add)").values[0]
    return _fit(y, target_len) if abs(len(y) - target_len) < 0.2 * target_len else y.astype(np.float32)
