"""Shared helpers: paths, audio I/O, dB maths, crossfades, room tone."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

SR = 22050
ROOT = Path(__file__).resolve().parent.parent          # flawline-dataset/
GEN = ROOT / "generator"
TAKES = ROOT / "takes"
VARIANTS = ROOT / "variants"
BASELINES = ROOT / "baselines"
FILLERS = ROOT / "fillers"
PILOT = ROOT / "pilot"
GENERATOR_VERSION = "1.0.0"


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_config():
    return load_yaml(GEN / "config.yaml")


def load_baselines():
    return load_yaml(GEN / "baselines.yaml")["baselines"]


def db(x: float) -> float:
    return 20 * np.log10(max(float(x), 1e-9))


def undb(d: float) -> float:
    return 10 ** (d / 20)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-18))


def read_audio(path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def write_audio(path, x: np.ndarray, sr: int = SR):
    sf.write(str(path), np.clip(x, -1.0, 1.0).astype(np.float32), sr, format="FLAC", subtype="PCM_16")


def s2n(t: float, sr: int = SR) -> int:
    return int(round(t * sr))


def sola_trim(tail: np.ndarray, p: np.ndarray, fade: int, maxlag: int) -> int:
    """How many samples to trim off the start of `p` so its first `fade` samples are best phase-aligned (max normalised
    correlation) with the last `fade` samples of `tail`. Waveform-matched joins avoid the click of an out-of-phase splice."""
    if len(tail) < fade or len(p) < fade + maxlag:
        return 0
    ref = tail[-fade:].astype(np.float64)
    best, best_l = -2.0, 0
    for lag in range(0, maxlag + 1):
        c = p[lag: lag + fade].astype(np.float64)
        d = np.sqrt(np.dot(ref, ref) * np.dot(c, c)) + 1e-12
        v = float(np.dot(ref, c) / d)
        if v > best:
            best, best_l = v, lag
    return best_l


def crossfade_concat(pieces: list[np.ndarray], fade: int, align: bool = False) -> np.ndarray:
    """Join pieces with a linear crossfade of `fade` samples at each joint (optionally waveform-matched, SOLA-style)."""
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    out = pieces[0].astype(np.float32).copy()
    for p in pieces[1:]:
        p = p.astype(np.float32)
        if align:
            p = p[sola_trim(out, p, fade, s2n(0.006)):]
        f = min(fade, len(out), len(p))
        if f > 1:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            a_, b_ = out[-f:].astype(np.float64), p[:f].astype(np.float64)
            corr = float(np.dot(a_, b_) / (np.sqrt(np.dot(a_, a_) * np.dot(b_, b_)) + 1e-12))
            if corr > 0.5:                       # phase-aligned (pitch-synchronous) join: linear keeps amplitude
                mixed = out[-f:] * (1 - ramp) + p[:f] * ramp
            else:                                # uncorrelated: equal-power keeps loudness
                mixed = out[-f:] * np.cos(ramp * np.pi / 2) + p[:f] * np.sin(ramp * np.pi / 2)
            out = np.concatenate([out[:-f], mixed, p[f:]])
        else:
            out = np.concatenate([out, p])
    return out


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic pink noise, unit RMS."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    spec /= np.sqrt(f)
    y = np.fft.irfft(spec, n)
    y /= np.sqrt(np.mean(y ** 2)) + 1e-12
    return y.astype(np.float32)


def frame_rms_db(x: np.ndarray, hop: int = 220, win: int = 441) -> np.ndarray:
    n = max(1, 1 + (len(x) - win) // hop) if len(x) >= win else 1
    out = np.empty(n)
    for i in range(n):
        seg = x[i * hop: i * hop + win]
        out[i] = db(rms(seg)) if len(seg) else -120.0
    return out


def estimate_room_tone_db(x: np.ndarray) -> float:
    """Noise floor estimate: mean level of the quietest 5% of 20 ms frames."""
    fr = np.sort(frame_rms_db(x))
    k = max(1, int(0.05 * len(fr)))
    return float(np.mean(fr[:k]))


def room_tone(n: int, like: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Room tone matched to the noise floor of `like`. Never digital silence."""
    level = max(estimate_room_tone_db(like), -70.0)
    return pink_noise(n, rng) * undb(level)


def ffmpeg_filter(x: np.ndarray, filt: str, sr: int = SR) -> np.ndarray:
    """Run mono float audio through an ffmpeg -af chain, return float32 at the same rate."""
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(a), x.astype(np.float32), sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a), "-af", filt,
                        "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", str(b)], check=True)
        y, _ = sf.read(str(b), dtype="float32")
    return y


def soft_limit(x: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """Smooth limiter so gain-type flaws never clip (QA: no clipping)."""
    knee = 0.8 * ceiling
    y = x.copy()
    a = np.abs(y)
    m = a > knee
    y[m] = np.sign(y[m]) * (knee + (ceiling - knee) * np.tanh((a[m] - knee) / (ceiling - knee)))
    return y


_PUNCT = re.compile(r"[,;:.?!]+$")


def tokenize(text: str) -> list[dict]:
    """Whitespace tokens with trailing punctuation split off; sentence/clause flags."""
    out = []
    for i, tok in enumerate(text.split()):
        m = _PUNCT.search(tok)
        punct = m.group(0) if m else ""
        clean = re.sub(r"[^A-Za-z']", "", tok).lower()
        out.append({"i": i, "text": tok, "clean": clean, "punct": punct,
                    "sent_end": bool(punct and punct[-1] in ".?!"),
                    "clause_end": bool(punct and punct[-1] in ",;:")})
    return out


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- splice quality (shared by generator gate + meter)
def click_db(x: np.ndarray, t: float) -> float:
    """Classic splice-click detector: peak of the 2nd difference within +-1.5 ms of time t vs the RMS of the 2nd difference in
    the surrounding +-25 ms (join excluded). A phase-continuous join looks like ordinary speech."""
    c = s2n(t)
    r, ex = s2n(0.025), s2n(0.0015)
    seg = x[max(0, c - r - 2): c + r + 2]
    d2 = np.abs(np.diff(seg.astype(np.float64), 2))
    m = len(d2) // 2
    near = d2[max(0, m - ex): m + ex]
    far = np.concatenate([d2[: max(0, m - ex)], d2[m + ex:]])
    if len(near) == 0 or len(far) == 0:
        return 0.0
    return float(20 * np.log10((near.max() + 1e-12) / (np.sqrt(np.mean(far ** 2)) + 1e-12)))


def join_clicks(xb: np.ndarray, xv: np.ndarray, regions: list[dict]) -> list[float]:
    """Click excess (dB over what the BASELINE does naturally at the same place) at every region edge."""
    out = []
    dur_v, dur_b = len(xv) / SR, len(xb) / SR
    fr = frame_rms_db(xb)
    floor = float(np.percentile(fr, 5))
    local = lambda x, t: db(rms(x[max(0, s2n(t) - s2n(0.01)): s2n(t) + s2n(0.01)]))
    for r in regions:
        pts = [("start_s", "baseline_start_s"), ("end_s", "baseline_end_s")] if r["end_s"] - r["start_s"] > 0.02 else [("start_s", "baseline_start_s")]
        for tv_k, tb_k in pts:
            tv, tb = r[tv_k], r[tb_k]
            if 0.05 < tv < dur_v - 0.05 and 0.05 < tb < dur_b - 0.05:
                if local(xv, tv) < floor + 15.0:          # a join in near-silence (<15 dB above the floor) is inaudible, and the detector
                    out.append(0.0)                       # would only measure noise statistics)
                    continue
                out.append(max(0.0, click_db(xv, tv) - click_db(xb, tb)))
    return out


def high_shelf(x: np.ndarray, fc: float, gain_db: float) -> np.ndarray:
    """RBJ high-shelf biquad (spectral-tilt flattening that accompanies real vocal effort)."""
    from scipy.signal import lfilter
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    cw, alpha = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
    return lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], x).astype(np.float32)
