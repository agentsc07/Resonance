"""Baseline-vs-participant DTW on speaker-normalised spectral features: the timing backbone (pace, pauses, insertions, deletions)."""
from __future__ import annotations

import librosa
import numpy as np

from .audio import SR

HOP = 160           # 10 ms
FLOOR_MARGIN = None  # log-mel units above each band's 10th percentile below which a frame is treated as noise (None = off)


def feats(x: np.ndarray) -> np.ndarray:
    """log-mel + deltas with per-utterance mean/variance normalisation (removes speaker and channel offsets)."""
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=400, hop_length=HOP, n_mels=40, fmin=60, fmax=7600)
    lm = np.log(m + 1e-8)
    # floor-aware: below (10th percentile + margin) per band is noise, not speech: clamp it so silent/noisy frames look alike on both sides
    # and noise detail (babble, hiss, reverb tails) cannot steer the alignment
    floor = np.percentile(lm, 10, axis=1, keepdims=True)
    if FLOOR_MARGIN is not None:
        lm = np.maximum(lm, floor + FLOOR_MARGIN)
    lm = (lm - lm.mean(axis=1, keepdims=True)) / (lm.std(axis=1, keepdims=True) + 1e-6)
    d = librosa.feature.delta(lm, width=5)
    return np.vstack([lm, 0.5 * d]).T.astype(np.float32)


def align(ref_x: np.ndarray, part_x: np.ndarray, band: int | None = None):
    """Returns (ref_frame, part_frame, mean path cost, local cost along the path). Path arrays are nondecreasing. Steps allow 0.5x-2x tempo plus
    horizontal/vertical moves (insertion / deletion) at a small penalty."""
    A, B = feats(ref_x), feats(part_x)
    C = 1 - (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :] + 1e-9)
    steps = np.array([[1, 1], [1, 2], [2, 1], [0, 1], [1, 0]])
    w = np.array([1.0, 1.0, 1.0, 0.6, 0.6])
    D, wp = librosa.sequence.dtw(C=C, step_sizes_sigma=steps, weights_mul=w, weights_add=np.array([0, 0, 0, 0.3, 0.3]), backtrack=True)
    wp = wp[::-1]
    return wp[:, 0], wp[:, 1], float(D[-1, -1] / len(wp)), C[wp[:, 0], wp[:, 1]]
