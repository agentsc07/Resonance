"""Needleman-Wunsch on word strings: participant (recognised) words vs reference words. Exposes skips (reference words with no partner),
insertions (fillers, repeats) and swaps (substitutions)."""
from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def needleman_wunsch(ref: list[str], hyp: list[str], gap: float = -0.6) -> list[tuple[int | None, int | None]]:
    n, m = len(ref), len(hyp)
    S = np.zeros((n + 1, m + 1))
    S[:, 0] = np.arange(n + 1) * gap
    S[0, :] = np.arange(m + 1) * gap
    sc = np.array([[(1.0 if ref[i] == hyp[j] else (2 * sim(ref[i], hyp[j]) - 1) if sim(ref[i], hyp[j]) > 0.6 else -1.0)
                    for j in range(m)] for i in range(n)]) if n and m else np.zeros((n, m))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i, j] = max(S[i - 1, j - 1] + sc[i - 1, j - 1], S[i - 1, j] + gap, S[i, j - 1] + gap)
    i, j, path = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(S[i, j], S[i - 1, j - 1] + sc[i - 1, j - 1]):
            path.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and np.isclose(S[i, j], S[i - 1, j] + gap):
            path.append((i - 1, None)); i -= 1
        else:
            path.append((None, j - 1)); j -= 1
    return path[::-1]
