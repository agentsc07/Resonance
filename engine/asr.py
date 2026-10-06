"""Recognised words with timestamps (faster-whisper). Two passes: unprompted (text fidelity) and disfluency-prompted (fillers survive).
Cached on disk by audio hash so repeated evaluation runs are fast and deterministic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import SR

CACHE = Path(__file__).parent / ".cache"
PROMPT = "Um, uh, so, like, er, you know, uhm."
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm", "uhm", "eh"}
_MODEL = None


def _norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha() or c == "'")


def words(x: np.ndarray, prompted: bool, model_name: str = "base.en") -> list[dict]:
    key = hashlib.sha256(x.tobytes() + f"{prompted}{model_name}".encode()).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    global _MODEL
    from faster_whisper import WhisperModel
    if _MODEL is None or _MODEL[0] != model_name:
        _MODEL = (model_name, WhisperModel(model_name, device="cpu", compute_type="int8"))
    segs, _ = _MODEL[1].transcribe(x, word_timestamps=True, language="en", beam_size=5, condition_on_previous_text=False,
                                   temperature=0.0, initial_prompt=PROMPT if prompted else None)
    out = [{"w": _norm(w.word), "raw": w.word.strip(), "start": float(w.start), "end": float(w.end)}
           for s in segs for w in s.words if _norm(w.word)]
    f.write_text(json.dumps(out))
    return out
