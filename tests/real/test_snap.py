"""Unit checks for word snapping and the long-hesitation / held-vowel rules on the real recordings (skipped when the WAVs are absent).
   python -m pytest tests/real/test_snap.py"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from engine import asr, audio, detect, free, transcript  # noqa: E402


def analyse(name):
    p = HERE / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    detect.use("free")
    x = audio.normalise(audio.load(str(p)))[0]
    heard = " ".join(h["raw"] for h in asr.words(x, False))
    return free.analyse(x, "B01", transcript.words_from_text(heard))


def test_silence_inside_word_is_given_to_the_boundary():
    fa = analyse("test-2.wav")
    k = next(i for i, w in enumerate(fa.words) if w["clean"] == "your")
    L, a, b = free.pauses(fa)[k]
    assert 0.5 <= L <= 0.62 and abs(a - 0.89) < 0.05


def test_hesitation_and_filler_found():
    fa = analyse("test-uhh.wav")
    got = {(c.flaw, round(c.start, 1)) for c in free.run(fa, detect.TH)}
    assert ("PAUSE_BAD", 3.8) in got and any(f == "FILLER" and 3.3 <= s <= 3.6 for f, s in got)
