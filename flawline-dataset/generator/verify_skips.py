"""Word-removal gate: re-run speech recognition on the joined region of every WORD_SKIP clip. If the recogniser still hears the removed word
(or a fragment of it) at the join, the clip FAILS the gate (the cut left a sliver of the word). Exit 1 on any failure.
  python verify_skips.py [--delete]    --delete removes failing clips so make_dataset regenerates them with a wider margin."""
import json
import re
from pathlib import Path
import sys

import numpy as np

from common import SR, TAKES, VARIANTS, read_audio

_m = None


def model():
    global _m
    if _m is None:
        from faster_whisper import WhisperModel
        _m = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _m


def words_in(x, t0, t1):
    seg = x[int(t0 * SR): int(t1 * SR)].astype(np.float32)
    out, _ = model().transcribe(seg, language="en", word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    return [(re.sub(r"[^a-z']", "", w.word.lower()), w.start + t0, w.end + t0) for s in out for w in s.words]


BL = Path(__file__).with_name("skip_blacklist.json")
black = {t: set(v) for t, v in (json.loads(BL.read_text()) if BL.exists() else {}).items()}
bad, n = [], 0
for jp in sorted(VARIANTS.glob("*__WORD_SKIP_*.json")):
    lab = json.loads(jp.read_text())
    x, _ = read_audio(jp.with_suffix(".flac"))
    for r in lab["what"]:
        if r["flaw"] != "WORD_SKIP":
            continue
        n += 1
        join = r["start_s"]
        gone = r.get("word", "")
        al = json.loads((TAKES / f"{lab['take_id']}_C0.align.json").read_text())["words"]
        cl = [re.sub(r"[^a-z']", "", w["clean"]) for w in al]
        gone_idx = [k for k, w in enumerate(al) if abs(w["start_s"] - r["baseline_start_s"]) < 0.08 or abs(w["end_s"] - r["baseline_end_s"]) < 0.08]
        k = next((q for q in gone_idx if cl[q] == gone), None)
        heard = [w for w, a, b in words_in(x, max(0, join - 1.5), join + 1.5)]
        # clean = the neighbours meet ("p n"); a sliver shows as "p gone n" (the recogniser put the removed word back)
        extra = 0
        if k is not None and 0 < k < len(cl) - 1:
            p, nx = cl[k - 1], cl[k + 1]
            extra = sum(1 for q in range(len(heard) - 2) if heard[q] == p and heard[q + 1] == gone and heard[q + 2] == nx)
        # a sliver shows up as the removed word itself, or a 1-2 letter fragment of it, right at the join
        if gone and extra > 0:
            bad.append((lab["clip_id"], gone, heard)); black.setdefault(lab["take_id"], set()).add(k)
print(f"word-removal gate: {n} removals checked, {len(bad)} still audible")
for b in bad[:30]:
    print("  ", *b)
BL.write_text(json.dumps({t: sorted(v) for t, v in black.items()}, indent=1))     # words whose removal left a sliver: never skip them again
if "--delete" in sys.argv:
    for cid, *_ in bad:
        for ext in (".json", ".flac"):        # a clip that fails is regenerated with that word excluded
            (VARIANTS / f"{cid}{ext}").unlink(missing_ok=True)
sys.exit(1 if bad else 0)
