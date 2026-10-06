"""Audit baselines for natural disfluencies BEFORE they are used as the 'clean' yardstick.

A baseline must be clean: no fillers, repeats, false starts, skipped/misread words, odd pauses, clipping.
Method: verbatim-biased ASR (faster-whisper, prompted with disfluent text so it keeps um/uh), diffed against the
reference text. ASR can miss a quiet filler, so a PASS means "nothing found", not proof; a hit is listened to.
Output: baselines/audit.json  (read by the dashboard).  Usage: python audit_baselines.py [B01 ...]
"""
from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher

import numpy as np
from scipy.signal import resample_poly

from common import BASELINES, GEN, SR, TAKES, db, frame_rms_db, read_audio, tokenize

FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm", "uhm", "eh"}
PROMPT = "Um, so, uh, I mean, like, the, the, uh... well, er, you know, uhm."
_M = None


NUM = {str(i): n for i, n in enumerate("zero one two three four five six seven eight nine ten".split())}
SPELL = {"colours": "colors", "colour": "color", "grey": "gray", "centre": "center", "metre": "meter", "realise": "realize"}


def norm(t: str) -> str:
    return SPELL.get(t, t)


def asr(x, prompt=None):
    global _M
    from faster_whisper import WhisperModel
    if _M is None:
        _M = WhisperModel("small.en", device="cpu", compute_type="int8")
    x16 = resample_poly(x, 320, 441).astype(np.float32)
    segs, _ = _M.transcribe(x16, word_timestamps=True, language="en", beam_size=5, initial_prompt=prompt,
                            condition_on_previous_text=False, temperature=0.0)
    out = []
    for s in segs:
        for w in s.words:
            raw = w.word.strip().lower()
            t = NUM.get(raw.strip(".,"), "".join(c for c in raw if c.isalpha() or c == "'"))
            if t:
                out.append((t, w.start, w.end))
    return out


def audit(slot: str) -> dict:
    take = f"{slot}-CHAMP"
    x, _ = read_audio(TAKES / f"{take}_C0.flac")
    al = json.loads((TAKES / f"{take}_C0.align.json").read_text())["words"]
    al = [w for w in al if w["clean"]]                            # drop non-word tokens
    ref = [w["clean"] for w in al]
    hyp = [(norm(t), a, b) for t, a, b in asr(x)]                 # text fidelity: unprompted run
    verb = asr(x, PROMPT)                                         # filler hunt: disfluency-biased run
    sm = SequenceMatcher(a=ref, b=[h[0] for h in hyp], autojunk=False)
    issues = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        rw, hw = ref[i1:i2], [h[0] for h in hyp[j1:j2]]
        t = hyp[j1][1] if j2 > j1 else al[min(i1, len(al) - 1)]["start_s"]
        if tag == "insert" or (tag == "replace" and all(h in FILLERS for h in hw)):
            kind = "filler" if all(h in FILLERS for h in hw) else "extra words"
        elif tag == "delete":
            kind = "ASR dropout (recheck)" if len(rw) > 6 else "skipped word(s)"
        else:
            kind = "misread / ASR mismatch"
        issues.append({"kind": kind, "time_s": round(float(t), 2), "reference": " ".join(rw), "heard": " ".join(hw)})
    for t, a, b in verb:                                          # fillers heard only in the biased run
        if t in FILLERS and not any(abs(a - i["time_s"]) < 0.5 and i["kind"] == "filler" for i in issues):
            issues.append({"kind": "filler", "time_s": round(float(a), 2), "reference": "", "heard": t})
    # repeated adjacent words that are not in the reference
    for (a, ta, _), (b, tb, _) in zip(hyp, hyp[1:]):
        if a == b and a not in FILLERS and not any(r == a and r2 == a for r, r2 in zip(ref, ref[1:])):
            issues.append({"kind": "repeat", "time_s": round(float(ta), 2), "reference": "", "heard": f"{a} {b}"})
    gaps = [(al[i + 1]["start_s"] - al[i]["end_s"], al[i]["end_s"], al[i]) for i in range(len(al) - 1)
            if not al[i]["punct"]]
    for g, t, w in gaps:
        if g > 0.45:
            issues.append({"kind": "pause inside phrase" if g > 0.6 else "long pause (minor)", "time_s": round(t, 2),
                           "reference": w["w"], "heard": f"{g:.2f} s gap"})
    clip = float(np.mean(np.abs(x) > 0.99))
    fr = frame_rms_db(x)
    snr = float(np.percentile(fr, 95) - np.mean(np.sort(fr)[: max(1, len(fr) // 20)]))
    src = json.loads((BASELINES / slot / "source.json").read_text())
    spont = src.get("genre") == "extemporaneous"
    n_f = sum(i["kind"] == "filler" for i in issues)
    n_other = len(issues) - n_f
    hard = [i for i in issues if i["kind"] in ("filler", "repeat", "pause inside phrase", "extra words")]
    word_err = sum(len(i["reference"].split()) or 1 for i in issues if i["kind"] in ("skipped word(s)", "misread / ASR mismatch")) / max(len(ref), 1)
    minor = [i for i in issues if i["kind"] == "long pause (minor)"]
    if clip > 0.001:
        status = "FAIL"
    elif hard:
        status = "NATURAL" if spont else "FAIL"           # spontaneous speech: disfluencies are real, recorded, not errors
    elif word_err > 0.03 or minor or snr < 20:
        status = "REVIEW"
    else:
        status = "PASS"
    return {"slot": slot, "status": status, "n_words": len(ref), "fillers": n_f, "issues": sorted(issues, key=lambda i: i["time_s"]),
            "asr_mismatch_rate": round(word_err, 3), "clipped_frac": round(clip, 5), "snr_db_est": round(snr, 1),
            "spontaneous": spont, "natural_disfluencies": len(hard) if spont else 0}


if __name__ == "__main__":
    want = set(sys.argv[1:])
    res = {}
    for d in sorted(BASELINES.iterdir()):
        if d.is_dir() and (not want or d.name in want):
            res[d.name] = audit(d.name)
            r = res[d.name]
            print(f"{d.name} {r['status']:8s} fillers={r['fillers']} flags={len(r['issues'])} mismatch={r['asr_mismatch_rate']:.1%} snr~{r['snr_db_est']}dB clip={r['clipped_frac']:.3%}")
            for i in r["issues"][:12]:
                print(f"    {i['time_s']:6.2f}s {i['kind']:24s} ref='{i['reference']}' heard='{i['heard']}'")
    out = BASELINES / "audit.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(res)
    out.write_text(json.dumps(old, indent=2) + "\n")
