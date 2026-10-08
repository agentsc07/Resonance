"""Tuning/eval loop on cached analyses. Thresholds are fit on TRAIN only; DEV checks; TEST is touched once at the end.
   python -m engine.tune --split train --mode same [--limit N]"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import metrics as M  # noqa: E402

from . import detect, predict  # noqa: E402

DATA = ROOT / "flawline-dataset"
CACHE = Path("/tmp/engine_cache")


def clips(split: str, limit=None, flawed_only=False):
    rows = list(csv.DictReader(open(DATA / "manifest.csv")))
    rows = [r for r in rows if r["split"] == split and not r["added_condition"] and (r["flaw_codes"] or not flawed_only)]
    out = []
    for r in rows:
        p = DATA / "variants" / f"{r['clip_id']}.json"
        if not p.exists():
            continue
        out.append(json.loads(p.read_text()))
    return out[:limit] if limit else out


def analysed(lab: dict, mode: str, ref_id: str | None = None):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{lab['clip_id']}__{mode}{'__' + ref_id.replace(':', '_') if ref_id else ''}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    meta = {k: v for k, v in lab.items() if k not in ("what", "seed", "join_check", "multi_set")}
    C, cands, q, ref, n = predict.analyse(str(DATA / "variants" / f"{lab['clip_id']}.flac"), meta, mode, None, ref_id)
    f.write_bytes(pickle.dumps((C, q, ref, n)))
    return C, q, ref, n


_LING: dict = {}


def run_detectors(C, ref_id, flaw=None):
    from .linguistics import annotate
    from wordfreq import zipf_frequency
    if ref_id not in _LING:
        ref_w = predict.reference.words_of(ref_id)
        _LING[ref_id] = (ref_w, annotate([{"w": r["w"], "sent_end": r["sent_end"], "clause_end": r["clause_end"]} for r in ref_w]))
    ref_w, ling = _LING[ref_id]
    bt = ling["b"][: C.n - 1] + ["sentence"]
    zn = [zipf_frequency(ref_w[min(k + 1, len(ref_w) - 1)]["clean"], "en") if ling["w"][min(k + 1, len(ref_w) - 1)]["content"] and len(ref_w[min(k + 1, len(ref_w) - 1)]["clean"]) > 4 else None for k in range(C.n)]
    return detect.run_flaw(flaw, C, bt, zn) if flaw else detect.run_all(C, bt, zn)


def refs_of(lab: dict, mode: str) -> list[str | None]:
    """Reference ids to analyse a clip against: same mode = the clip's own clean take; cross mode = the text's panel (primary + members)."""
    if mode != "cross":
        return [None]
    r1, members = predict.reference.panel_ids(lab["baseline_id"])
    return [None] + members


FREE_CACHE = Path("/tmp/free_cache")


def free_analysis(lab: dict):
    """Reference-free analysis of a dataset clip (ASR + frames), cached. The transcript is the text of the clip's baseline."""
    from . import audio, free
    FREE_CACHE.mkdir(exist_ok=True)
    f = FREE_CACHE / f"{lab['clip_id']}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    x = audio.normalise(audio.load(str(DATA / "variants" / f"{lab['clip_id']}.flac")))[0]
    fa = free.analyse(x, lab["baseline_id"])
    f.write_bytes(pickle.dumps(fa))
    return fa


CLEAN_TAKES = {"train": ["B01", "B02", "B04", "B06", "B07"], "dev": ["B03"], "extra": ["B09", "B10"]}


def clean_free_analysis(bid: str):
    """Reference-free analysis of a clean baseline reading (the take, with its own text), cached: the yardstick for false flags on clean speech."""
    from . import audio, free
    FREE_CACHE.mkdir(exist_ok=True)
    f = FREE_CACHE / f"clean_{bid}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    x = audio.normalise(audio.load(str(DATA / "takes" / f"{bid}-CHAMP_C0.flac")))[0]
    fa = free.analyse(x, bid)
    f.write_bytes(pickle.dumps(fa))
    return fa


def cands_for(lab: dict, mode: str, flaw=None):
    """Detections for a clip (consensus across the panel in cross mode). Returns (cands, C of the primary reference, its ref id)."""
    from . import consensus
    if mode == "free":
        from . import free
        fa = free_analysis(lab)
        return free.run(fa, detect.TH, flaw), fa, lab["baseline_id"]
    per, first = [], None
    for r in refs_of(lab, mode):
        C, q, ref, n = analysed(lab, mode, r)
        per.append(run_detectors(C, ref, flaw))
        first = first or (C, ref)
    return consensus.vote(per), first[0], first[1]


def to_pred(c):
    return {"flaw": c.flaw, "category": c.category, "start_s": c.start, "end_s": c.end, "word_start": c.w0, "word_end": c.w1, "severity": c.d, "kind": "modify"}


def evaluate(labs, mode, th=None, verbose=True, flaw=None):
    if th is None and flaw is None and detect.MODE != mode:
        detect.use(mode)
    if th:
        detect.TH.update(th)
    pairs = []
    for lab in labs:
        cands, C, ref = cands_for(lab, mode, flaw)
        preds = [to_pred(c) for c in cands if flaw or c.flaw not in detect.DISABLED]
        pairs.append((lab, {"what": preds, "duration_s": lab["duration_s"]}))
    g = M.grounding(pairs, 0.5, "flaw")
    if verbose:
        print(f"F1@0.5 overall {g['overall']['f1']:.3f} (P {g['overall']['precision']:.2f} R {g['overall']['recall']:.2f}, {len(pairs)} clips)")
        for k, v in g["per"].items():
            print(f"  {k:11s} P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}  (gt {v['n_gt']}, pred {v['n_pred']}, tp {v['tp']})")
    return g, pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--mode", default="same")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    detect.use(a.mode)
    evaluate(clips(a.split, a.limit), a.mode)
