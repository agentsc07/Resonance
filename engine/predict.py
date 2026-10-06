"""predict(audio_path, meta, mode) -> prediction JSON in label schema 1.1.0. `meta` carries only: clip_id, baseline_id (which text/reference),
genre, duration_s, who/where echo. It never carries ground-truth regions."""
from __future__ import annotations

import numpy as np

from . import arbitrate, audio, compare, conditions, detect, reference, score as scoring
from .explain import explain
from .linguistics import annotate

_REF_CACHE: dict = {}


def _ref(bid: str):
    if bid not in _REF_CACHE:
        w = reference.words_of(bid)
        x = audio.normalise(audio.load(reference.audio_of(bid)))[0]
        _REF_CACHE[bid] = (x, w, annotate([{"w": r["w"], "sent_end": r["sent_end"], "clause_end": r["clause_end"]} for r in w]))
    return _REF_CACHE[bid]


def analyse(audio_path: str, meta: dict, mode: str, ref_override: str | None = None):
    detect.use(mode)
    bid = meta["baseline_id"]
    if ref_override:                                           # e.g. a dashboard upload: the user names the baseline whose text they read
        bid = ref_override
    ref_id, n = reference.choose(bid, {"same": "same", "cross": "stitch", "free": "same"}[mode])
    ref_x, ref_w, ling = _ref(ref_id)
    par_x, _ = audio.normalise(audio.load(audio_path))
    if ref_id.startswith(reference.STITCH):                    # trailing sentences no other speaker read cannot be compared: crop them off
        st = reference.stitched(bid)
        if st["covered_to"] < st["n_mine"] - 1:
            own = reference.words_of(bid)
            frac = own[st["covered_to"]]["end_s"] / own[-1]["end_s"]
            par_x = par_x[: int(min(len(par_x), (frac * len(par_x) / audio.SR + 1.5) * audio.SR))]
    q = audio.quality(par_x)
    ref_m, cond = conditions.match(ref_x, par_x)                   # degrade the reference to the participant's measured conditions
    q["matched"] = cond
    C = compare.build(ref_m, ref_w, par_x, n)
    bt = ling["b"][: C.n - 1] + ["sentence"]
    from wordfreq import zipf_frequency
    zn = [zipf_frequency(ref_w[min(k + 1, len(ref_w) - 1)]["clean"], "en") if ling["w"][min(k + 1, len(ref_w) - 1)]["content"] and len(ref_w[min(k + 1, len(ref_w) - 1)]["clean"]) > 4 else None
          for k in range(C.n)]
    cands = detect.run_all(C, bt, zn)
    return C, cands, q, ref_id, n


def predict(audio_path: str, meta: dict, mode: str = "same", rubric: dict | None = None, dont_score_fluency: bool = False,
            ref_override: str | None = None, full: bool = False):
    """Prediction in label schema 1.1.0: regions with calibrated severity, confidence, causal explanation and points lost, plus scores.
    full=True also returns the comparison object (for the dashboard's time-warped overlay)."""
    C, cands, q, ref_id, n = analyse(audio_path, meta, mode, ref_override)
    ref_w = reference.words_of(ref_id)
    iso = scoring.load_iso(mode)
    rel = scoring.load_reliability(mode)
    cands = arbitrate.arbitrate(cands, rel)
    what, tips = [], []
    for c in cands:
        sev = scoring.severity(c.flaw, c.d, iso)
        sent, tip = explain(c, C, ref_w, sev)
        what.append({"flaw": c.flaw, "category": c.category, "start_s": round(c.start, 3), "end_s": round(c.end, 3),
                     "word_start": int(c.w0), "word_end": int(c.w1), "kind": "modify", "severity": round(sev, 3),
                     "confidence": q["confidence"], "reliability": round(rel.get(c.flaw, 1.0), 2), "explanation": sent,
                     "params": {"d": round(c.d, 3), "tip": tip, **c.facts}})
    sc = scoring.score(what, meta["duration_s"], meta.get("genre"), rubric, dont_score_fluency)
    for r, row in zip(what, sc["regions"]):
        r["points_lost"] = row["points_lost"]
    out = {"schema_version": "1.1.0", "clip_id": meta["clip_id"], "take_id": meta["take_id"], "baseline_id": meta["baseline_id"],
            "genre": meta.get("genre"), "who": meta.get("who", {"speaker_id": "?"}), "where": meta.get("where", {"base_condition": "C0", "added_condition": None}),
            "what": what, "duration_s": meta["duration_s"], "labels_from": "engine",
            "scores": {"overall": sc["overall"], "band": sc["band"], "categories": sc["categories"], "reference_mode": {"same": "same-speaker", "cross": "cross-speaker", "free": "reference-free"}[mode]},
            "quality": q, "reference": {"baseline_id": ref_id, "common_words": n}}
    return (out, C, ref_id) if full else out
