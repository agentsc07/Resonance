"""Cross-detector arbitration (locality). One cause often moves several measurements: a rushed stretch has short pauses, a shouted clause is
also 'prominent', a dulled one is also 'flat'. Each detector sees only its own symptom, so overlapping detections of different causes are resolved
here: known secondary symptoms are dropped inside a primary region, and any remaining overlap keeps the detection whose flaw is most reliable
(its precision on the TRAIN split at the calibrated thresholds), so a noisy detector never outbids a trustworthy one."""
from __future__ import annotations

# primary flaw -> flaws that are symptoms of it when they sit inside its region
SECONDARY = {
    "PACE_FAST": {"PAUSE_BAD", "PAUSE_LOST", "FILLER", "RARE_HESIT", "EMPH_FLAT", "SLUR", "UPTALK"},
    "PACE_SLOW": {"PAUSE_BAD", "PAUSE_LOST", "FILLER", "RARE_HESIT", "EMPH_FLAT", "SLUR", "UPTALK"},
    "SHOUT": {"EMPH_FLAT", "SLUR", "MONOTONE"},
    "FADE": {"EMPH_FLAT", "SLUR", "UPTALK"},
    "SLUR": {"EMPH_FLAT"},
    "MONOTONE": {"EMPH_FLAT"},
    "WORD_SKIP": {"PACE_FAST", "PAUSE_LOST", "EMPH_FLAT"},
    "WORD_SWAP": {"EMPH_FLAT", "SLUR"},
}


def _ov(a, b) -> float:
    """overlap as a fraction of the shorter region"""
    o = min(a.end, b.end) - max(a.start, b.start)
    return max(0.0, o) / max(min(a.end - a.start, b.end - b.start), 0.05)


def arbitrate(cands: list, reliability: dict[str, float], min_overlap: float = 0.5) -> list:
    drop = set()
    for i, p in enumerate(cands):
        for j, q in enumerate(cands):
            if i == j or j in drop or i in drop:
                continue
            if q.flaw in SECONDARY.get(p.flaw, ()) and _ov(p, q) >= 0.3:
                drop.add(j)
    keep = [c for k, c in enumerate(cands) if k not in drop]
    drop = set()
    for i, p in enumerate(keep):
        for j in range(i + 1, len(keep)):
            q = keep[j]
            if i in drop or j in drop or p.category == q.category or _ov(p, q) < min_overlap:
                continue
            rp, rq = reliability.get(p.flaw, 1.0), reliability.get(q.flaw, 1.0)
            if abs(rp - rq) < 0.05:
                continue
            drop.add(j if rp > rq else i)
    return [c for k, c in enumerate(keep) if k not in drop]
