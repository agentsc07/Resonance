"""Cross-speaker consensus. A different speaker's natural pace, pausing, level and pitch contour look like flaws, but they differ from one
reference voice to the next while a genuine flaw does not. The participant is compared with several clean reference voices that read the same
sentences (the text's panel, never a held-out speaker); a detection survives only when at least `need` references agree on it (same flaw,
overlapping region). This is the 'natural variation envelope' reduced to what the available speakers support."""
from __future__ import annotations

NEED = 2          # references that must agree (the primary reference plus at least one panel member); with a single reference there is no vote


def iou(a, b) -> float:
    o = min(a.end, b.end) - max(a.start, b.start)
    u = max(a.end, b.end) - min(a.start, b.start)
    return max(o, 0.0) / max(u, 1e-6)


def vote(per_ref: list[list], need: int = NEED, min_iou: float = 0.3) -> list:
    """Keep the primary reference's detections that `need - 1` other references also report. Their agreement count goes into facts."""
    if len(per_ref) <= 1:
        return per_ref[0] if per_ref else []
    need = min(need, len(per_ref))
    out = []
    for c in per_ref[0]:
        n = 1 + sum(any(o.flaw == c.flaw and iou(c, o) >= min_iou for o in per_ref[k]) for k in range(1, len(per_ref)))
        if n >= need:
            c.facts = {**c.facts, "references_agreeing": n, "references": len(per_ref)}
            out.append(c)
    return out
