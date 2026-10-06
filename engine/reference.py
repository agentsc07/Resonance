"""Reference selection and loading. The engine only needs: a reference audio file, its text, and word timings for that text
(dataset assets: takes/*.flac + takes/*.align.json + baselines/*/reference.txt). It never reads the generator's code or labels."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "flawline-dataset"
BASELINES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]


STITCH = "cross:"          # reference ids of the form "cross:B03" mean "other speakers' sentences stitched into B03's text order"
_STITCHED: dict = {}


def words_of(bid: str) -> list[dict]:
    if bid.startswith(STITCH):
        return _member(bid)["words"]
    return json.loads((DATA / "takes" / f"{bid}-CHAMP_C0.align.json").read_text())["words"]


def audio_of(bid: str) -> str:
    if bid.startswith(STITCH):
        return _member(bid)["path"]
    return str(DATA / "takes" / f"{bid}-CHAMP_C0.flac")


def _member(rid: str) -> dict:
    """cross:B01 -> the primary stitched reference; cross:B01@B04 -> speaker B04's own version of the same sentences"""
    body = rid[len(STITCH):]
    if "@" not in body:
        return stitched(body)
    bid, p = body.split("@")
    r1 = stitched(bid)
    return stitched(bid, primary=p, only=frozenset(r1["sent"]))


def _sentences(W: list[dict]) -> list[tuple[int, int]]:
    out, s = [], 0
    for i, w in enumerate(W):
        if w["sent_end"] or i == len(W) - 1:
            out.append((s, i))
            s = i + 1
    return out


def stitched(bid: str, avoid: set[str] | None = None, min_ratio: float = 0.8, primary: str | None = None, only: frozenset | None = None) -> dict:
    """Cross-speaker reference for the text of `bid`: for each of its sentences take the best-matching sentence (word-sequence ratio >= min_ratio)
    from ANOTHER, non-held-out speaker's clean take and stitch those pieces, with the speaker's own following pause, in the participant's sentence
    order. Speakers read the same passages in different orders, so a common-prefix reference covers 14 words; this covers ~100 %.
      primary=None  -> one voice for as much of the reference as possible (fallback to others for the rest)
      primary="B04" -> that speaker ONLY (a panel member); sentences they did not read are skipped
      only          -> restrict to these sentence indices of the participant text (panel members match the primary reference's coverage)
    Returns {words, path, sources, sent (covered sentence indices), covered_to (last participant word covered), n_mine}."""
    import difflib

    import soundfile as sf

    from . import audio
    avoid = avoid or {"B05", "B08"}
    key = (bid, tuple(sorted(avoid)), primary, only)
    if key in _STITCHED:
        return _STITCHED[key]
    mine = words_of(bid)
    pool = [b for b in BASELINES if b != bid and b not in avoid]
    if primary:
        pool = [primary]
    W = {b: words_of(b) for b in pool}
    S = {b: _sentences(W[b]) for b in pool}
    X = {}
    sents = _sentences(mine)
    mine_s = [[w["clean"] for w in mine[s:e + 1]] for s, e in sents]
    tot = {o: sum(max([difflib.SequenceMatcher(None, a, [w["clean"] for w in W[o][s2:e2 + 1]]).ratio() for s2, e2 in S[o]] or [0]) for a in mine_s)
           for o in pool}
    lead = max(pool, key=lambda o: (tot[o], -BASELINES.index(o)))             # one voice for as much of the reference as possible
    pieces, prev, covered_to, got = [], None, -1, []
    for si, (s, e) in enumerate(sents):
        if only is not None and si not in only:
            continue
        a = mine_s[si]
        best = (0.0, None, None)
        for o in pool:
            for s2, e2 in S[o]:
                r = difflib.SequenceMatcher(None, a, [w["clean"] for w in W[o][s2:e2 + 1]]).ratio()
                r += 0.05 if o == lead else (0.01 if o == prev else 0.0)
                if r > best[0]:
                    best = (r, o, (s2, e2))
        if best[0] < min_ratio:
            continue
        if covered_to == s - 1 or covered_to == -1 and s == 0:
            covered_to = e
        o, (s2, e2) = best[1], best[2]
        if o not in X:
            X[o] = audio.normalise(audio.load(str(DATA / "takes" / f"{o}-CHAMP_C0.flac")))[0]
        wo = W[o]
        t0 = 0.0 if s2 == 0 else (wo[s2 - 1]["end_s"] + wo[s2]["start_s"]) / 2
        t1 = len(X[o]) / audio.SR if e2 + 1 >= len(wo) else (wo[e2]["end_s"] + wo[e2 + 1]["start_s"]) / 2
        pieces.append((o, t0, t1, wo[s2:e2 + 1]))
        got.append(si)
        prev = o
    words, chunks, off = [], [], 0.0
    for o, t0, t1, ws in pieces:
        for w in ws:
            d = dict(w)
            d["start_s"] = round(w["start_s"] - t0 + off, 4)
            d["end_s"] = round(w["end_s"] - t0 + off, 4)
            d["i"] = len(words)
            words.append(d)
        chunks.append(X[o][int(t0 * audio.SR): int(t1 * audio.SR)])
        off += (t1 - t0)
    out_dir = DATA / ".cache" / "stitched"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = bid + (f"@{primary}" if primary else "") + (f"#{len(only)}" if only is not None else "")
    path = out_dir / f"{tag}.wav"
    sf.write(str(path), (import_np().concatenate(chunks) if chunks else import_np().zeros(1600)), audio.SR)
    res = {"words": words, "path": str(path), "sources": sorted({p[0] for p in pieces}), "sent": got, "covered_to": covered_to, "n_mine": len(mine)}
    _STITCHED[key] = res
    return res


def panel_ids(bid: str, avoid: set[str] | None = None) -> tuple[str, list[str]]:
    """(primary reference id, member ids) for the text of `bid`. Members are other clean speakers (never held out, never `bid`) who read every
    sentence the primary reference covers, each stitched from that speaker alone: the natural variation envelope of the text."""
    avoid = avoid or {"B05", "B08"}
    r1 = stitched(bid, avoid)
    lead = r1["sources"][0] if len(r1["sources"]) == 1 else None
    members = []
    for p in BASELINES:
        if p in avoid or p == bid or (lead and p == lead):
            continue
        m = stitched(bid, avoid, primary=p, only=frozenset(r1["sent"]))
        if len(m["sent"]) == len(r1["sent"]):
            members.append(f"{STITCH}{bid}@{p}")
    return STITCH + bid, members


def import_np():
    import numpy
    return numpy


def common_prefix(a: list[dict], b: list[dict]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x["clean"] != y["clean"]:
            break
        n += 1
    return n


def choose(bid: str, mode: str, avoid: set[str] | None = None) -> tuple[str, int]:
    """Returns (reference baseline id, n common words). same = the clip's own clean take; cross = ANOTHER speaker's clean reading of the same text
    (longest matching text; never a held-out speaker so the test speakers are not used as yardsticks)."""
    mine = words_of(bid)
    if mode == "same":
        return bid, len(mine)
    if mode == "stitch":
        st = stitched(bid)
        return STITCH + bid, len(st["words"])
    avoid = avoid or {"B05", "B08"}
    best = max((b for b in BASELINES if b != bid and b not in avoid),
               key=lambda b: (common_prefix(mine, words_of(b)), -BASELINES.index(b)))
    return best, common_prefix(mine, words_of(best))
