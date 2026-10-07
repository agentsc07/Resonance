"""Human reference set: labelling API (lab only). Raters listen to clips in flawline-dataset/human/clips/, press SPACE where something slips, mark an area,
and give an overall 1-10 score. Engine output is never served here, and hidden_map.json / transcripts.json are never read by these routes.
Labels land in flawline-dataset/human/labels_<rater>.json; the engine-vs-human unmatched-flag review (agree / disagree / unsure) in review_<reviewer>.json."""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

HUMAN = Path(__file__).resolve().parent.parent / "flawline-dataset" / "human"
CLIPS = HUMAN / "clips"
EXTS = {".wav", ".m4a", ".flac", ".mp3", ".ogg"}
AREAS = ["Pacing", "Pausing", "Intonation", "Volume", "Fluency", "Clarity", "Text fidelity"]
router = APIRouter(prefix="/api/human")


def rater_slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    if not s:
        raise HTTPException(400, "Enter your name first.")
    return s[:40]


def clip_names() -> list[str]:
    return sorted(p.name for p in CLIPS.iterdir() if p.suffix.lower() in EXTS) if CLIPS.exists() else []


def order_for(slug: str) -> list[str]:
    """A fixed shuffled order per rater (stable across sessions and across added clips of other names)."""
    return sorted(clip_names(), key=lambda n: hashlib.sha256(f"{slug}|{n}".encode()).hexdigest())


def labels_path(slug: str) -> Path:
    return HUMAN / f"labels_{slug}.json"


def load_labels(slug: str) -> dict:
    p = labels_path(slug)
    return json.loads(p.read_text()) if p.exists() else {"rater": slug, "labels": {}}


class Mark(BaseModel):
    t_s: float
    area: str
    pressed_s: float | None = None


class Save(BaseModel):
    rater: str
    clip: str
    marks: list[Mark]
    score_1_10: int
    note: str = ""


@router.get("/clips")
def clips(rater: str):
    slug = rater_slug(rater)
    done = load_labels(slug)["labels"]
    return {"rater": slug, "areas": AREAS, "clips": [{"clip": n, "done": n in done} for n in order_for(slug)]}


@router.get("/audio/{name}")
def audio(name: str):
    p = CLIPS / Path(name).name
    if p.suffix.lower() not in EXTS or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@router.get("/labels/{rater}")
def get_labels(rater: str):
    return load_labels(rater_slug(rater))


@router.post("/save")
def save(req: Save):
    slug = rater_slug(req.rater)
    if req.clip not in clip_names():
        raise HTTPException(404, "unknown clip")
    if not 1 <= req.score_1_10 <= 10:
        raise HTTPException(400, "The overall score is 1 to 10.")
    if any(m.area not in AREAS for m in req.marks):
        raise HTTPException(400, "Every mark needs an area.")
    d = load_labels(slug)
    d["labels"][req.clip] = {"clip": req.clip, "rater": slug, "marks": [{"t_s": round(m.t_s, 3), "area": m.area, **({"pressed_s": round(m.pressed_s, 3)} if m.pressed_s is not None else {})} for m in sorted(req.marks, key=lambda m: m.t_s)],
                             "score_1_10": req.score_1_10, "note": req.note.strip(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    labels_path(slug).write_text(json.dumps(d, indent=1) + "\n")
    return {"saved": req.clip, "n_done": len(d["labels"])}


# ---------------------------------------------------------------- A5.3: listening confirmation of engine flags no human mark matched
class Verdict(BaseModel):
    reviewer: str
    flag: str                  # "<clip>@<start_s>"
    verdict: str               # agree | disagree | unsure


@router.get("/review/flags")
def review_flags(reviewer: str):
    slug = rater_slug(reviewer)
    p = ROOT_RESULTS / "human_unmatched_flags.json"
    if not p.exists():
        raise HTTPException(404, "Run eval/human_agreement.py first.")
    flags = json.loads(p.read_text())
    mine = json.loads((HUMAN / f"review_{slug}.json").read_text()) if (HUMAN / f"review_{slug}.json").exists() else {}
    return {"reviewer": slug, "flags": [{**f, "verdict": mine.get(f["flag"])} for f in flags]}


@router.post("/review/verdict")
def review_verdict(v: Verdict):
    slug = rater_slug(v.reviewer)
    if v.verdict not in ("agree", "disagree", "unsure"):
        raise HTTPException(400)
    p = HUMAN / f"review_{slug}.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[v.flag] = v.verdict
    p.write_text(json.dumps(d, indent=1) + "\n")
    return {"n": len(d)}


ROOT_RESULTS = Path(__file__).resolve().parent.parent / "results"
