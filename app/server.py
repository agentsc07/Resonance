"""Flawline web app. `python -m app.server` (or `make app`) serves the UI on http://localhost:8501.
LAB=1 adds the lab pages (Baselines, Alterations review, Pilot gate); they run as the Streamlit lab app, started alongside and linked from the nav."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import human as H
from . import retake as RT
from . import service as S

STATIC = Path(__file__).resolve().parent / "static"
LAB = os.environ.get("LAB") == "1"
LAB_PORT = int(os.environ.get("LAB_PORT", "8766"))
HUMAN = LAB or os.environ.get("HUMAN") == "1"       # the human-listening pages (/label, /review) are lab-only
BRAND = json.loads((Path(__file__).resolve().parent / "brand.json").read_text())      # product name + tagline: the only place the name is set (dataset stays "Flawline")
app = FastAPI(title=BRAND["product"], docs_url=None, redoc_url=None)


class AnalyseReq(BaseModel):
    source: str                       # clip | upload
    id: str
    mode: str = "free"
    genre: str | None = None
    no_fluency: bool = False
    experimental: bool = False
    rubric: str | None = None
    text_id: str | None = None
    transcript: str | None = None
    reference_id: str | None = None


@app.get("/api/meta")
def meta():
    return {**S.catalog(), "brand": BRAND, "lab": LAB, "lab_url": f"http://localhost:{LAB_PORT}" if LAB else None}


@app.post("/api/analyse")
def analyse(req: AnalyseReq):
    try:
        return S.analyse(req.model_dump())
    except (ValueError, KeyError, StopIteration) as e:
        raise HTTPException(400, str(e) if not isinstance(e, StopIteration) else "unknown upload")
    except Exception as e:                                       # the engine failed on this clip: say so instead of a blank page
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 60_000_000:
        raise HTTPException(413, "File too large (limit 60 MB).")
    try:
        return S.save_upload(data, file.filename or "audio.wav")
    except Exception:
        raise HTTPException(400, "Could not read this audio file. Try WAV, FLAC, MP3 or M4A.")


@app.get("/api/retake/prepare")
def retake_prepare(key: str, region: int):
    try:
        return S.retake_prepare(key, region)
    except (ValueError, KeyError, IndexError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/retake")
async def retake(key: str, region: int, file: UploadFile = File(...)):
    data = await file.read()
    try:
        return S.retake_score(key, region, data)
    except (ValueError, KeyError, IndexError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/audio/{kind}/{ident}.wav")
def audio(kind: str, ident: str):
    if kind not in ("clip", "upload"):
        raise HTTPException(404)
    try:
        return FileResponse(S.audio_file(kind, ident), media_type="audio/wav")
    except Exception:
        raise HTTPException(404)


@app.get("/api/truth/{clip_id}")
def truth(clip_id: str):
    return S.truth(clip_id)


@app.get("/api/pair/{i}")
def pair(i: int):
    return S.pair(i)


@app.get("/api/dataset")
def dataset():
    return S.dataset()


@app.get("/api/about")
def about():
    return S.about()


if HUMAN:
    app.include_router(H.router)

    @app.get("/label")
    def label_page():
        return FileResponse(STATIC / "label.html", headers={"Cache-Control": "no-store"})

    @app.get("/review")
    def review_page():
        return FileResponse(STATIC / "review.html", headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8501"))
    lab = None
    if LAB:
        lab = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py"),
                                "--server.headless", "true", "--server.port", str(LAB_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Lab pages: http://localhost:{LAB_PORT}")
    if HUMAN:
        print(f"Human listening pages: http://localhost:{port}/label  and  http://localhost:{port}/review")
    try:
        print(f"{BRAND['product']}: http://localhost:{port}")
        uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")
    finally:
        if lab:
            lab.terminate()


if __name__ == "__main__":
    main()
