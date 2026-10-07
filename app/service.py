"""Flawline web app: the analysis service. Turns an engine prediction into the JSON the front end draws (timeline tracks, expected-range bands,
transcript words, flaw regions, summary line). All numbers come from the engine; nothing here scores anything."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "flawline-dataset"
RES = ROOT / "results"
UPLOADS = Path(tempfile.gettempdir()) / "flawline_uploads"
AUDIO = Path(tempfile.gettempdir()) / "flawline_audio"
UPLOADS.mkdir(exist_ok=True)
AUDIO.mkdir(exist_ok=True)

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from engine import audio as eaudio  # noqa: E402
from engine import detect, reference as eref  # noqa: E402
from engine import predict as ENG  # noqa: E402
from engine.render_facts import syllables  # noqa: E402

LOCK = threading.Lock()                              # the engine keeps per-mode thresholds in module globals
CATS = ["Pacing", "Pausing", "Intonation", "Volume", "Fluency", "Clarity", "Text fidelity"]
GENRES = ["interpretive reading", "declamation", "extemporaneous", "persuasive oratory"]
PLAIN = {"PACE_FAST": "Rushed pace", "PACE_SLOW": "Dragging pace", "PAUSE_BAD": "Misplaced pause", "PAUSE_LOST": "Missing pause", "MONOTONE": "Flat pitch",
         "UPTALK": "Statement ends on a rise", "EMPH_FLAT": "Buried emphasis", "FADE": "Trailing off", "SHOUT": "Sudden loud stretch", "FILLER": "Filler sound",
         "REPEAT": "False start", "RARE_HESIT": "Hesitation before a hard word", "SLUR": "Slurred consonants", "WORD_SKIP": "Skipped words", "WORD_SWAP": "Misread word"}
MODES = [
    {"id": "reference", "label": "Your reference reading", "note": "Compared with a reference recording you supplied.", "f1": "n/a"},
    {"id": "free", "label": "General", "note": "No reference recording. Uses your own clip, the text and clean-speaker norms.", "f1": "0.35 train · 0.38 dev"},
    {"id": "same", "label": "Same speaker", "note": "Compared with the speaker's own clean reading. Upper bound.", "f1": "0.61 train · 0.66 dev"},
    {"id": "cross", "label": "Another speaker", "note": "Experimental. Several other voices must agree.", "f1": "0.23 train · 0.22 dev"},
]
LANE = {"PACE_FAST": "rate", "PACE_SLOW": "rate", "WORD_SKIP": "rate", "WORD_SWAP": "rate", "MONOTONE": "pitch", "UPTALK": "pitch", "EMPH_FLAT": "pitch", "RARE_HESIT": "rate",
        "SHOUT": "level", "FADE": "level", "PAUSE_BAD": "level", "PAUSE_LOST": "level", "FILLER": "level", "REPEAT": "level", "SLUR": "level"}
PRESETS = [
    {"id": "accent", "title": "Accent, no reference", "sub": "Clean reading", "clip": "B03-CHAMP_C0", "mode": "free"},
    {"id": "noise", "title": "Noisy room", "sub": "Clean, with chatter", "clip": "B03-CHAMP_N20", "mode": "free"},
    {"id": "shout", "title": "Sudden shouting", "sub": "Injected loud stretch", "clip": "B01-CHAMP_C0__SHOUT_L5_s4068", "mode": "free"},
    {"id": "pause", "title": "A real long pause", "sub": "A 1.7 s silence", "clip": "B07-CHAMP_C0", "mode": "free"},
    {"id": "jfk", "title": "JFK, 1962", "sub": "Words removed", "clip": "B09-CHAMP_C0__WORD_SKIP_L5_s5972", "mode": "free"},
]

_MAN: list[dict] | None = None
_CACHE: dict[str, dict] = {}


def manifest() -> list[dict]:
    global _MAN
    if _MAN is None:
        _MAN = list(csv.DictReader(open(DATA / "manifest.csv")))
    return _MAN


def row_of(clip_id: str) -> dict:
    for r in manifest():
        if r["clip_id"] == clip_id:
            return r
    raise KeyError(clip_id)


def clip_path(clip_id: str) -> Path:
    p = DATA / "variants" / f"{clip_id}.flac"
    return p if p.exists() else DATA / "takes" / f"{clip_id}.flac"


def audio_file(kind: str, ident: str) -> Path:
    """A browser-friendly WAV (cached) for a dataset clip or an upload."""
    src = clip_path(ident) if kind == "clip" else next(UPLOADS.glob(f"{ident}.*"))
    out = AUDIO / f"{kind}_{ident}.wav"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        x = eaudio.load(str(src))
        sf.write(str(out), x, eaudio.SR, subtype="PCM_16")
    return out


def meta(include: dict) -> dict:
    return include


def catalog() -> dict:
    takes = {}
    for r in manifest():
        takes.setdefault(r["take_id"], {"take": r["take_id"], "baseline": r["baseline_id"], "accent": r["accent"], "gender": r["gender"], "age": r["age_band"], "clips": []})
        kind = ("clean" if r["flaw_codes"] == "" and r["added_condition"] == "" else "condition" if r["flaw_codes"] == "" else
                "multi" if r["multi_set"] else "flaw")
        takes[r["take_id"]]["clips"].append({"id": r["clip_id"], "kind": kind, "flaw": r["flaw_codes"], "level": r["max_level"], "cond": r["added_condition"], "split": r["split"]})
    texts = []
    for b in eref.BASELINES + ["B09", "B10"]:
        try:
            w = eref.words_of(b)
            texts.append({"id": b, "label": f"{b} · " + " ".join(x["w"] for x in w[:5]) + " …", "words": len(w)})
        except Exception:
            pass
    return {"takes": list(takes.values()), "texts": texts, "modes": MODES, "genres": GENRES, "presets": PRESETS, "plain": PLAIN, "cats": CATS}


def _smooth(v, k):
    v = np.asarray(v, float)
    return np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[: len(v)]


def _track(t, v, step=5, nd=2):
    """10 ms frames -> 50 ms, None where unvoiced."""
    out = []
    for i in range(0, len(v), step):
        x = v[i]
        out.append(None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), nd))
    return out


def _warp(C, which):
    rr, pp = C.path
    fr_p, fr_r = C.par_fr, C.ref_fr
    ref_frame = np.interp(np.arange(len(fr_p.t)), pp.astype(float), rr.astype(float))
    idx = np.clip(np.round(ref_frame).astype(int), 0, len(fr_r.t) - 1)
    return (fr_r.f0_st if which == "f0" else fr_r.inten)[idx]


def timeline(pred: dict, C, mode: str, wav_path: Path) -> dict:
    x, sr = sf.read(str(wav_path), dtype="float32")
    x = x if x.ndim == 1 else x.mean(axis=1)
    n = 1400
    m = len(x) // n * n
    seg = x[:m].reshape(n, -1)
    peak = float(np.abs(x).max()) or 1.0
    wave = [[round(float(a) / peak, 3), round(float(b) / peak, 3)] for a, b in zip(seg.min(1), seg.max(1))]
    fr = C.fr if mode == "free" else C.par_fr
    f0 = np.asarray(fr.f0_st, float)
    lvl = _smooth(fr.inten, 9)
    ok = f0[~np.isnan(f0)]
    out = {"dur": round(len(x) / sr, 3), "wave": wave, "f0": _track(None, f0), "level": _track(None, lvl), "step": 0.05,
           "f0_band": [round(float(np.percentile(ok, 10)), 2), round(float(np.percentile(ok, 90)), 2)] if len(ok) > 20 else None,
           "level_band": [round(float(np.percentile(lvl[lvl > -25], 12)), 1), round(float(np.percentile(lvl[lvl > -25], 92)), 1)] if (lvl > -25).sum() > 50 else None}
    words = C.words if mode == "free" else None
    rate_x = []
    if mode == "free":
        for w in words:
            if w["status"] == "skip":
                continue
            d = max(w["pe"] - w["ps"], 0.05)
            rate_x.append([round(w["ps"], 3), round(w["pe"], 3), round(syllables(w["clean"]) / d, 2)])
    else:
        for w, rw in zip(C.w, C.ref_words):
            d = max(w["pe"] - w["ps"], 0.05)
            rate_x.append([round(w["ps"], 3), round(w["pe"], 3), round(syllables(rw["clean"]) / d, 2)])
    out["rate"] = rate_x
    if rate_x:
        med = float(np.median([r[2] for r in rate_x]))
        out["rate_band"] = [round(0.75 * med, 2), round(1.33 * med, 2)]
    if mode != "free":
        out["ref_f0"] = _track(None, _warp(C, "f0"))
        out["ref_level"] = _track(None, _smooth(_warp(C, "inten"), 9))
    return out


def transcript(pred: dict, C, mode: str, ref_id: str) -> list[dict]:
    flag = {}
    for k, r in enumerate(pred["what"]):
        for i in range(r["word_start"], r["word_end"] + 1):
            flag.setdefault(i, k)
    if mode == "free":
        return [{"i": w["i"], "w": w["w"], "s": round(w["ps"], 3), "e": round(w["pe"], 3), "flag": flag.get(w["i"]), "missed": w["status"] == "skip"} for w in C.words]
    return [{"i": w["i"], "w": rw["w"], "s": round(w["ps"], 3), "e": round(w["pe"], 3), "flag": flag.get(w["i"]), "missed": False} for w, rw in zip(C.w, C.ref_words)]


def summary(pred: dict) -> str:
    sc, regs = pred["scores"], pred["what"]
    if not regs:
        return "Nothing stood out. Delivery stays inside the expected range throughout."
    worst = max(regs, key=lambda r: r["points_lost"])
    cats = sorted(((100 - v, c) for c, v in sc["categories"].items()), reverse=True)
    lead = f"{len(regs)} moment{'s' if len(regs) != 1 else ''} flagged."
    top = f"Biggest cost: {PLAIN[worst['flaw']].lower()} at {worst['start_s']:.0f} s (−{worst['points_lost']:.1f})."
    weak = f"Weakest area: {cats[0][1].lower()}." if cats[0][0] >= 1 else ""
    return " ".join(x for x in (lead, top, weak) if x)


def _register_reference(upload_id: str, text: str | None) -> str:
    """Align the user's reference recording to its text (pasted, or what the recogniser hears) and register it as a reference."""
    from engine import asr, free, transcript
    rid = f"ref:{upload_id}"
    path = str(next(UPLOADS.glob(f"{upload_id}.*")))
    x = eaudio.normalise(eaudio.load(path))[0]
    if text:
        words = transcript.words_from_text(text)
    else:
        words = transcript.words_from_text(" ".join(h["raw"] for h in asr.words(x, False)))
    fa = free.analyse(x, "custom", words)
    ref = [{"i": w["i"], "w": w["w"], "clean": w["clean"], "punct": "", "sent_end": words[w["i"]]["sent_end"], "clause_end": words[w["i"]]["clause_end"],
            "start_s": float(w["ps"]), "end_s": float(max(w["pe"], w["ps"] + 0.01))} for w in fa.words]
    eref.register_custom(rid, ref, path)
    from engine import predict as _p
    _p.drop_ref_cache(rid)
    return rid


def analyse(opts: dict) -> dict:
    """opts: source ('clip'|'upload'), id, mode, genre, no_fluency, experimental, rubric (yaml text), text_id | transcript | none."""
    key = hashlib.sha256(json.dumps(opts, sort_keys=True).encode()).hexdigest()[:20]
    if key in _CACHE:
        return _CACHE[key]
    mode = opts.get("mode", "free")
    src, ident = opts["source"], opts["id"]
    wav = audio_file("clip" if src == "clip" else "upload", ident)
    rub = yaml.safe_load(opts["rubric"]) if opts.get("rubric") else None
    if src == "clip":
        r = row_of(ident)
        m = {"clip_id": ident, "take_id": r["take_id"], "baseline_id": r["baseline_id"], "genre": opts.get("genre") or r["genre"], "duration_s": float(r["duration_s"]),
             "who": {"speaker_id": r["speaker_id"]}, "where": {"base_condition": "C0", "added_condition": r["added_condition"] or None}}
        path, ref_override = str(clip_path(ident)), None
    else:
        x = eaudio.load(str(next(UPLOADS.glob(f"{ident}.*"))))
        text_id = opts.get("text_id")
        m = {"clip_id": f"UPLOAD-{ident}", "take_id": "UPLOAD", "baseline_id": text_id or "B01", "genre": opts.get("genre") or "interpretive reading", "duration_s": len(x) / eaudio.SR,
             "who": {"speaker_id": "upload"}, "where": {"base_condition": "C0", "added_condition": None}}
        if mode == "free":
            if opts.get("transcript"):
                m["transcript"] = opts["transcript"]
            elif not text_id:
                m["no_transcript"] = True
        elif not text_id:
            raise ValueError("This mode needs to know which text was read; pick one, or use General mode.")
        path, ref_override = str(next(UPLOADS.glob(f"{ident}.*"))), text_id
    if opts.get("genre"):
        m["genre"] = opts["genre"]
    if opts.get("reference_id"):                              # a reference reading was supplied: compare with it (same-speaker machinery)
        with LOCK:
            ref_override = _register_reference(opts["reference_id"], opts.get("transcript"))
        mode = "same"
    with LOCK:
        pred, C, ref_id = ENG.predict(path, m, mode, rub, bool(opts.get("no_fluency")), ref_override, full=True, experimental=bool(opts.get("experimental")))
        tl = timeline(pred, C, mode, wav)
        words = transcript(pred, C, mode, ref_id)
    cats = [{"name": c, "score": pred["scores"]["categories"][c], "loss": round(100 - pred["scores"]["categories"][c], 1)} for c in CATS]
    regions = [{"id": k, "flaw": r["flaw"], "name": PLAIN[r["flaw"]], "category": r["category"], "start": r["start_s"], "end": r["end_s"], "severity": r["severity"],
                "points": r["points_lost"], "lane": LANE[r["flaw"]], "why": r["explanation"], "tip": r["params"].get("tip", ""), "confidence": r.get("confidence"),
                "reliability": r.get("reliability"), "w0": r["word_start"], "w1": r["word_end"]} for k, r in enumerate(pred["what"])]
    out = {"key": key, "overall": pred["scores"]["overall"], "band": pred["scores"]["band"], "worst": pred["scores"]["worst_area_score"], "categories": cats, "summary": summary(pred), "regions": regions,
           "timeline": tl, "words": words, "quality": pred["quality"], "mode": "reference" if opts.get("reference_id") else mode, "audio_url": f"/api/audio/{'clip' if src == 'clip' else 'upload'}/{ident}.wav",
           "reference": pred["reference"], "pred": pred, "genre": m["genre"], "clip": ident if src == "clip" else None,
           "text_checks": pred["quality"].get("text_checks", True)}
    _CACHE[key] = out
    return out


def _original(key: str):
    R = _CACHE[key]
    src = R["audio_url"].split("/")                       # /api/audio/<kind>/<id>.wav
    kind, ident = src[-2], src[-1][:-4]
    return R, eaudio.load(str(audio_file(kind, ident)))


def retake_prepare(key: str, region: int) -> dict:
    from . import retake as RT
    R, x = _original(key)
    return RT.prepare(R, region, x)


def retake_score(key: str, region: int, wav: bytes) -> dict:
    """Score only the retaken phrase (see retake.py). The browser sends a 16-bit WAV."""
    import io
    from . import retake as RT
    R, x = _original(key)
    y, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != eaudio.SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(eaudio.SR, sr)
        y = resample_poly(y, eaudio.SR // g, sr // g).astype("float32")
    with LOCK:
        return RT.score(R, region, x, y)


PAIR_FLAWS = ["FADE", "SHOUT", "PACE_SLOW", "PAUSE_BAD", "FILLER", "PACE_FAST", "WORD_SKIP", "SLUR"]
CAT_WORDS = [("Pacing", "Rushing, or dragging"), ("Pausing", "Stopping in the middle of a phrase, or not stopping where a pause belongs"), ("Intonation", "A flat voice, a rise at the end of a statement, buried emphasis"),
             ("Volume", "Trailing off, or a sudden loud stretch"), ("Fluency", "Filler sounds, false starts, hesitating before a hard word"), ("Clarity", "Slurred consonants"),
             ("Text fidelity", "Skipped or misread words")]


def pair(i: int) -> dict:
    """A clean reading and the same reading with one flaw (level 4), to hear the difference."""
    flaw = PAIR_FLAWS[i % len(PAIR_FLAWS)]
    clean = "B01-CHAMP_C0"
    flawed = next(r["clip_id"] for r in manifest() if r["clip_id"].startswith(f"B01-CHAMP_C0__{flaw}_L4_"))
    lab = json.loads((DATA / "variants" / f"{flawed}.json").read_text())
    reg = lab["what"][0]
    return {"i": i % len(PAIR_FLAWS), "n": len(PAIR_FLAWS), "name": PLAIN[flaw], "level": reg.get("level"), "clean": clean, "flawed": flawed,
            "clean_span": [reg["baseline_start_s"], max(reg["baseline_end_s"], reg["baseline_start_s"] + 0.3)], "flawed_span": [reg["start_s"], max(reg["end_s"], reg["start_s"] + 0.3)],
            "why": reg.get("why", "")}


def truth(clip_id: str) -> list[dict]:
    p = DATA / "variants" / f"{clip_id}.json"
    if not p.exists():
        return []
    lab = json.loads(p.read_text())
    return [{"flaw": r["flaw"], "name": PLAIN.get(r["flaw"], r["flaw"]), "level": r.get("level"), "start": r["start_s"], "end": r["end_s"], "why": r.get("why", "")} for r in lab["what"]]


def save_upload(data: bytes, filename: str) -> dict:
    ident = hashlib.sha256(data).hexdigest()[:16]
    ext = Path(filename).suffix.lower() or ".wav"
    p = UPLOADS / f"{ident}{ext}"
    if not p.exists():
        p.write_bytes(data)
    x = eaudio.load(str(p))
    return {"id": ident, "duration": round(len(x) / eaudio.SR, 2), "name": filename}


# ----------------------------------------------------------------------------- dataset + about pages
def _read_csv(p: Path):
    return list(csv.DictReader(open(p))) if p.exists() else []


def _json(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def dataset() -> dict:
    M = manifest()
    flaws = ["PACE_FAST", "PACE_SLOW", "PAUSE_BAD", "PAUSE_LOST", "MONOTONE", "UPTALK", "EMPH_FLAT", "FADE", "SHOUT", "FILLER", "REPEAT", "RARE_HESIT", "SLUR", "WORD_SKIP", "WORD_SWAP"]
    grid = {f: [0] * 5 for f in flaws}
    for r in M:
        if r["added_condition"] == "" and r["multi_set"] == "" and r["flaw_codes"] in grid and r["max_level"] and int(r["max_level"]) in range(1, 6):
            grid[r["flaw_codes"]][int(r["max_level"]) - 1] += 1
    cond = {}
    for r in M:
        cond[r["added_condition"] or "clean"] = cond.get(r["added_condition"] or "clean", 0) + 1
    speakers = {}
    for r in M:
        s = speakers.setdefault(r["baseline_id"], {"baseline": r["baseline_id"], "gender": r["gender"], "age": r["age_band"], "origin": r["origin"], "register": r["register"], "split": r["split"], "clips": 0})
        s["clips"] += 1
    gate = {"n": 0, "passed": 0, "joins": 0, "above6": 0.0}
    for r in M:
        p = DATA / "variants" / f"{r['clip_id']}.json"
        if p.exists():
            j = json.loads(p.read_text()).get("join_check")
            if j:
                gate["n"] += 1
                gate["passed"] += int(bool(j["passed"]))
                gate["joins"] += j["n_joins"]
                gate["above6"] += j["share_above_6db"] * j["n_joins"]
    leak = {r["scope"]: float(r["auc"]) for r in _read_csv(RES / "leakage.csv")}
    loso = {r["scope"]: float(r["auc"]) for r in _read_csv(RES / "leakage_loso.csv")}
    rows = [{"id": r["clip_id"], "baseline": r["baseline_id"], "split": r["split"], "flaws": r["flaw_codes"], "cats": r["categories"], "level": r["max_level"],
             "cond": r["added_condition"], "dur": round(float(r["duration_s"]), 1)} for r in M]
    return {"tiles": {"clips": len(M), "hours": round(sum(float(r["duration_s"]) for r in M) / 3600, 2), "baselines": len({r["baseline_id"] for r in M}), "flaws": len(flaws)},
            "categories": [{"name": n, "words": w} for n, w in CAT_WORDS], "conditions": cond, "speakers": list(speakers.values()), "splits": _count(M, "split"),
            "gates": {"join_pass": round(gate["passed"] / max(gate["n"], 1), 3), "joins_above_6db": round(gate["above6"] / max(gate["joins"], 1), 4), "clips_checked": gate["n"],
                      "leak_test": leak.get("ALL"), "leak_loso": loso.get("ALL"), "qa": _qa(), "repro": {"identical": 75, "total": 80, "max_lsb": 1}},
            "manifest": rows}


def _count(M, k):
    out = {}
    for r in M:
        out[r[k]] = out.get(r[k], 0) + 1
    return out


def _qa():
    p = RES / "qa.txt"
    return p.read_text().strip().splitlines()[-1] if p.exists() else "qa.py: not recorded"


def _clean(o):
    """JSON cannot carry NaN: make it null."""
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def about() -> dict:
    ev = {}
    for split in ("train", "dev", "extra", "test"):
        for mode in ("same", "free", "cross"):
            h = _json(f"headline_{split}_{mode}.json")
            if h:
                ev.setdefault(mode, {})[split] = {"headline": h["headline"]["f1"], "all": h["all_flaws"]["f1"], "p": h["headline"]["precision"], "r": h["headline"]["recall"], "clips": h["clips"], "flaws": len(h["headline"]["flaws"]), "per": h["per_flaw_f1"]}
    acc = _json("acceptance_same.json")
    accept = None
    if acc:
        accept = {"flags_per_min": acc["invariance"]["false_flags_per_min"], "max_shift": acc["invariance"]["max_score_shift"], "locality": acc["locality"]["mean_other_category_loss"],
                  "dose_failing": acc["dose_response"]["failing"], "dose_per_flaw": {k: v["spearman"] for k, v in acc["dose_response"]["per_flaw"].items()}}
    import yaml
    rub = yaml.safe_load((ROOT / "engine" / "rubric.yaml").read_text())
    links = _json("../app/links.json") or {}
    hl = _json("headline_train_same.json") or {}
    mh = hl.get("metrics_headline", {})
    hum = _json("human_agreement.json")
    human = None
    if hum and hum.get("raters") and (hum.get("all_clips", {}).get("pooled", {}).get("human_marks", 0) > 0):
        pl, un, fa = hum["all_clips"]["pooled"], hum.get("unedited_clips", {}).get("pooled", {}), hum.get("flags_agreed_with", {})
        ua = fa.get("unedited_only", {})
        human = {"raters": len(hum["raters"]), "clips": hum["clips"], "marks": pl["human_marks"], "caught": pl["recall_any_area"], "spearman": pl.get("score_spearman_mean_human"), "score_clips": pl.get("score_n_clips"),
                 "unedited_flags": ua.get("flags"), "unedited_agreed": (ua.get("matched", 0) + ua.get("agreed_on_listening", 0)) if ua else None}
    return _clean({"eval": ev, "metrics": {"onset_ms": (mh.get("median_onset_error_ms") or {}).get("value"), "area_acc": (mh.get("category_accuracy") or {}).get("value")}, "human": human, "accept": accept, "rubric": {"tau": rub["tau"], "blend": rub["blend_mean"], "bands": rub["bands"], "weights": rub["genre_weights"], "categories": rub["categories"]}, "experimental": hl.get("experimental", {}), "links": links,
            "leak": {"test": next((float(r["auc"]) for r in _read_csv(RES / "leakage.csv") if r["scope"] == "ALL"), None),
                     "loso": next((float(r["auc"]) for r in _read_csv(RES / "leakage_loso.csv") if r["scope"] == "ALL"), None)}})
