"""Generate variants + label JSONs, then rebuild manifest.csv and checksums.sha256.

  python make_dataset.py --pilot B01-CHAMP        75 flaw variants + 5 multi-flaw + 6 clean-with-condition
  python make_dataset.py --l3                     every baseline: 15 flaws at L3 + 5 multi-flaw (review sheet)
  python make_dataset.py --grid B02-CHAMP ...     full 15 flaws x 5 levels + 5 multi for the given takes
  python make_dataset.py --invariance             clean take x 6 conditions, every CHAMP take
  python make_dataset.py --flawed-cond 40         N random L3 variants re-recorded under one random condition
  python make_dataset.py --manifest               rebuild manifest.csv + checksums.sha256 only
  python make_dataset.py --check-repro B01-CHAMP  regenerate one clip, compare SHA-256
Existing outputs are kept (idempotent); delete a file to regenerate it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time

import numpy as np

import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'schema'))
from validate import validate as validate_label  # noqa: E402
from common import (BASELINES, GENERATOR_VERSION, ROOT, SR, TAKES, VARIANTS, dump_json, load_baselines, load_config,
                    read_audio, write_audio)
from conditions import CFG as COND_CFG
from conditions import apply as apply_condition
from flaws import FLAW_CODES, Ctx, make_flawed, seed_for

CFGALL = load_config()
CONDS = list(COND_CFG)
_CTX: dict[str, Ctx] = {}


def ctx_for(take: str) -> Ctx:
    if take not in _CTX:
        _CTX[take] = Ctx(take)
    return _CTX[take]


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def who_where(ctx: Ctx, cond: str | None, params: dict | None):
    s = ctx.source
    return (
        {"speaker_id": ctx.take_id.split("-")[1], "age_band": s.get("age_band"), "accent": s.get("accent"),
         "gender": s.get("gender"), "fluency_profile": "typical", "synthetic_voice": s.get("status") == "placeholder_tts",
         "origin": s.get("origin"), "register": s.get("register")},
        {"base_condition": "C0", "added_condition": cond, "params": params or {}},
    )


def job_clip_id(take: str, cond: str | None, tag: str | None, level: int | None) -> str:
    where = cond or "C0"
    if tag is None:
        return f"{take}_{where}"
    return f"{take}_{where}__{tag}_L{level}_s{seed_for(take, tag):04d}"


def run_job(job: dict) -> str:
    take, tag, level, cond = job["take"], job["tag"], job["level"], job["cond"]
    cid = job_clip_id(take, cond, tag, level)
    if not job.get("force") and (VARIANTS / f"{cid}.json").exists():
        return f"skip {cid}"
    ctx = ctx_for(take)
    if tag is None:                                   # clean + condition (invariance)
        y, what, seed, genre_tag = ctx.x.copy(), [], 0, None
    else:
        flaws = job["flaws"]
        seed = seed_for(take, job.get('seed_tag', tag))      # PURE_* clips reuse the bundled flaw's seed: same anchors
        try:
            y, what = make_flawed(ctx, flaws, seed, pure=job.get('pure', False))
        except RuntimeError as e:                    # e.g. no word with Zipf < 3.5 in this script
            for ext in (".flac", ".json"):               # never leave a stale clip from an earlier build behind
                (VARIANTS / f"{cid}{ext}").unlink(missing_ok=True)
            return f"n/a  {cid} :: {e}"
    cp = None
    if cond:
        y, cp = apply_condition(y, cond, np.random.default_rng(seed_for(take, cond) + 7), take)
    who, where = who_where(ctx, cond, cp)
    write_audio(VARIANTS / f"{cid}.flac", y)
    label = {
        "schema_version": "1.1.0", "clip_id": cid, "take_id": take, "baseline_id": take.split("-")[0],
        "genre": ctx.source.get("genre"), "who": who, "where": where, "what": what,
        "multi_set": job.get("multi_set"), "duration_s": round(len(y) / SR, 4), "seed": seed,
        "generator_version": GENERATOR_VERSION, "labels_from": "generator",
        "align_source": ctx.align_source,
        "join_check": getattr(ctx, "last_join", None) if tag is not None else None,
    }
    validate_label(label)
    dump_json(VARIANTS / f"{cid}.json", label)
    return f"ok   {cid}"


def jobs_flaw_grid(take, levels, multi=True, cond=None):
    js = [{"take": take, "tag": c, "level": l, "flaws": [(c, l)], "cond": cond} for c in FLAW_CODES for l in levels]
    if multi:
        L = CFGALL["multi_flaw_level"]
        for n, fl in enumerate(CFGALL["multi_flaw_sets"], 1):
            js.append({"take": take, "tag": f"MULTI{n}", "level": L, "flaws": [(c, L) for c in fl], "cond": cond,
                       "multi_set": fl})
    return js


def jobs_pure(take):
    """Ablation slice: the same flaws and anchors rendered with a SINGLE cue (v1.1 'pure' slice; default clips use natural bundles)."""
    return [{"take": take, "tag": f"PURE_{c}", "seed_tag": c, "level": l, "flaws": [(c, l)], "cond": None, "pure": True}
            for c in FLAW_CODES for l in range(1, 6)]


def jobs_invariance(take):
    return [{"take": take, "tag": None, "level": None, "flaws": [], "cond": c} for c in CONDS]


def champ_takes():
    return [f"{b['id']}-CHAMP" for b in load_baselines()]


def log_na(msg: str):
    """Flaws that cannot be placed on a take are recorded, never silently dropped or faked."""
    with open(ROOT / "not_applicable.txt", "a") as f:
        f.write(msg[5:] + "\n")


def run_all(jobs, procs):
    t0 = time.time()
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            for i, msg in enumerate(pool.imap_unordered(run_job, jobs, chunksize=2), 1):
                print(f"[{i}/{len(jobs)}] {msg}", flush=True)
                if msg.startswith("n/a"):
                    log_na(msg)
    else:
        for i, j in enumerate(jobs, 1):
            m = run_job(j)
            print(f"[{i}/{len(jobs)}] {m}", flush=True)
            if m.startswith("n/a"):
                log_na(m)
    print(f"done {len(jobs)} jobs in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- manifest
COLS = ["clip_id", "take_id", "baseline_id", "genre", "speaker_id", "age_band", "accent", "gender", "fluency_profile",
        "base_condition", "added_condition", "flaw_codes", "categories", "max_level", "n_regions", "duration_s", "split",
        "labels_from", "licence", "sha256", "origin", "register", "synthetic_voice", "multi_set"]


def split_of(baseline_id: str) -> str:
    s = CFGALL["splits"]
    return "test" if baseline_id in s["test"] else "dev" if baseline_id in s["dev"] else "train"


def clean_label(take: str):
    ctx = ctx_for(take)
    who, where = who_where(ctx, None, None)
    return {"clip_id": f"{take}_C0", "take_id": take, "baseline_id": take.split("-")[0], "genre": ctx.source.get("genre"),
            "who": who, "where": where, "what": [], "duration_s": round(ctx.dur, 4), "labels_from": "generator"}, TAKES / f"{take}_C0.flac"


def build_manifest():
    rows = []
    entries = [clean_label(t) for t in champ_takes() if (TAKES / f"{t}_C0.flac").exists()]
    for jp in sorted(VARIANTS.glob("*.json")):
        entries.append((json.loads(jp.read_text()), jp.with_suffix(".flac")))
    sums = []
    for lab, audio in entries:
        w = lab["what"]
        who, where = lab["who"], lab["where"]
        rows.append({
            "clip_id": lab["clip_id"], "take_id": lab["take_id"], "baseline_id": lab["baseline_id"], "genre": lab["genre"],
            "speaker_id": who["speaker_id"], "age_band": who["age_band"], "accent": who["accent"], "gender": who["gender"],
            "fluency_profile": who["fluency_profile"], "base_condition": where["base_condition"],
            "added_condition": where["added_condition"] or "", "flaw_codes": "|".join(sorted({r["flaw"] for r in w})),
            "categories": "|".join(sorted({r["category"] for r in w})), "max_level": max([r["level"] for r in w], default=0),
            "n_regions": len(w), "duration_s": lab["duration_s"], "split": split_of(lab["baseline_id"]),
            "labels_from": lab.get("labels_from", "generator"), "licence": "CC0 (synthetic placeholder)" if who["synthetic_voice"] else "see LICENSES.md",
            "sha256": sha256(audio), "origin": who.get("origin", ""), "register": who.get("register", ""),
            "synthetic_voice": who["synthetic_voice"], "multi_set": "|".join(lab.get("multi_set") or []),
        })
        sums.append(f"{rows[-1]['sha256']}  {audio.relative_to(ROOT)}")
    with open(ROOT / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(rows)
    (ROOT / "checksums.sha256").write_text("\n".join(sums) + "\n")
    print(f"manifest.csv: {len(rows)} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", metavar="TAKE")
    ap.add_argument("--l3", action="store_true")
    ap.add_argument("--grid", nargs="*", metavar="TAKE")
    ap.add_argument("--invariance", action="store_true")
    ap.add_argument("--pure", metavar="TAKE", help="ablation slice: 15 flaws x 5 levels, single-cue rendering")
    ap.add_argument("--flawed-cond", type=int, metavar="N")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate even if the clip exists")
    ap.add_argument("--check-repro", metavar="TAKE")
    ap.add_argument("--procs", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    a = ap.parse_args()
    VARIANTS.mkdir(exist_ok=True)
    for t in champ_takes():                       # once, before any worker starts (no write races)
        if (TAKES / f"{t}_C0.flac").exists():
            Ctx(t).save_refined()
    jobs = []
    if a.pilot:
        jobs += jobs_flaw_grid(a.pilot, range(1, 6)) + jobs_invariance(a.pilot)
    if a.l3:
        for t in champ_takes():
            jobs += jobs_flaw_grid(t, [3])
    if a.grid:
        for t in a.grid:
            jobs += jobs_flaw_grid(t, range(1, 6))
    if a.pure:
        jobs += jobs_pure(a.pure)
    if a.invariance:
        for t in champ_takes():
            jobs += jobs_invariance(t)
    if a.flawed_cond:
        rng = np.random.default_rng(2026)
        for _ in range(a.flawed_cond):
            t = champ_takes()[int(rng.integers(0, len(champ_takes())))]
            c = FLAW_CODES[int(rng.integers(0, len(FLAW_CODES)))]
            jobs.append({"take": t, "tag": c, "level": 3, "flaws": [(c, 3)], "cond": CONDS[int(rng.integers(0, len(CONDS)))]})
    if a.check_repro:
        j = {"take": a.check_repro, "tag": "PACE_FAST", "level": 3, "flaws": [("PACE_FAST", 3)], "cond": "N20"}
        cid = job_clip_id(j["take"], "N20", "PACE_FAST", 3)
        run_job(j)
        h1 = sha256(VARIANTS / f"{cid}.flac")
        (VARIANTS / f"{cid}.flac").unlink()
        (VARIANTS / f"{cid}.json").unlink()
        _CTX.clear()
        run_job(j)
        h2 = sha256(VARIANTS / f"{cid}.flac")
        print("reproducible" if h1 == h2 else "MISMATCH", h1[:16], h2[:16])
        sys.exit(0 if h1 == h2 else 1)
    if jobs:
        # dedupe (an L3 grid overlaps the pilot grid)
        seen, uniq = set(), []
        for j in jobs:
            k = job_clip_id(j["take"], j["cond"], j["tag"], j["level"])
            if k not in seen:
                seen.add(k)
                uniq.append(j)
        for j in uniq:
            j["force"] = a.force
        run_all(uniq, a.procs)
    build_manifest()
