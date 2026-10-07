"""Human reference set: how well do the engine's flags and score agree with people listening to the same clips?
The engine runs in FREE (upload) mode on every clip (the text, when known, is passed like a pasted transcript); nothing is tuned on this set.
   python eval/human_agreement.py           -> results/human_agreement.json, results/human_agreement.txt, results/human_unmatched_flags.json
A mark and a flag agree when the mark falls within +-1.0 s of the flag's span. Every percentage is printed with its counts."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HUMAN = ROOT / "flawline-dataset" / "human"
CACHE = Path("/tmp/human_pred_cache")
WIN = 1.0
NAME = {"FADE": "Trailing off", "SHOUT": "Sudden loud stretch", "PACE_SLOW": "Slowing down", "PACE_FAST": "Rushing", "PAUSE_BAD": "Pause in the wrong place",
        "PAUSE_LOST": "Missing pause", "FILLER": "Filler sound", "WORD_SKIP": "Skipped word", "WORD_SWAP": "Misread word", "SLUR": "Slurred speech",
        "MONOTONE": "Flat delivery", "REPEAT": "Repeated word", "RARE_HESIT": "Hesitation", "UPTALK": "Rise at the end", "EMPH_FLAT": "Buried emphasis"}


def frac(a: int, b: int) -> str:
    return f"{a} of {b} ({100 * a / b:.0f}%)" if b else f"0 of 0 (n/a)"


def near(t: float, a: float, b: float, w: float = WIN) -> bool:
    return a - w <= t <= b + w


def engine_flags(clip: str) -> dict:
    from engine import audio as eaudio, predict as P
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{clip}.json"
    if f.exists():
        return json.loads(f.read_text())
    path = str(HUMAN / "clips" / clip)
    x = eaudio.load(path)
    texts = json.loads((HUMAN / "transcripts.json").read_text()) if (HUMAN / "transcripts.json").exists() else {}
    m = {"clip_id": "HUMAN-" + clip, "take_id": "HUMAN", "baseline_id": "B01", "genre": "interpretive reading", "duration_s": len(x) / eaudio.SR,
         "who": {"speaker_id": "human-set"}, "where": {"base_condition": "C0", "added_condition": None}}
    if texts.get(clip):
        m["transcript"] = texts[clip]
    else:
        m["no_transcript"] = True
    out = P.predict(path, m, "free")
    res = {"overall": out["scores"]["overall"], "band": out["scores"]["band"], "duration_s": m["duration_s"],
           "flags": [{"flaw": r["flaw"], "name": NAME.get(r["flaw"], r["flaw"]), "category": r["category"], "start_s": r["start_s"], "end_s": r["end_s"], "severity": r["severity"]} for r in out["what"]]}
    f.write_text(json.dumps(res))
    return res


def load_raters() -> dict[str, dict]:
    return {Path(p).stem[len("labels_"):]: json.loads(Path(p).read_text())["labels"] for p in sorted(glob.glob(str(HUMAN / "labels_*.json")))}


def eng_vs_human(E: dict, labs: dict, clips: list[str]) -> dict:
    """counts for one rater (or pooled via summation) over the given clips"""
    n_marks = rec_any = rec_same = n_flags = prec = 0
    for c in clips:
        if c not in labs:
            continue
        marks, flags = labs[c]["marks"], E[c]["flags"]
        for m in marks:
            n_marks += 1
            hit = [f for f in flags if near(m["t_s"], f["start_s"], f["end_s"])]
            rec_any += bool(hit)
            rec_same += any(f["category"] == m["area"] for f in hit)
        for f in flags:
            n_flags += 1
            prec += any(near(m["t_s"], f["start_s"], f["end_s"]) for m in marks)
    return {"human_marks": n_marks, "recall_any_area": rec_any, "recall_same_area": rec_same, "engine_flags": n_flags, "flags_near_a_human_mark": prec}


def human_vs_human(ref: dict, oth: dict, clips: list[str]) -> dict:
    n_ref = rec_any = rec_same = n_oth = prec = 0
    for c in clips:
        if c not in ref or c not in oth:
            continue
        for m in ref[c]["marks"]:
            n_ref += 1
            hit = [o for o in oth[c]["marks"] if abs(o["t_s"] - m["t_s"]) <= WIN]
            rec_any += bool(hit)
            rec_same += any(o["area"] == m["area"] for o in hit)
        for o in oth[c]["marks"]:
            n_oth += 1
            prec += any(abs(o["t_s"] - m["t_s"]) <= WIN for m in ref[c]["marks"])
    return {"reference_marks": n_ref, "recall_any_area": rec_any, "recall_same_area": rec_same, "other_marks": n_oth, "marks_near_a_reference_mark": prec}


def spear(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return round(float(spearmanr(x, y).statistic), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    clips = sorted(p.name for p in (HUMAN / "clips").iterdir() if p.suffix.lower() in {".wav", ".m4a", ".flac", ".mp3", ".ogg"})
    hidden = json.loads((HUMAN / "hidden_map.json").read_text()) if (HUMAN / "hidden_map.json").exists() else {}
    injected = {c: v["injected"] for c, v in hidden.items() if v["injected"]}
    raters = load_raters()
    E = {c: engine_flags(c) for c in clips}
    unedited = [c for c in clips if c not in injected]
    res: dict = {"window_s": WIN, "clips": len(clips), "unedited_clips": len(unedited), "hidden_injected_clips": len(injected), "raters": sorted(raters),
                 "labels": {r: len(v) for r, v in raters.items()}, "engine_mode": "free (upload)",
                 "definitions": {"recall_any_area": "share of human marks with an engine flag whose span is within 1.0 s of the mark",
                                 "recall_same_area": "same, and the engine flag names the same area as the mark",
                                 "precision": "share of engine flags within 1.0 s of a human mark by that rater",
                                 "score_agreement": "Spearman correlation between the engine's overall score and the human 1-10 score, one point per clip"}}
    lines = [f"Human reference set: {len(clips)} clips ({len(unedited)} unedited, {len(injected)} with hidden injected flaws), {len(raters)} rater(s): {', '.join(sorted(raters)) or 'none yet'}",
             f"Engine: reference-free upload mode, no tuning on this set. A mark and a flag agree when they are within {WIN:.1f} s.", ""]
    for scope, subset in (("all_clips", clips), ("unedited_clips", unedited)):
        res[scope] = {"per_rater": {}}
        lines.append(f"== Engine vs human, {scope.replace('_', ' ')}")
        tot: dict[str, int] = {}
        for r, labs in raters.items():
            c = eng_vs_human(E, labs, subset)
            xs = [E[k]["overall"] for k in subset if k in labs]
            ys = [labs[k]["score_1_10"] for k in subset if k in labs]
            c["score_spearman"], c["score_n_clips"] = spear(xs, ys), len(xs)
            res[scope]["per_rater"][r] = c
            for k, v in c.items():
                if isinstance(v, int):
                    tot[k] = tot.get(k, 0) + v
            lines.append(f"  {r}: recall any area {frac(c['recall_any_area'], c['human_marks'])}; same area {frac(c['recall_same_area'], c['human_marks'])}; "
                         f"precision {frac(c['flags_near_a_human_mark'], c['engine_flags'])}; score Spearman {c['score_spearman']} (n = {c['score_n_clips']} clips)")
        if raters:
            mean_h = {k: float(np.mean([l[k]["score_1_10"] for l in raters.values() if k in l])) for k in subset if any(k in l for l in raters.values())}
            pooled_sp = spear([E[k]["overall"] for k in mean_h], list(mean_h.values()))
            pooled = {**tot, "score_spearman_mean_human": pooled_sp, "score_n_clips": len(mean_h)}
            res[scope]["pooled"] = pooled
            lines.append(f"  POOLED: recall any area {frac(tot.get('recall_any_area', 0), tot.get('human_marks', 0))}; same area {frac(tot.get('recall_same_area', 0), tot.get('human_marks', 0))}; "
                         f"precision {frac(tot.get('flags_near_a_human_mark', 0), tot.get('engine_flags', 0))}; score Spearman vs mean human score {pooled_sp} (n = {len(mean_h)} clips)")
        lines.append("")

    if len(raters) >= 2:
        res["human_vs_human"] = {}
        lines.append("== Human vs human (rater A as reference for rater B)")
        for a, b in permutations(sorted(raters), 2):
            c = human_vs_human(raters[a], raters[b], clips)
            both = [k for k in clips if k in raters[a] and k in raters[b]]
            c["score_spearman"], c["score_n_clips"] = spear([raters[a][k]["score_1_10"] for k in both], [raters[b][k]["score_1_10"] for k in both]), len(both)
            res["human_vs_human"][f"{a}->{b}"] = c
            lines.append(f"  A={a}, B={b}: recall any area {frac(c['recall_any_area'], c['reference_marks'])}; same area {frac(c['recall_same_area'], c['reference_marks'])}; "
                         f"precision {frac(c['marks_near_a_reference_mark'], c['other_marks'])}; score Spearman {c['score_spearman']} (n = {len(both)} clips)")
        lines.append("")

    # hidden injected clips: injected flaws found by each human and by the engine (span +-1 s)
    res["hidden_injected"] = {"engine": {}, "raters": {}}
    n_inj = sum(len(v) for v in injected.values())
    eng_found = sum(any(near(f["start_s"], r["start_s"], r["end_s"]) or near((f["start_s"] + f["end_s"]) / 2, r["start_s"], r["end_s"]) or
                        near(r["start_s"], f["start_s"], f["end_s"]) for f in E[c]["flags"]) for c, rs in injected.items() for r in rs)
    res["hidden_injected"]["engine"] = {"found": eng_found, "injected": n_inj}
    lines.append(f"== Hidden injected clips ({len(injected)} clips, {n_inj} injected flaws)")
    lines.append(f"  engine found {frac(eng_found, n_inj)}")
    for r, labs in raters.items():
        got = tot_inj = 0
        for c, rs in injected.items():
            if c in labs:
                for x in rs:
                    tot_inj += 1
                    got += any(near(m["t_s"], x["start_s"], x["end_s"]) for m in labs[c]["marks"])
        res["hidden_injected"]["raters"][r] = {"found": got, "injected": tot_inj}
        lines.append(f"  {r} found {frac(got, tot_inj)}")
    lines.append("")

    # engine flags that no human mark matched, for the listening confirmation page
    unmatched = []
    for c in clips:
        for f in E[c]["flags"]:
            hit = any(near(m["t_s"], f["start_s"], f["end_s"]) for labs in raters.values() if c in labs for m in labs[c]["marks"])
            if not hit and any(c in labs for labs in raters.values()):
                unmatched.append({"flag": f"{c}@{f['start_s']:.2f}", "clip": c, "start_s": f["start_s"], "end_s": f["end_s"], "flaw": f["flaw"], "name": f["name"], "category": f["category"], "unedited": c not in injected})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "human_unmatched_flags.json").write_text(json.dumps(unmatched, indent=1) + "\n")

    # listening confirmation of unmatched flags (review_<reviewer>.json); a flag "a human agreed with" = matched by a mark, or confirmed on listening
    rev = {}
    for p in sorted(glob.glob(str(HUMAN / "review_*.json"))):
        for k, v in json.loads(Path(p).read_text()).items():
            rev.setdefault(k, []).append(v)
    n_flags = sum(len(E[c]["flags"]) for c in clips if any(c in labs for labs in raters.values()))
    n_unm = len(unmatched)
    agreed = sum(1 for u in unmatched if rev.get(u["flag"]) and rev[u["flag"]].count("agree") > rev[u["flag"]].count("disagree"))
    res["flags_agreed_with"] = {"flags_on_labelled_clips": n_flags, "matched_a_human_mark": n_flags - n_unm, "unmatched": n_unm, "unmatched_reviewed": sum(1 for u in unmatched if u["flag"] in rev),
                                "unmatched_agreed_on_listening": agreed}
    for scope, keep in (("unedited_only", lambda u: u["unedited"]),):
        un = [u for u in unmatched if keep(u)]
        fl = sum(len(E[c]["flags"]) for c in unedited if any(c in labs for labs in raters.values()))
        ag = sum(1 for u in un if rev.get(u["flag"]) and rev[u["flag"]].count("agree") > rev[u["flag"]].count("disagree"))
        res["flags_agreed_with"][scope] = {"flags": fl, "matched": fl - len(un), "unmatched": len(un), "agreed_on_listening": ag}
        lines.append(f"== Flags a human agreed with ({scope.replace('_', ' ')}): {frac(fl - len(un) + ag, fl)}  (matched a human mark {fl - len(un)}, confirmed on listening {ag}, unmatched and not confirmed {len(un) - ag}; {sum(1 for u in un if u['flag'] in rev)} of {len(un)} unmatched reviewed)")
    lines.append(f"== Flags a human agreed with (all clips): {frac(n_flags - n_unm + agreed, n_flags)}  (matched {n_flags - n_unm}, confirmed on listening {agreed}, unmatched {n_unm})")
    (ROOT / "results" / "human_agreement.json").write_text(json.dumps(res, indent=1) + "\n")
    (ROOT / "results" / "human_agreement.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
