"""Flawline LAB pages (for the team, and for screen-recording the data-engineering part of the video). Started by `LAB=1 python -m app.server`
(or `make lab`) and linked from the web app's nav; the public pages (Analyse, Dataset, About) live in app/.

  streamlit run dashboard/app.py

Pages: Baselines (coverage across age / origin / register, disfluency audit) · Alterations review (A/B player + verdict, writes
pilot/realism.csv) · Pilot gate (spec pass rule per flaw).
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from player import synced_player  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "flawline-dataset"
VARIANTS, TAKES, PILOT = ROOT / "variants", ROOT / "takes", ROOT / "pilot"
REALISM = PILOT / "realism.csv"
RCOLS = ["clip_id", "take_id", "flaw", "level", "realism", "level_feel", "notes", "reviewer", "timestamp"]
REALISM_OPTS = ["natural", "slight artifact", "robotic"]
FEEL_OPTS = ["too weak", "right", "too strong"]
FLAW_ORDER = ["PACE_FAST", "PACE_SLOW", "PAUSE_BAD", "PAUSE_LOST", "MONOTONE", "UPTALK", "EMPH_FLAT", "FADE", "SHOUT",
              "FILLER", "REPEAT", "RARE_HESIT", "SLUR", "WORD_SKIP", "WORD_SWAP"]
AGE_ORDER = ["18-25", "26-45", "46-65", "65+"]
REG_ORDER = ["formal", "neutral", "colloquial", "slang-heavy"]
# tokens: neutral waveform ink, one accent for "this is the flaw", one sequential hue for counts
INK, MUTED, GRID, ACCENT, OK, BAD = "#4b5563", "#8b93a1", "#e5e7eb", "#d9480f", "#2f855a", "#c53030"

st.set_page_config(page_title="Flawline · dataset review", layout="wide", page_icon="🎧")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1250px}
h1,h2,h3{letter-spacing:-.01em}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:600;margin-right:6px}
.pill.warn{background:#fff4e6;color:#9c4a00}.pill.ok{background:#e6f4ea;color:#1e6b3a}.pill.mute{background:#eef0f3;color:#4b5563}
.script{line-height:1.9;font-size:1.02rem}
.script mark.slang{background:#fff3bf;padding:0 2px;border-radius:3px}
.script span.flaw{background:#ffe3d3;border-bottom:2px solid #d9480f;border-radius:3px;padding:0 1px}
.script span.gone{background:#ffe3d3;text-decoration:line-through;color:#9c4a00;border-radius:3px;padding:0 1px}
.script span.ins{border-left:3px solid #d9480f;margin-left:2px;padding-left:2px}
.small{color:#6b7280;font-size:.85rem}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data
def load_manifest(mtime: float) -> pd.DataFrame:
    m = pd.read_csv(ROOT / "manifest.csv", keep_default_na=False)
    m["level"] = m["max_level"]
    m["flaw"] = np.where(m["flaw_codes"] == "", "", np.where(m["multi_set"] != "", "MULTI", m["flaw_codes"]))
    return m


@st.cache_data
def load_baselines():
    cfg = yaml.safe_load(open(ROOT / "generator" / "baselines.yaml"))["baselines"]
    cand = yaml.safe_load(open(ROOT / "generator" / "candidates.yaml"))["candidates"]
    status = {}
    for b in cfg:
        p = ROOT / "baselines" / b["id"] / "source.json"
        status[b["id"]] = json.loads(p.read_text())["status"] if p.exists() else "unbuilt"
    return cfg, cand, status


@st.cache_data
def load_label(clip_id: str) -> dict:
    p = VARIANTS / f"{clip_id}.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_align(take_id: str) -> list[dict]:
    return json.loads((TAKES / f"{take_id}_C0.align.json").read_text())["words"]


@st.cache_data
def load_wave(path: str, bins: int = 1600):
    x, sr = sf.read(path, dtype="float32")
    n = len(x) // bins * bins
    seg = x[:n].reshape(bins, -1) if n else x.reshape(1, -1)
    t = (np.arange(len(seg)) + 0.5) * (len(x) / len(seg)) / sr
    return t, seg.min(axis=1), seg.max(axis=1), len(x) / sr


def audio_path(clip_id: str) -> Path:
    p = VARIANTS / f"{clip_id}.flac"
    return p if p.exists() else TAKES / f"{clip_id}.flac"


def play(clip_id: str):
    st.audio(audio_path(clip_id).read_bytes(), format="audio/flac")


def load_realism() -> pd.DataFrame:
    return pd.read_csv(REALISM, keep_default_na=False) if REALISM.exists() else pd.DataFrame(columns=RCOLS)


def save_verdict(row: dict):
    df = load_realism()
    df = df[df.clip_id != row["clip_id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[RCOLS]
    PILOT.mkdir(exist_ok=True)
    df.to_csv(REALISM, index=False)


mf = ROOT / "manifest.csv"
if not mf.exists():
    st.error("manifest.csv not found. Run `python flawline-dataset/generator/make_dataset.py --pilot B01-CHAMP --l3`.")
    st.stop()
M = load_manifest(mf.stat().st_mtime)
BASE, CAND, STATUS = load_baselines()

page = st.sidebar.radio("View", ["Alterations review", "Baselines", "Pilot gate"], label_visibility="collapsed")
st.sidebar.caption("Flawline · Track C dataset v1 (dev)")
reviewer = st.sidebar.text_input("Reviewer", value="Sai")
rated = load_realism()
st.sidebar.metric("Clips reviewed", f"{len(rated)}")


# --------------------------------------------------------------------------- charts
def waveform(clip_id: str, regions: list[tuple[float, float]], title: str, height=170):
    t, lo, hi, dur = load_wave(str(audio_path(clip_id)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="lines", line=dict(width=0),
                             fillcolor="rgba(75,85,99,.55)", hoverinfo="skip", showlegend=False))
    for a, b in regions:
        if b - a < 0.05:      # deletion / point: draw a marker line
            fig.add_vline(x=a, line=dict(color=ACCENT, width=2))
        else:
            fig.add_vrect(x0=a, x1=b, fillcolor=ACCENT, opacity=0.22, line=dict(color=ACCENT, width=1))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=26, b=0), title=dict(text=title, x=0, font=dict(size=13, color=MUTED)),
                      xaxis=dict(range=[0, dur], showgrid=False, ticksuffix=" s", color=MUTED), yaxis=dict(visible=False, range=[-1, 1]),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode=False)
    return fig


def script_html(words: list[dict], flagged: dict[int, str], slang: list[str] | None = None) -> str:
    out = []
    for w in words:
        t = html.escape(w["w"])
        k = flagged.get(w["i"])
        if k == "flaw":
            t = f'<span class="flaw">{t}</span>'
        elif k == "gone":
            t = f'<span class="gone">{t}</span>'
        elif k == "ins":
            t = f'<span class="ins">{t}</span>'
        out.append(t)
    s = " ".join(out)
    for term in sorted(slang or [], key=len, reverse=True):
        s = re.sub(rf"(?i)(?<![\w>]){re.escape(html.escape(term))}(?![\w<])", lambda m: f'<mark class="slang">{m.group(0)}</mark>', s, count=0)
    return f'<div class="script">{s}</div>'


# --------------------------------------------------------------------------- page: baselines
if page == "Baselines":
    st.title("Baselines")
    st.caption("The yardsticks every alteration is made from. Slots cover age band × origin/accent × register (slang level).")
    bdf = pd.DataFrame(BASE)
    bdf["status"] = bdf.id.map(STATUS)
    AUD = json.loads((ROOT / "baselines" / "audit.json").read_text()) if (ROOT / "baselines" / "audit.json").exists() else {}
    bdf["disfluency audit"] = bdf.id.map(lambda i: AUD.get(i, {}).get("status", "not run"))
    placeholder = (bdf.status == "placeholder_tts").any()
    real = bdf.status.str.startswith("real")
    if real.any():
        st.success(f"**{int(real.sum())} of {len(bdf)} baselines are real recordings** (VCTK 0.92: studio-recorded, CC BY 4.0, read aloud). "
                   "Limits: speakers are aged 18–38 only, the text is read (no slang, not a champion speech). "
                   "Slang and older speakers are planned through reader recordings (`readers/RECORDING_SHEET.md`).")
    if placeholder:
        st.warning("**Placeholder audio.** These baselines are original CC0 scripts read by macOS system voices matched to the target accent. "
                   "They let the flaw factory and this dashboard be built now. Age band is the *target slot* (a TTS voice can't sound its age). "
                   "The frozen v1.0 set must use real champion recordings (see the candidate pool below).")

    c1, c2, c3, c4 = st.columns(4)
    f_age = c1.multiselect("Age band", AGE_ORDER)
    f_reg = c2.multiselect("Register", REG_ORDER)
    f_gen = c3.multiselect("Gender", sorted(bdf.gender.unique()))
    f_org = c4.multiselect("Origin", sorted(bdf.origin.unique()))
    v = bdf
    if f_age: v = v[v.age_band.isin(f_age)]
    if f_reg: v = v[v.register.isin(f_reg)]
    if f_gen: v = v[v.gender.isin(f_gen)]
    if f_org: v = v[v.origin.isin(f_org)]

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(v[[c for c in ["id", "title", "age", "age_band", "origin", "gender", "register", "status", "disfluency audit"] if c in v.columns]],
                     hide_index=True, use_container_width=True)
    with right:
        z = np.zeros((len(AGE_ORDER), len(REG_ORDER)), int)
        for _, r in bdf.iterrows():
            z[AGE_ORDER.index(r.age_band), REG_ORDER.index(r.register)] += 1
        fig = go.Figure(go.Heatmap(z=z, x=REG_ORDER, y=AGE_ORDER, colorscale=[[0, "#f3f4f6"], [1, "#2b6cb0"]], showscale=False,
                                   text=z, texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z} baseline(s)<extra></extra>", xgap=3, ygap=3))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=28, b=0), title=dict(text="Coverage: age band × register (empty = gap)", x=0, font=dict(size=13, color=MUTED)),
                          yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spec coverage check")
    ok_ = lambda c: "✅" if c else "❌"
    n_ind = bdf.accent.str.contains("Indian|Tamil").sum()
    checks = [
        (f"≥ 2 genders", bdf.gender.nunique() >= 2),
        (f"≥ 2 accents incl. Indian English ({bdf.accent.nunique()} accents, {n_ind} Indian)", bdf.accent.nunique() >= 2 and n_ind >= 1),
        ("All four age bands present", set(AGE_ORDER) <= set(bdf.age_band)),
        ("All four genres, 2 each", (bdf.genre.value_counts() == 2).all() and bdf.genre.nunique() == 4),
        ("Slang-heavy and colloquial registers present", {"slang-heavy", "colloquial"} <= set(bdf.register)),
        ("Real recordings (not placeholders)", bdf.status.str.startswith("real").all()),
    ]
    st.markdown("\n".join(f"- {ok_(c)} {t}" for t, c in checks))

    st.subheader("Listen")
    pick = st.selectbox("Baseline", v.id.tolist() or bdf.id.tolist(), key="base_pick", format_func=lambda i: f"{i} · {bdf[bdf.id == i].title.iloc[0]}")
    b = bdf[bdf.id == pick].iloc[0]
    st.markdown(f'<span class="pill {"warn" if b.status == "placeholder_tts" else "ok"}">{b.status.replace("_", " ")}</span>'
                f'<span class="pill mute">{b.genre}</span><span class="pill mute">{b.age_band}</span><span class="pill mute">{b.accent}</span>'
                f'<span class="pill mute">{b.gender}</span><span class="pill mute">{b.register}</span>', unsafe_allow_html=True)
    prov = json.loads((ROOT / "baselines" / pick / "source.json").read_text())
    if prov.get("corpus"):
        st.caption(f"Source: {prov['corpus']} · speaker {prov['speaker_id']} · {prov['licence']} · [{prov['url']}]({prov['url']})")
    play(f"{pick}-CHAMP_C0")
    a = AUD.get(pick)
    if a:
        icon = {"PASS": "✅", "REVIEW": "🟡", "FAIL": "❌", "NATURAL": "🗣️"}.get(a["status"], "•")
        with st.expander(f"{icon} Disfluency audit: {a['status']} · {a['fillers']} fillers · {len(a['issues'])} flags", expanded=a["status"] in ("FAIL", "REVIEW")):
            if a.get("spontaneous"):
                st.caption("NATURAL = spontaneous speech, so its own hesitations and pauses are real, recorded, and expected. They are listed so "
                           "they are never confused with injected flaws. The human transcript omits fillers, so reference text ≠ spoken words.")
            else:
                st.caption("Verbatim ASR diffed against the reference text. A baseline must be clean (no ums, repeats, skipped words). "
                           "ASR can miss a quiet filler, so PASS means nothing was found, not proof. Listen to every flag.")
            st.caption(f"Estimated SNR {a.get('snr_db_est', '?')} dB · clipped samples {a.get('clipped_frac', 0):.3%}")
            if a["issues"]:
                st.dataframe(pd.DataFrame(a["issues"]), hide_index=True, use_container_width=True)
                t0 = st.selectbox("Play from a flag", [i["time_s"] for i in a["issues"]],
                                  format_func=lambda t: f"{t:.2f} s · " + next(i["kind"] for i in a["issues"] if i["time_s"] == t))
                st.audio(audio_path(f"{pick}-CHAMP_C0").read_bytes(), format="audio/flac", start_time=max(0, int(t0) - 2))
            else:
                st.write("Nothing flagged.")
    st.plotly_chart(waveform(f"{pick}-CHAMP_C0", [], "Waveform"), use_container_width=True)
    st.markdown(script_html(load_align(f"{pick}-CHAMP"), {}, b.slang_terms), unsafe_allow_html=True)
    if b.slang_terms:
        st.markdown(f'<span class="small">Highlighted = slang / dialect terms: {", ".join(b.slang_terms)}. Never scored (spec: vocabulary is not scored); '
                    f'listed so WORD_SWAP is never confused with slang.</span>', unsafe_allow_html=True)

    with st.expander("Candidate real speeches to replace the placeholders (unverified)"):
        st.caption("From memory, nothing here is verified. Check each licence and source before use; spec rule: licence unclear → drop it.")
        st.dataframe(pd.DataFrame(CAND), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- page: review
elif page == "Alterations review":
    st.title("Alterations review")
    st.caption("Listen to each alteration against its baseline, check the flagged region, and record a verdict. Verdicts go to `pilot/realism.csv`.")
    alt = M[(M.flaw != "") & (M.added_condition == "")]
    f1, f2, f3, f4 = st.columns([1.4, 1.4, 1, 1.4])
    takes = sorted(alt.take_id.unique())
    take = f1.selectbox("Baseline", takes, key="rev_take", format_func=lambda t: f"{t.split('-')[0]} · " + next(b["title"] for b in BASE if b["id"] == t.split("-")[0]))
    flaws = [f for f in FLAW_ORDER + ["MULTI"] if f in set(alt[alt.take_id == take].flaw)]
    flaw = f2.selectbox("Alteration", flaws, key="rev_flaw")
    pool = alt[(alt.take_id == take) & (alt.flaw == flaw)].sort_values("level")
    levels = pool.level.tolist()
    level = f3.selectbox("Level", levels, key=f"rev_level_{take}_{flaw}", index=min(2, len(levels) - 1) if 3 not in levels else levels.index(3))
    only_new = f4.checkbox("Unreviewed only (this baseline)", value=False)
    if only_new:
        done = set(rated.clip_id)
        pool = pool[~pool.clip_id.isin(done) | (pool.level == level)]
    row = pool[pool.level == level]
    if row.empty:
        st.info("Nothing to show.")
        st.stop()
    cid = row.clip_id.iloc[0]
    lab = load_label(cid)
    base_id = f"{take}_C0"

    st.markdown(f"**{cid}**")
    if STATUS.get(take.split("-")[0]) == "placeholder_tts":
        st.markdown('<span class="pill warn">baseline = synthetic TTS voice, not a human recording</span>', unsafe_allow_html=True)
    what = lab["what"]
    words = load_align(take)
    flagged = {}
    for r in what:
        for i in range(r["word_start"], r["word_end"] + 1):
            flagged[i] = "gone" if r["kind"] == "delete" and r["flaw"] == "WORD_SKIP" else "ins" if r["kind"] == "insert" else "flaw"
    synced_player(audio_path(base_id), audio_path(cid), what, words, flagged)
    st.markdown(script_html(words, flagged), unsafe_allow_html=True)
    st.markdown('<span class="small">Orange = altered words · struck-through = skipped · bar = insertion point (pause / filler / repeat).</span>', unsafe_allow_html=True)

    reg = pd.DataFrame([{"#": i + 1, "flaw": r["flaw"], "where (grammar)": r.get("boundary", ""), "why": r.get("why", ""),
                         "context": r.get("context", ""), "clip s": f'{r["start_s"]:.2f}–{r["end_s"]:.2f}',
                         "baseline s": f'{r["baseline_start_s"]:.2f}–{r["baseline_end_s"]:.2f}',
                         "params": json.dumps({k: v for k, v in r["params"].items() if k != "evidence"})} for i, r in enumerate(what)])
    st.markdown("**Why these spots**")
    for i, r in enumerate(what):
        ev = r["params"].get("evidence")
        evs = ""
        if ev:
            evs = " · evidence: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in ev.items())
        st.markdown(f"{i + 1}. **{r['flaw']}** — {r.get('why', '(no grammar rule for this flaw)')}"
                    + (f"  \n   `{r['context']}`" if r.get("context") else "") + evs)
    jc = lab.get("join_check")
    if jc:
        st.markdown(f"**Artifact gate:** {'✅ passed' if jc['passed'] else '🟡 best available placement (gate not met)'} · worst join "
                    f"{jc['max_click_db']} dB over baseline · {jc['share_above_6db']:.0%} of {jc['n_joins']} joins above 6 dB · "
                    f"placement attempt {jc['attempt']}" + (" · single-cue (pure) rendering" if "PURE_" in cid else " · natural bundle rendering"))
    with st.expander("Regions and parameters", expanded=False):
        st.dataframe(reg, hide_index=True, use_container_width=True)
    ac = PILOT / "artifact_check.csv"
    if ac.exists():
        A = pd.read_csv(ac)
        A = A[A.clip_id == cid]
        if not A.empty:
            worst = float(A.click_db.max())
            tag = "✅ smooth" if worst < 12 else "🟡 listen closely" if worst < 18 else "🔴 likely audible click"
            st.markdown(f"**Join check:** {tag} · worst click {worst:.1f} dB over what the baseline does naturally at that spot "
                        f"(below ~12 dB is normal). This is a meter, not a substitute for listening.")
            with st.expander("Per-edge join metrics"):
                st.dataframe(A.drop(columns=["clip_id"]), hide_index=True, use_container_width=True)
    oc = PILOT / "objective_check.csv"
    if oc.exists() and flaw != "MULTI":
        o = pd.read_csv(oc)
        o = o[(o.take_id == take) & (o.flaw == flaw)]
        if not o.empty:
            o = o.iloc[0]
            vals = [o[f"L{l}"] for l in range(1, 6)]
            st.markdown(f"**Measured effect** ({o.unit}): " + " → ".join(f"**{v:g}**" if l == level else f"{v:g}" for l, v in zip(range(1, 6), vals))
                        + f"  · Spearman {o.spearman:+.2f}")
    with st.expander("Label JSON"):
        st.json(lab)

    ladder = pool.sort_values("level")
    if len(ladder) > 1:
        st.markdown("**Level ladder** — do they sound like they get worse in order?")
        cols = st.columns(len(ladder))
        for c, (_, r) in zip(cols, ladder.iterrows()):
            with c:
                st.caption(f"L{r.level}" + ("  ◀" if r.level == level else ""))
                play(r.clip_id)

    prev = rated[rated.clip_id == cid]
    p = prev.iloc[0] if len(prev) else None
    st.markdown("### Verdict")
    with st.form(f"v-{cid}"):
        c1, c2 = st.columns(2)
        real = c1.radio("Does it sound real?", REALISM_OPTS, index=REALISM_OPTS.index(p.realism) if p is not None and p.realism in REALISM_OPTS else 0, horizontal=True)
        feel = c2.radio(f"Does L{level} feel…", FEEL_OPTS, index=FEEL_OPTS.index(p.level_feel) if p is not None and p.level_feel in FEEL_OPTS else 1, horizontal=True)
        notes = st.text_input("Notes", value=p.notes if p is not None else "")
        if st.form_submit_button("Save verdict", type="primary"):
            save_verdict({"clip_id": cid, "take_id": take, "flaw": flaw, "level": level, "realism": real, "level_feel": feel,
                          "notes": notes, "reviewer": reviewer, "timestamp": datetime.now().isoformat(timespec="seconds")})
            st.success("Saved")
            st.rerun()
    if p is not None:
        st.caption(f"Last verdict: {p.realism} · {p.level_feel} · {p.reviewer} · {p.timestamp}")


# --------------------------------------------------------------------------- page: pilot gate
elif page == "Pilot gate":
    st.title("Pilot gate")
    st.caption("Spec rule: a flaw passes when ≥ 80% of its rated clips sound natural or only slightly artifacted **and** its levels sound in order. A flaw that can't pass is dropped from v1, not shipped.")
    scope = st.selectbox("Baseline", sorted(M[M.flaw != ""].take_id.unique()), index=0, key="gate_take")
    R = rated[rated.take_id == scope]
    oc = PILOT / "objective_check.csv"
    O = pd.read_csv(oc) if oc.exists() else pd.DataFrame()
    rows = []
    for f in FLAW_ORDER:
        total = int(((M.take_id == scope) & (M.flaw == f) & (M.added_condition == "")).sum())
        r = R[R.flaw == f]
        n = len(r)
        pct = float((r.realism != "robotic").mean()) if n else np.nan
        feel = r.assign(s=r.level_feel.map({"too weak": -1, "right": 0, "too strong": 1})).groupby("level").s.mean()
        ordered = bool(len(feel) >= 3 and feel.is_monotonic_increasing)
        o = O[(O.take_id == scope) & (O.flaw == f)]
        obj = float(o.spearman.iloc[0]) if len(o) else np.nan
        status = "not started" if n == 0 else "incomplete" if n < total else ("PASS" if pct >= 0.8 and ordered else "FAIL")
        rows.append({"flaw": f, "rated": f"{n}/{total}", "natural or slight": f"{pct:.0%}" if n else "–",
                     "levels in order (by ear)": "yes" if ordered else ("–" if len(feel) < 3 else "no"),
                     "objective Spearman": round(obj, 2) if not np.isnan(obj) else None, "status": status})
    T = pd.DataFrame(rows)
    st.dataframe(T.style.map(lambda s: f"color:{OK};font-weight:700" if s == "PASS" else f"color:{BAD};font-weight:700" if s == "FAIL" else "", subset=["status"]),
                 hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pass", int((T.status == "PASS").sum()))
    c2.metric("Fail", int((T.status == "FAIL").sum()))
    c3.metric("Still to review", int(T.status.isin(["not started", "incomplete"]).sum()))
    st.caption("Objective Spearman comes from `generator/verify_flaws.py` (measured effect vs level, no listening). "
               "It is a sanity check, not a substitute for the by-ear gate.")
