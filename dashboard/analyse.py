"""Analyse page: upload (or pick) a clip -> quality, scorecard, time-warped timeline, flaw cards with A/B audio, table, export."""
from __future__ import annotations

import html
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
import yaml
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import audio as eaudio  # noqa: E402
from engine import predict as ENG  # noqa: E402
from engine import reference as eref  # noqa: E402
from engine.render_facts import syllables  # noqa: E402

DATA = ROOT / "flawline-dataset"
CAT_COLOR = {"Pacing": "#3b82c4", "Pausing": "#8b5cf6", "Intonation": "#f08c00", "Volume": "#2f9e44",
             "Fluency": "#e03131", "Clarity": "#0c8599", "Text fidelity": "#7c6f64"}
CATS = list(CAT_COLOR)
GENRES = ["interpretive reading", "declamation", "extemporaneous", "persuasive oratory"]
MODE_LABEL = {"same": "Same speaker (upper bound)", "cross": "Cross-speaker (EXPERIMENTAL: F1 0.20 train / 0.16 dev, 10 of 15 flaws)", "free": "Reference-free"}
BAND_COLOR = {"polished": "#2f9e44", "strong": "#2b6cb0", "noticeable": "#f08c00", "needs work": "#e03131"}


@st.cache_data(show_spinner=False)
def run_engine(path: str, meta_json: str, mode: str, ref_override: str | None, rubric_yaml: str | None, dont_fluency: bool, experimental: bool = False):
    meta = json.loads(meta_json)
    rub = yaml.safe_load(rubric_yaml) if rubric_yaml else None
    pred, C, ref_id = ENG.predict(path, meta, mode, rub, dont_fluency, ref_override, full=True, experimental=experimental)
    return pred, C, ref_id


def _warp(C, which: str):
    """Reference feature track mapped onto PARTICIPANT time through the DTW path (words line up)."""
    rr, pp = C.path
    fr_p, fr_r = C.par_fr, C.ref_fr
    ref_frame = np.interp(np.arange(len(fr_p.t)), pp.astype(float), rr.astype(float))
    idx = np.clip(np.round(ref_frame).astype(int), 0, len(fr_r.t) - 1)
    return (fr_r.f0_st if which == "f0" else fr_r.inten)[idx]


def timeline(pred: dict, C, wave: np.ndarray, sr: int, sel: int | None):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.34, 0.24, 0.24, 0.18],
                        subplot_titles=("Waveform and flaw regions", "Pitch (semitones re own median)", "Loudness (dB re own speech level)", "Speaking rate (syllables/s)"))
    n = 1500
    m = len(wave) // n * n
    seg = wave[:m].reshape(n, -1)
    t = (np.arange(n) + 0.5) * (len(wave) / sr / n)
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[seg.max(1), seg.min(1)[::-1]], fill="toself", mode="lines", line=dict(width=0),
                             fillcolor="rgba(75,85,99,.55)", hoverinfo="skip", showlegend=False), row=1, col=1)
    for i, r in enumerate(pred["what"]):
        col = CAT_COLOR[r["category"]]
        fig.add_vrect(x0=r["start_s"], x1=max(r["end_s"], r["start_s"] + 0.15), fillcolor=col, opacity=0.38 if i == sel else 0.22,
                      line=dict(color=col, width=2 if i == sel else 1), row="all", col=1)
        fig.add_trace(go.Scatter(x=[(r["start_s"] + r["end_s"]) / 2], y=[0.92], mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"),
                                 hovertemplate=f"<b>{r['flaw']}</b> · {r['category']}<br>severity {r['severity']:.1f} · −{r['points_lost']:.1f} pts<br>%{{x:.2f}} s<extra></extra>",
                                 showlegend=False), row=1, col=1)
    fp, fr = C.par_fr, C.ref_fr
    smooth = lambda v, k: np.convolve(np.pad(np.asarray(v, float), k // 2, mode="edge"), np.ones(k) / k, mode="valid")[: len(v)]
    for row, key, name in ((2, "f0_st", "f0"), (3, "inten", "inten")):
        yp_ = getattr(fp, key) if key == "f0_st" else smooth(getattr(fp, key), 9)
        yr_ = _warp(C, name) if key == "f0_st" else smooth(_warp(C, name), 9)
        fig.add_trace(go.Scatter(x=fp.t, y=yp_, mode="lines", line=dict(color="#4dabf7", width=1.6), name="You", legendgroup="you",
                                 showlegend=row == 2, connectgaps=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=fp.t, y=yr_, mode="lines", line=dict(color="#d9480f", width=1.2, dash="dot"), name="Reference (time-warped)",
                                 legendgroup="ref", showlegend=row == 2, connectgaps=False), row=row, col=1)
    xs, yp, yr = [], [], []
    for w, rw in zip(C.w, C.ref_words):
        syl = max(syllables(rw["clean"]), 1)
        dp, dr = max(w["pe"] - w["ps"], 0.05), max(w["re"] - w["rs"], 0.05)
        xs += [w["ps"], w["pe"]]
        yp += [syl / dp] * 2
        yr += [syl / dr * 1.0] * 2
    fig.add_trace(go.Scatter(x=xs, y=np.clip(yp, 0, 14), mode="lines", line=dict(color="#4dabf7", width=1.4, shape="hv"), showlegend=False,
                             hovertemplate="%{y:.1f} syll/s<extra>You</extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=xs, y=np.clip(yr, 0, 14), mode="lines", line=dict(color="#d9480f", width=1.0, dash="dot", shape="hv"), showlegend=False,
                             hovertemplate="%{y:.1f} syll/s<extra>Reference</extra>"), row=4, col=1)
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=34, b=0), hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.25)", zeroline=False)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(ticksuffix=" s", showgrid=False)
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", font=dict(size=12, color="#6b7280"))
    return fig


def cut_wav(x: np.ndarray, sr: int, a: float, b: float) -> bytes:
    seg = x[max(0, int(a * sr)): int(b * sr)]
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def script_html(ref_words: list[dict], flagged: dict[int, str]) -> str:
    out = []
    for i, w in enumerate(ref_words):
        t = html.escape(w["w"])
        if i in flagged:
            out.append(f'<span style="background:{flagged[i]}33;border-bottom:2px solid {flagged[i]};border-radius:3px;padding:0 1px">{t}</span>')
        else:
            out.append(t)
    return '<div style="line-height:1.9;font-size:1.02rem">' + " ".join(out) + "</div>"


def report_html(pred: dict) -> str:
    rows = "".join(f"<tr><td>{r['start_s']:.1f}–{r['end_s']:.1f}</td><td>{r['category']}</td><td>{r['flaw']}</td><td>{r['severity']:.1f}</td>"
                   f"<td>{r['points_lost']:.1f}</td><td>{html.escape(r.get('explanation', ''))}</td></tr>" for r in pred["what"])
    cats = "".join(f"<li>{c}: {v}</li>" for c, v in pred["scores"]["categories"].items())
    return (f"<html><body style='font:14px sans-serif;max-width:900px;margin:2rem auto'><h1>Delivery report</h1>"
            f"<h2>{pred['scores']['overall']} / 100 · {pred['scores']['band']}</h2><p>Reference mode: {pred['scores']['reference_mode']} · quality: {pred['quality']['badge']}</p>"
            f"<ul>{cats}</ul><table border=1 cellpadding=4 cellspacing=0><tr><th>time</th><th>category</th><th>flaw</th><th>severity</th><th>points lost</th><th>why</th></tr>{rows}</table></body></html>")


def render():
    st.title("Analyse")
    st.caption("Upload a speech (or pick a dataset clip). The engine compares it with a reference reading, flags delivery flaws with timestamps, explains each one, and scores it.")
    with st.sidebar:
        st.markdown("### Analyse inputs")
        src = st.radio("Source", ["Try a dataset clip", "Upload audio"], key="an_src")
        mode = st.radio("Reference", list(MODE_LABEL), format_func=lambda m: MODE_LABEL[m], index=0, key="an_mode",
                        help="Cross-speaker compares you with ANOTHER speaker reading the same text: the speaker-agnostic test. Same-speaker uses the clip's own clean take (an upper bound).")
        genre = st.selectbox("Genre preset", GENRES, key="an_genre")
        rub_file = st.file_uploader("…or your own rubric.yaml", type=["yaml", "yml"], key="an_rub")
        no_flu = st.checkbox("Don't score fluency", key="an_nofl", help="Fluency events stay visible but cost nothing (e.g. for a speaker who stammers).")
        show_exp = st.checkbox("Show experimental detectors", key="an_exp", help="EMPH_FLAT, REPEAT, UPTALK and RARE_HESIT are not reliable enough for the headline results (low precision or leakage); hidden unless you ask.")

    def _preset(mode_, take_, kind_, clip_):
        for k, v in (("an_src", "Try a dataset clip"), ("an_mode", mode_), ("an_take", take_), ("an_kind", kind_), ("an_clip", clip_)):
            st.session_state[k] = v
    st.markdown("**Demo clips**")
    d = st.columns(4)
    d[0].button("Accent vs another speaker", key="pr1", use_container_width=True, on_click=_preset, args=("cross", "B03-CHAMP", "Clean take", "B03-CHAMP_C0"),
                help="Indian-accent speaker compared with a different speaker's reading (cross-speaker, experimental). A clean reading should score high.")
    d[1].button("Noisy room, clean speech", key="pr2", use_container_width=True, on_click=_preset, args=("same", "B03-CHAMP", "Noisy / phone / room (clean speech)", "B03-CHAMP_N20"),
                help="Clean delivery recorded with babble noise at 20 dB: noise must not be mistaken for a flaw.")
    d[2].button("Very slow reading", key="pr3", use_container_width=True, on_click=_preset, args=("same", "B01-CHAMP", "Flaw at level 5", "B01-CHAMP_C0__PACE_SLOW_L5_s4414"),
                help="An egregious injected flaw (pace slowed at level 5).")
    d[3].button("A real long pause", key="pr4", use_container_width=True, on_click=_preset, args=("cross", "B07-CHAMP", "Clean take", "B07-CHAMP_C0"),
                help="An unedited VCTK reading with a natural 1.7 s pause; the engine's flag here is a real pause, not an injected one.")

    path, meta, ref_override, gt = None, None, None, None
    if src == "Try a dataset clip":
        man = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
        c1, c2, c3, c4 = st.columns(4)
        take = c1.selectbox("Baseline speaker", sorted(man.take_id.unique()), key="an_take")
        kinds = {"Clean take": "clean", "One flaw (level 3)": "L3", "Multi-flaw set": "MULTI", "Noisy / phone / room (clean speech)": "COND", "Flaw at level 5": "L5"}
        kind = c2.selectbox("Clip type", list(kinds), index=1, key="an_kind")
        sub = man[man.take_id == take]
        if kinds[kind] == "clean":
            pool = sub[(sub.flaw_codes == "") & (sub.added_condition == "")]
        elif kinds[kind] == "COND":
            pool = sub[(sub.flaw_codes == "") & (sub.added_condition != "")]
        elif kinds[kind] == "MULTI":
            pool = sub[sub.multi_set != ""]
        else:
            lv = "_L3_" if kinds[kind] == "L3" else "_L5_"
            pool = sub[sub.clip_id.str.contains(lv) & (sub.multi_set == "") & (sub.added_condition == "") & (~sub.clip_id.str.contains("PURE_")) & (sub.flaw_codes != "")]
        exp = {"EMPH_FLAT", "REPEAT", "UPTALK", "RARE_HESIT"}               # experimental flaws (see README): listed last
        fl = lambda c: c.split("__")[1].rsplit("_L", 1)[0] if "__" in c else ""
        ids = sorted(pool.clip_id.tolist(), key=lambda c: (fl(c) in exp, c))
        cid = c3.selectbox("Clip", ids, key="an_clip",
                           format_func=lambda c: (c.split("__")[-1] if "__" in c else c.split("_")[-1]) + (" · experimental flaw" if fl(c) in exp else ""))
        row = man[man.clip_id == cid].iloc[0]
        path = str(DATA / "variants" / f"{cid}.flac") if (DATA / "variants" / f"{cid}.flac").exists() else str(DATA / "takes" / f"{cid}.flac")
        meta = {"clip_id": cid, "take_id": row.take_id, "baseline_id": row.baseline_id, "genre": genre, "duration_s": float(row.duration_s),
                "who": {"speaker_id": row.speaker_id}, "where": {"base_condition": "C0", "added_condition": row.added_condition or None}}
        jp = DATA / "variants" / f"{cid}.json"
        gt = json.loads(jp.read_text()) if jp.exists() else None
        c4.caption("Ground truth (what was injected) is shown only in the expander below.")
    else:
        up = st.file_uploader("Audio (wav, mp3, m4a, flac)", type=["wav", "mp3", "m4a", "flac"], key="an_up")
        ref_override = st.selectbox("Which baseline text did you read? (reference)", eref.BASELINES, key="an_refpick",
                                    format_func=lambda b: f"{b} · " + eref.words_of(b)[0]["w"] + " … (same text as the other baselines)")
        if up:
            suffix = Path(up.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(up.read())
                path = tf.name
            x = eaudio.load(path)
            meta = {"clip_id": "B00-UPLOAD_C0", "take_id": "B00-UPLOAD", "baseline_id": ref_override, "genre": genre, "duration_s": len(x) / eaudio.SR,
                    "who": {"speaker_id": "UPLOAD"}, "where": {"base_condition": "C0", "added_condition": None}}
            mode = "same" if mode == "free" else mode
    if not path:
        st.info("Choose a clip or upload audio to begin.")
        return
    with st.spinner("Aligning against the reference and measuring delivery… (first run of a clip takes a few seconds)"):
        pred, C, ref_id = run_engine(path, json.dumps(meta), mode, ref_override, rub_file.getvalue().decode() if rub_file else None, no_flu, show_exp)
    wave, sr = sf.read(path, dtype="float32")
    wave = wave if wave.ndim == 1 else wave.mean(axis=1)

    q = pred["quality"]
    qcol = {"good": "#2f9e44", "fair": "#f08c00", "poor": "#e03131"}[q["badge"]]
    cA, cB = st.columns([1, 2])
    with cA:
        sc = pred["scores"]
        st.markdown(f"<div style='font-size:3.2rem;font-weight:700;color:{BAND_COLOR[sc['band']]};line-height:1'>{sc['overall']:.0f}</div>"
                    f"<div style='font-size:1.1rem;font-weight:600'>{sc['band']}</div>"
                    f"<div class='small'>reference: {MODE_LABEL[mode]} · {ref_id} · {pred['reference']['common_words']} words compared</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='pill' style='background:{qcol}22;color:{qcol}'>recording quality: {q['badge']}</span> "
                    f"<span class='small'>SNR {q['snr_db']} dB · bandwidth {q['bandwidth_hz']} Hz · clipped {q['clipped_frac']:.3%}</span>", unsafe_allow_html=True)
        if q["badge"] == "poor":
            st.warning("Low-confidence recording: flagged regions are shown but cost nothing.")
        if q.get("matched"):
            st.caption("Recording conditions matched on the reference: " + ", ".join(f"{k} {v}" for k, v in q["matched"].items()))
    with cB:
        lost = {c: 100 - v for c, v in sc["categories"].items()}
        fig = go.Figure(go.Bar(y=list(lost), x=[sc["categories"][c] for c in lost], orientation="h", marker_color=[CAT_COLOR[c] for c in lost],
                               text=[f"{sc['categories'][c]:.0f}  (−{lost[c]:.0f})" for c in lost], textposition="inside", hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=6, b=0), xaxis=dict(range=[0, 100], title="category score"), yaxis=dict(autorange="reversed"),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    regs = pred["what"]
    if not regs:
        st.success("No delivery flaws found against the reference.")
    order = sorted(range(len(regs)), key=lambda i: -regs[i]["points_lost"])
    sel = st.session_state.get("an_sel") if st.session_state.get("an_sel") in range(len(regs)) else (order[0] if order else None)
    st.plotly_chart(timeline(pred, C, wave, sr, sel), use_container_width=True)
    flagged = {}
    for r in regs:
        for i in range(r["word_start"], r["word_end"] + 1):
            flagged[i] = CAT_COLOR[r["category"]]
    st.markdown(script_html(C.ref_words, flagged), unsafe_allow_html=True)
    st.markdown("  ".join(f"<span class='pill' style='background:{c}22;color:{c}'>{k}</span>" for k, c in CAT_COLOR.items()), unsafe_allow_html=True)

    if regs:
        st.subheader("Flaw card")
        sel = st.selectbox("Region", order, index=0, key="an_sel", format_func=lambda i: f"{regs[i]['start_s']:.1f} s · {regs[i]['flaw']} · −{regs[i]['points_lost']:.1f} pts")
        r = regs[sel]
        col = CAT_COLOR[r["category"]]
        st.markdown(f"<span class='pill' style='background:{col}22;color:{col}'>{r['category']}</span> **{r['flaw']}** · severity {r['severity']:.1f}/5 · "
                    f"{r['start_s']:.2f}–{r['end_s']:.2f} s · **−{r['points_lost']:.1f} points**", unsafe_allow_html=True)
        st.markdown(r["explanation"])
        if r["params"].get("tip"):
            st.markdown(f"💡 *{r['params']['tip']}*")
        a, b = st.columns(2)
        a.markdown("**You**")
        a.audio(cut_wav(wave, sr, r["start_s"] - 0.4, r["end_s"] + 0.4), format="audio/wav")
        rx = eaudio.load(eref.audio_of(ref_id))
        w0, w1 = r["word_start"], r["word_end"]
        b.markdown(f"**Reference** ({ref_id}, same words)")
        b.audio(cut_wav(rx, eaudio.SR, C.w[w0]["rs"] - 0.4, C.w[w1]["re"] + 0.4), format="audio/wav")

    st.subheader("All flaws")
    if regs:
        df = pd.DataFrame([{"time": f"{r['start_s']:.1f}–{r['end_s']:.1f} s", "category": r["category"], "flaw": r["flaw"], "severity": r["severity"],
                            "points lost": r["points_lost"], "why": r["explanation"]} for r in regs])
        st.dataframe(df.sort_values("points lost", ascending=False), hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download score JSON (label format)", json.dumps(pred, indent=2, ensure_ascii=False), file_name=f"{meta['clip_id']}.score.json", mime="application/json")
    d2.download_button("Download one-page report (HTML)", report_html(pred), file_name=f"{meta['clip_id']}.report.html", mime="text/html")
    if gt is not None:
        with st.expander("Ground truth for this dataset clip (what was injected)"):
            st.dataframe(pd.DataFrame([{"flaw": r["flaw"], "level": r.get("level"), "time": f"{r['start_s']:.1f}–{r['end_s']:.1f} s", "why": r.get("why", "")}
                                       for r in gt["what"]]), hide_index=True, use_container_width=True)
