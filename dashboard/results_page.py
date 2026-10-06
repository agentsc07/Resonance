"""Dataset overview: who / where / what counts and the measured results (headline F1, acceptance tests, leakage audit). Reads results/*.json|csv."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RES = Path(__file__).resolve().parent.parent / "results"
GRID = "rgba(128,128,128,.25)"


def _json(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _bar(df: pd.DataFrame, x: str, y: str, title: str, color="#4dabf7", vline=None, height=360):
    fig = go.Figure(go.Bar(x=df[x], y=df[y], orientation="h", marker_color=color, text=[f"{v:.2f}" for v in df[x]], textposition="outside"))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dot", line_color="#e03131", annotation_text=f"{vline}", annotation_position="top")
    fig.update_layout(height=height, margin=dict(l=0, r=30, t=30, b=0), title=dict(text=title, x=0, font=dict(size=13)), yaxis=dict(autorange="reversed"),
                      xaxis=dict(gridcolor=GRID), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def counts(M: pd.DataFrame):
    st.subheader("Who · Where · What")
    base = M[(M.flaw_codes == "") & (M.added_condition == "")]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Who** (baseline speakers)")
        who = base.drop_duplicates("baseline_id")[["baseline_id", "age_band", "accent", "gender", "origin", "split"]].rename(columns={"baseline_id": "baseline"})
        st.dataframe(who, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Where** (recording conditions)")
        cond = M.assign(condition=M.added_condition.replace("", "clean (C0)")).groupby("condition").size().rename("clips").reset_index()
        st.dataframe(cond, hide_index=True, use_container_width=True)
        st.caption("N20/N10: reversed multi-talker babble · RVB: room reverb · PHN: phone band · MP3: codec · GAIN: ±level.")
    with c3:
        st.markdown("**What** (injected flaws)")
        wh = M[M.flaw_codes != ""].assign(flaw=lambda d: d.flaw_codes.str.split("+").str[0]).groupby("flaw").size().rename("clips").reset_index()
        st.dataframe(wh, hide_index=True, use_container_width=True)


def results():
    st.subheader("Measured results")
    st.caption("Event F1 at IoU 0.5 against the injected ground truth. Thresholds are fitted on TRAIN only. The test split (B05, B08) is evaluated once, at the end.")
    rows = []
    for split in ("train", "dev", "test"):
        for mode in ("same", "cross"):
            h = _json(f"headline_{split}_{mode}.json")
            if h:
                rows.append({"split": split, "reference": "same speaker" if mode == "same" else "cross speaker (experimental)", "clips": h["clips"],
                             "all flaws F1": h["all_flaws"]["f1"], "headline F1": h["headline"]["f1"], "headline flaws": len(h["headline"]["flaws"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    hs, hc = _json("headline_train_same.json"), _json("headline_train_cross.json")
    if hs and hc:
        df = pd.DataFrame({"flaw": list(hs["per_flaw_f1"]), "same": [hs["per_flaw_f1"][k] for k in hs["per_flaw_f1"]], "cross": [hc["per_flaw_f1"].get(k, 0) for k in hs["per_flaw_f1"]]})
        fig = go.Figure([go.Bar(y=df.flaw, x=df.same, orientation="h", name="same speaker", marker_color="#4dabf7"),
                         go.Bar(y=df.flaw, x=df.cross, orientation="h", name="cross speaker", marker_color="#f08c00")])
        fig.update_layout(height=520, barmode="group", margin=dict(l=0, r=0, t=70, b=0), title=dict(text="Event F1 per flaw (train)", x=0, font=dict(size=13)),
                          yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1], gridcolor=GRID), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.0, yanchor="bottom"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Experimental (excluded from headline): " + "; ".join(f"**{k}** ({v})" for k, v in hs["experimental"].items()))

    a = _json("acceptance_same.json")
    st.subheader("Acceptance tests (same-speaker)")
    if a:
        c1, c2, c3 = st.columns(3)
        dr, iv, lo = a["dose_response"], a["invariance"], a["locality"]
        c1.metric("Score vs level (median Spearman)", f"{dr['median_spearman']:.2f}", help="target ≤ −0.9 per flaw; failing: " + ", ".join(dr["failing"]))
        c2.metric("False flags / min under Where changes", f"{iv['false_flags_per_min']:.2f}", help="target ≤ 0.5")
        c3.metric("Other-category loss (locality)", f"{lo['mean_other_category_loss']:.1f} pts", help="target < 5 (stretch < 2)")
        d1, d2 = st.columns(2)
        with d1:
            per = dr["per_flaw"]
            fig = go.Figure()
            for f, v in per.items():
                fig.add_trace(go.Scatter(x=list(range(6)), y=v["scores_L0_to_L5"], mode="lines+markers", name=f, line=dict(width=1.6), marker=dict(size=5)))
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title=dict(text="Overall score by injected level (B01)", x=0, font=dict(size=13)),
                              xaxis=dict(title="level (0 = clean)", dtick=1, gridcolor=GRID), yaxis=dict(range=[60, 101], gridcolor=GRID), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            shift = pd.DataFrame({"condition": list(iv["mean_shift_by_condition"]), "mean |score shift|": list(iv["mean_shift_by_condition"].values())})
            st.plotly_chart(_bar(shift, "mean |score shift|", "condition", "Clean speech under a Where condition: score shift (pts, target < 3)", "#37b24d", 3.0, 380), use_container_width=True)
    lk, lo_ = RES / "leakage.csv", RES / "leakage_loso.csv"
    st.subheader("Leakage audit")
    st.caption("Can a classifier find the flaw from editing artifacts alone (clicks, noise-floor steps, flux spikes, exact repeats)? 0.5 = no. Pass mark 0.60.")
    if lk.exists() or lo_.exists():
        c1, c2 = st.columns(2)
        for col, f, title in ((c1, lk, "Held-out test windows (v2 audit)"), (c2, lo_, "Leave-one-speaker-out, train+dev speakers")):
            if f.exists():
                L = pd.read_csv(f)
                tot = L[L.scope == "ALL"].auc.iloc[0]
                col.metric(title, f"{tot:.3f}", delta="≤ 0.60 pass" if tot <= 0.6 else "above 0.60", delta_color="normal" if tot <= 0.6 else "inverse")
                col.plotly_chart(_bar(L[L.scope != "ALL"].sort_values("auc", ascending=False), "auc", "scope", "AUC per flaw", "#845ef7", 0.6, 460), use_container_width=True)
        st.caption("Per-flaw numbers on the held-out split rest on 8–24 windows each and are noisy; the leave-one-speaker-out figures use about 5× more.")
