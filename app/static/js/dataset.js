import { $, api, esc } from "./main.js";

const P = (d) => `<svg viewBox="0 0 24 24">${d}</svg>`;
const CATICON = {
  Pacing: P('<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/>'),
  Pausing: P('<path d="M9 5v14M15 5v14"/>'),
  Intonation: P('<path d="M3 15c3-8 5-8 8 0s5 8 10-4"/>'),
  Volume: P('<path d="M4 10v4h4l5 4V6L8 10z"/><path d="M17 9a4 4 0 0 1 0 6"/>'),
  Fluency: P('<path d="M5 12h2M10 8v8M14 5v14M18 9v6"/>'),
  Clarity: P('<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>'),
  "Text fidelity": P('<path d="M5 6h14M5 12h8M5 18h11"/>'),
};
const COND = { N20: "Background chatter", N10: "Steady hiss", RVB: "Echoey room", PHN: "Phone-call quality", MP3: "Compressed (MP3)", GAIN: "Louder or quieter recording" };

export async function initDataset() {
  const d = await api("/api/dataset");
  const T = d.tiles;
  $("#tiles").innerHTML = [[T.clips, "clips"], [T.hours.toFixed(1), "hours of audio"], [T.baselines, "baseline voices"], [T.flaws, "kinds of flaw"]].map(([v, l]) => `<div class="panel tile"><b>${v}</b><span>${l}</span></div>`).join("");
  $("#catlist").innerHTML = d.categories.map((c) => `<div><i class="cic">${CATICON[c.name] || ""}</i><div><b>${esc(c.name)}</b><span>${esc(c.words)}</span></div></div>`).join("");

  // conditions, in plain words (clean speech is the baseline, not a bar)
  const cond = Object.entries(d.conditions).filter(([k]) => k !== "clean"), cm = Math.max(...cond.map((c) => c[1]));
  $("#conds").innerHTML = `<div class="dim" style="font-size:13px"><b class="mono" style="color:var(--ink)">${d.conditions.clean}</b> clips are clean recordings. The rest add:</div>` +
    cond.sort((a, b) => b[1] - a[1]).map(([k, v]) => `<div class="hbar"><span>${COND[k] || k}</span><span class="b"><i style="width:${(v / cm) * 100}%"></i></span><span class="mono dim">${v}</span></div>`).join("");

  // speakers: no repeated "neutral"
  $("#spk").innerHTML = d.speakers.map((s) => `<div class="row"><span class="id">${s.baseline}</span><span class="nm">${esc(s.origin)}<small>${s.gender === "F" ? "Female" : "Male"} · ${s.age}${s.register && s.register !== "neutral" ? " · " + esc(s.register) : ""}</small></span><span class="tag">${s.split}</span></div>`).join("");

  // quality gates
  const g = d.gates, col = (ok) => (ok ? "var(--good)" : "var(--warn)");
  $("#gates").innerHTML = [
    [`${(g.join_pass * 100).toFixed(0)}%`, `of edited clips have clean joins: no click or jump where we spliced`, g.join_pass >= 0.95],
    [g.leak_test?.toFixed(3) ?? "–", `a classifier trying to spot the edits from traces alone scores this (0.5 means it cannot tell; we require 0.60 or less)`, (g.leak_test ?? 1) <= 0.6],
    [`${g.repro.identical}/${g.repro.total}`, `clips rebuild identically on a fresh Linux machine; the rest differ by one audio step at most`, true],
    ["0", esc(g.qa), true],
    ["Held out", "two speakers are reserved for the final test and never used for tuning", true],
  ].map(([b, s, ok]) => `<div class="gate" style="--gc:${col(ok)}"><b>${b}</b><span>${s}</span></div>`).join("");

  // manifest
  let page = 0, q = "";
  const PS = 40;
  const render = () => {
    const f = d.manifest.filter((r) => !q || (r.id + r.baseline + r.flaws + r.split + r.cond).toLowerCase().includes(q));
    const rows = f.slice(page * PS, (page + 1) * PS);
    $("#mf").innerHTML = `<tr><th>Clip</th><th>Speaker</th><th>Split</th><th>Flaws</th><th>Level</th><th>Condition</th><th>Seconds</th><th></th></tr>` +
      rows.map((r) => `<tr><td class="mono">${esc(r.id)}</td><td>${r.baseline}</td><td>${r.split}</td><td>${esc(r.flaws || "–")}</td><td>${r.level || ""}</td><td>${COND[r.cond] || r.cond || ""}</td><td class="mono">${r.dur}</td><td><button class="play sm" data-id="${esc(r.id)}" data-dur="${r.dur}" aria-label="Play clip ${esc(r.id)}">▶</button></td></tr>`).join("");
    $("#mf").querySelectorAll("button.play").forEach((b) => (b.onclick = () => playSpan(b, b.dataset.id, [0, +b.dataset.dur], 0, 0.1)));
    $("#mf-n").textContent = `${f.length ? page * PS + 1 : 0}–${Math.min((page + 1) * PS, f.length)} of ${f.length}`;
    $("#mf-prev").disabled = page === 0; $("#mf-next").disabled = (page + 1) * PS >= f.length;
  };
  $("#mf-q").addEventListener("input", (e) => { q = e.target.value.toLowerCase(); page = 0; render(); });
  $("#mf-prev").onclick = () => { page--; render(); }; $("#mf-next").onclick = () => { page++; render(); };
  render();
  initPair();
}

// ---- Hear a pair: the same reading, clean and with one flaw, each plays only the changed moment
const blobs = {};
let cur = null, timer = null;
async function blobFor(id) {
  if (!blobs[id]) blobs[id] = URL.createObjectURL(await (await fetch(`/api/audio/clip/${id}.wav`)).blob());
  return blobs[id];
}
async function playSpan(btn, id, span, lead = 2.0, tail = 2.0) {
  if (cur) { cur.audio.pause(); cur.btn.classList.remove("on"); cur.btn.textContent = "▶"; clearInterval(timer); if (cur.btn === btn) { cur = null; return; } }
  const a = new Audio(await blobFor(id));
  await new Promise((r) => { a.addEventListener("canplay", r, { once: true }); a.load(); });
  a.currentTime = Math.max(0, span[0] - lead);
  await a.play(); btn.classList.add("on"); btn.textContent = "■"; cur = { audio: a, btn };
  const end = span[1] + tail;
  timer = setInterval(() => { if (a.currentTime >= end || a.ended) { a.pause(); btn.classList.remove("on"); btn.textContent = "▶"; clearInterval(timer); cur = null; } }, 60);
}
async function initPair() {
  let i = 0, P = null;
  const load = async () => {
    P = await api("/api/pair/" + i);
    $("#pair-sub").textContent = "A few sample pairs of modifications: the original recording and the same recording with one flaw added. Press play on each to hear the difference.";
    $("#pc-s").textContent = `Original, unchanged · around ${P.clean_span[0].toFixed(0)} s`;
    $("#pf-t").textContent = `Modified: ${P.name.toLowerCase()}`;
    $("#pf-s").textContent = `Flaw added around ${P.flawed_span[0].toFixed(0)} s · strength ${P.level} of 5`;
    $("#pn").textContent = `${P.i + 1} of ${P.n}`;
  };
  await load();
  $("#pc").onclick = (e) => playSpan(e.currentTarget, P.clean, P.clean_span);
  $("#pf").onclick = (e) => playSpan(e.currentTarget, P.flawed, P.flawed_span);
  $("#pnext").onclick = async () => { if (cur) { cur.audio.pause(); cur.btn.classList.remove("on"); cur.btn.textContent = "▶"; clearInterval(timer); cur = null; } i = (i + 1) % P.n; await load(); };
}
