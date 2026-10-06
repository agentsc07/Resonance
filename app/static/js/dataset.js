import { $, api, esc } from "./main.js";

const COND = { N20: "Background chatter", N10: "Steady hiss", RVB: "Echoey room", PHN: "Phone-call quality", MP3: "Compressed (MP3)", GAIN: "Louder or quieter recording" };

export async function initDataset() {
  const d = await api("/api/dataset");
  const T = d.tiles;
  $("#tiles").innerHTML = [[T.clips, "clips"], [T.hours.toFixed(1), "hours of audio"], [T.baselines, "baseline voices"], [T.flaws, "kinds of flaw"]].map(([v, l]) => `<div class="panel tile"><b>${v}</b><span>${l}</span></div>`).join("");
  $("#catlist").innerHTML = d.categories.map((c) => `<div><b>${esc(c.name)}</b><span>${esc(c.words)}</span></div>`).join("");

  // conditions, in plain words (clean speech is the baseline, not a bar)
  const cond = Object.entries(d.conditions).filter(([k]) => k !== "clean"), cm = Math.max(...cond.map((c) => c[1]));
  $("#conds").innerHTML = `<div class="dim" style="font-size:13px"><b class="mono" style="color:var(--ink)">${d.conditions.clean}</b> clips are clean recordings. The rest add:</div>` +
    cond.sort((a, b) => b[1] - a[1]).map(([k, v]) => `<div class="hbar"><span>${COND[k] || k}</span><span class="b"><i style="width:${(v / cm) * 100}%"></i></span><span class="mono dim">${v}</span></div>`).join("");

  // speakers: no repeated "neutral"
  $("#spk").innerHTML = d.speakers.map((s) => `<div class="row"><span class="id">${s.baseline}</span><span class="nm">${esc(s.origin)}<small>${s.gender === "F" ? "Female" : "Male"} · ${s.age}${s.register && s.register !== "neutral" ? " · " + esc(s.register) : ""}</small></span><span class="tag">${s.split}</span></div>`).join("");

  // quality gates
  const g = d.gates, col = (ok) => (ok ? "var(--good)" : "var(--warn)");
  $("#gates").innerHTML = [
    [`${(g.join_pass * 100).toFixed(0)}%`, `of ${g.clips_checked} edited clips pass the splice-click check (${(g.joins_above_6db * 100).toFixed(1)}% of joins above 6 dB)`, g.join_pass >= 0.95],
    [g.leak_test?.toFixed(3) ?? "–", `can a classifier spot the edit from artifacts alone? 0.5 = no, pass ≤ 0.60 · ${g.leak_loso?.toFixed(3) ?? "–"} leave-one-speaker-out`, (g.leak_test ?? 1) <= 0.6 && (g.leak_loso ?? 1) <= 0.6],
    [`${g.repro.identical}/${g.repro.total}`, `clips come out bit-identical on a clean Linux build; the rest differ by ≤ ${g.repro.max_lsb} step`, g.repro.identical === g.repro.total],
    ["0", esc(g.qa), true],
    ["Held out", "speakers B05 and B08 are not used until the final run", true],
  ].map(([b, s, ok]) => `<div class="gate" style="--gc:${col(ok)}"><b>${b}</b><span>${s}</span></div>`).join("");

  // manifest
  let page = 0, q = "";
  const PS = 40;
  const render = () => {
    const f = d.manifest.filter((r) => !q || (r.id + r.baseline + r.flaws + r.split + r.cond).toLowerCase().includes(q));
    const rows = f.slice(page * PS, (page + 1) * PS);
    $("#mf").innerHTML = `<tr><th>Clip</th><th>Speaker</th><th>Split</th><th>Flaws</th><th>Level</th><th>Condition</th><th>Seconds</th></tr>` +
      rows.map((r) => `<tr><td class="mono">${esc(r.id)}</td><td>${r.baseline}</td><td>${r.split}</td><td>${esc(r.flaws || "–")}</td><td>${r.level || ""}</td><td>${COND[r.cond] || r.cond || ""}</td><td class="mono">${r.dur}</td></tr>`).join("");
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
async function playSpan(btn, id, span) {
  if (cur) { cur.audio.pause(); cur.btn.classList.remove("on"); cur.btn.textContent = "▶"; clearInterval(timer); if (cur.btn === btn) { cur = null; return; } }
  const a = new Audio(await blobFor(id));
  await new Promise((r) => { a.addEventListener("canplay", r, { once: true }); a.load(); });
  a.currentTime = Math.max(0, span[0] - 0.7);
  await a.play(); btn.classList.add("on"); btn.textContent = "■"; cur = { audio: a, btn };
  const end = span[1] + 0.7;
  timer = setInterval(() => { if (a.currentTime >= end || a.ended) { a.pause(); btn.classList.remove("on"); btn.textContent = "▶"; clearInterval(timer); cur = null; } }, 60);
}
async function initPair() {
  let i = 0, P = null;
  const load = async () => {
    P = await api("/api/pair/" + i);
    $("#pair-sub").textContent = "The same reading twice. Each button plays just the moment that was changed.";
    $("#pc-s").textContent = `at ${P.clean_span[0].toFixed(0)} s, as recorded`;
    $("#pf-t").textContent = `With a flaw: ${P.name.toLowerCase()}`;
    $("#pf-s").textContent = `at ${P.flawed_span[0].toFixed(0)} s, strength ${P.level} of 5`;
    $("#pn").textContent = `${P.i + 1} of ${P.n}`;
  };
  await load();
  $("#pc").onclick = (e) => playSpan(e.currentTarget, P.clean, P.clean_span);
  $("#pf").onclick = (e) => playSpan(e.currentTarget, P.flawed, P.flawed_span);
  $("#pnext").onclick = async () => { if (cur) { cur.audio.pause(); cur.btn.classList.remove("on"); cur.btn.textContent = "▶"; clearInterval(timer); cur = null; } i = (i + 1) % P.n; await load(); };
}
