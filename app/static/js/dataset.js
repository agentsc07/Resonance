import { $, api, esc } from "./main.js";

export async function initDataset() {
  const d = await api("/api/dataset");
  const T = d.tiles, tc = ["rgba(240,192,90,.38)", "rgba(124,92,255,.38)", "rgba(45,212,191,.32)", "rgba(251,113,133,.32)"];
  $("#tiles").innerHTML = [[T.clips, "clips"], [T.hours.toFixed(1), "hours of audio"], [T.baselines, "baseline voices"], [T.flaws, "delivery flaws"]]
    .map(([v, l], i) => `<div class="panel tile" style="--tc:${tc[i]}"><b>${v}</b><span>${l}</span></div>`).join("");
  const mx = Math.max(...d.grid.flatMap((g) => g.levels), 1);
  $("#heat").innerHTML = `<span></span>${[1, 2, 3, 4, 5].map((l) => `<span class="h">L${l}</span>`).join("")}` +
    d.grid.map((g) => `<span class="r">${esc(g.name)}</span>` + g.levels.map((v) => `<span class="c" style="--v:${(v / mx).toFixed(2)}">${v || ""}</span>`).join("")).join("");
  const clean = d.conditions.clean; const cond = Object.fromEntries(Object.entries(d.conditions).filter(([k]) => k !== "clean"));
  const cm = Math.max(...Object.values(cond));
  const label = { clean: "Clean", N20: "Babble 20 dB", N10: "Pink 10 dB", RVB: "Room reverb", PHN: "Phone band", MP3: "MP3 64k", GAIN: "Gain ±6 dB" };
  $("#conds").innerHTML = `<div class="dim" style="font-size:13px;margin-bottom:4px"><b class="mono" style="color:var(--ink)">${clean}</b> clips are clean. Added conditions:</div>` + Object.entries(cond).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<div class="hbar"><span>${label[k] || k}</span><span class="b"><i style="width:${(v / cm) * 100}%"></i></span><span class="mono dim">${v}</span></div>`).join("");
  $("#spk").innerHTML = d.speakers.map((s) => `<div class="row"><span class="id">${s.baseline}</span><span class="nm">${esc(s.origin)}<small>${s.gender === "F" ? "Female" : "Male"} · ${s.age} · ${esc(s.register)}</small></span><span class="tag">${s.split}</span></div>`).join("");
  const g = d.gates, col = (ok) => (ok ? "var(--good)" : "var(--warn)");
  $("#gates").innerHTML = [
    [`${(g.join_pass * 100).toFixed(0)}%`, `of ${g.clips_checked} clips pass the splice-click gate (${(g.joins_above_6db * 100).toFixed(1)}% of joins above 6 dB)`, g.join_pass >= 0.95],
    [g.leak_test?.toFixed(3) ?? "–", `leakage AUC, held-out windows (pass ≤ 0.60) · ${g.leak_loso?.toFixed(3) ?? "–"} leave-one-speaker-out`, (g.leak_test ?? 1) <= 0.6 && (g.leak_loso ?? 1) <= 0.6],
    [`${g.repro.identical}/${g.repro.total}`, `clips bit-identical on a clean Linux build; the rest differ by ≤ ${g.repro.max_lsb} LSB`, g.repro.identical === g.repro.total],
    ["0", esc(g.qa), true],
    ["Test", "speakers B05 and B08 held out; not used until the final run", true],
  ].map(([b, s, ok]) => `<div class="gate" style="--gc:${col(ok)}"><b>${b}</b><span>${s}</span></div>`).join("");

  let page = 0, q = "";
  const PS = 40;
  const render = () => {
    const f = d.manifest.filter((r) => !q || (r.id + r.baseline + r.flaws + r.split + r.cond).toLowerCase().includes(q));
    const rows = f.slice(page * PS, (page + 1) * PS);
    $("#mf").innerHTML = `<tr><th>Clip</th><th>Speaker</th><th>Split</th><th>Flaws</th><th>Level</th><th>Condition</th><th>Seconds</th></tr>` +
      rows.map((r) => `<tr><td class="mono">${esc(r.id)}</td><td>${r.baseline}</td><td>${r.split}</td><td>${esc(r.flaws || "–")}</td><td>${r.level || ""}</td><td>${r.cond || ""}</td><td class="mono">${r.dur}</td></tr>`).join("");
    $("#mf-n").textContent = `${f.length ? page * PS + 1 : 0}–${Math.min((page + 1) * PS, f.length)} of ${f.length}`;
    $("#mf-prev").disabled = page === 0; $("#mf-next").disabled = (page + 1) * PS >= f.length;
  };
  $("#mf-q").addEventListener("input", (e) => { q = e.target.value.toLowerCase(); page = 0; render(); });
  $("#mf-prev").onclick = () => { page--; render(); }; $("#mf-next").onclick = () => { page++; render(); };
  render();
}
