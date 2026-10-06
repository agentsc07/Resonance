import { $, api, esc } from "./main.js";

const pct = (v) => (v == null ? "–" : v.toFixed(3));

export async function initAbout() {
  const a = await api("/api/about");
  const modes = [["same", "Same speaker", "Compared with the speaker's own clean reading. The validated upper bound."], ["free", "General", "No reference. The dashboard default."], ["cross", "Another speaker", "Experimental."]];
  const cols = ["train", "dev", "extra", "test"], names = { train: "Train", dev: "Dev", extra: "JFK 1962", test: "Test" };
  let rows = "";
  for (const [id, name, note] of modes) {
    const e = a.eval[id] || {};
    rows += `<tr><td><b>${name}</b><div class="dimmer" style="font-size:12px">${note}</div></td>` + cols.map((c) => `<td class="mono" style="white-space:nowrap">${e[c] ? pct(e[c].headline) : `<span class="dimmer">${c === "test" ? "not run" : "–"}</span>`}</td>`).join("") + `<td class="mono">${(e.train || {}).flaws ?? "–"}</td></tr>`;
  }
  $("#ev").innerHTML = `<h3>How well it finds the flaws</h3><div class="dim" style="font-size:13.5px;margin-bottom:12px">Event F1 at IoU 0.5 against the injected truth, headline flaws only. Thresholds are fitted on the train speakers; the test speakers have not been touched.</div>
    <div class="scroll-x"><table class="t"><tr><th>Mode</th>${cols.map((c) => `<th>${names[c]}</th>`).join("")}<th>Flaws</th></tr>${rows}</table></div>
    <div class="dim" style="font-size:13px;margin-top:14px">Experimental, hidden by default: ${Object.keys(a.experimental).join(", ") || "none"}. General mode does not reach its 0.4 bar and does not transfer to the noisy 1962 recording.</div>`;
  const ac = a.accept;
  $("#acc").innerHTML = ac ? `<h3>Checks on the score</h3><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:12px">
      <div><div class="num-big mono">${ac.flags_per_min.toFixed(2)}</div><div class="dim" style="font-size:12.5px">false flags per minute under noise, phone, room, codec (target ≤ 0.5)</div></div>
      <div><div class="num-big mono">${ac.locality.toFixed(1)}</div><div class="dim" style="font-size:12.5px">points lost in other categories (target &lt; 5)</div></div>
      <div><div class="num-big mono" style="color:var(--bad)">${ac.max_shift.toFixed(1)}</div><div class="dim" style="font-size:12.5px">worst clean-speech score shift, points (target &lt; 3, not met)</div></div></div>
    <div class="dim" style="font-size:13px;margin-top:14px">Leakage audit (can a classifier spot the edit from artifacts alone?): AUC ${pct(a.leak.test)} on held-out windows, ${pct(a.leak.loso)} leave-one-speaker-out. Pass mark 0.60.</div>` : "";
  $("#lim").innerHTML = `<h3>What it does not do</h3><ul class="limits"><li>Nobody has listened to the injected flaws yet: realism is unverified.</li><li>Eight studio voices aged 18–38 plus one 1962 voice. No slang, spontaneous speech or older speakers.</li><li>It scores departure from a chosen yardstick, not how good a speaker is. Do not use it to rank people.</li></ul>`;
  const L = a.links, link = (label, url, hint) => url ? `<a class="btn" href="${esc(url)}" target="_blank" rel="noopener">${label} ↗</a>` : `<span class="btn" style="opacity:.45;cursor:default" title="${hint}">${label} · to be added</span>`;
  $("#links").innerHTML = link("Repository", L.repo, "Public repository link") + link("Technical report", L.report && !L.report.startsWith("http") ? "" : L.report, "docs/technical_report.md in the repository") + link("Datasheet", L.datasheet && !L.datasheet.startsWith("http") ? "" : L.datasheet, "DATASHEET.md in the repository") + link("Dataset download", L.dataset, "Released zip with checksums") + link("Video", L.video, "Demo video");
}
