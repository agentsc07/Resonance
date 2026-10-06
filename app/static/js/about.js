import { $, api, esc } from "./main.js";

const f3 = (v) => (v == null ? "–" : v.toFixed(3));
const f2 = (v) => (v == null ? "–" : (v + 1e-9).toFixed(2));

export async function initAbout() {
  const a = await api("/api/about");
  const same = (a.eval.same || {}), free = (a.eval.free || {}), ac = a.accept;

  // three numbers, in plain words
  $("#big3").innerHTML = [
    [f2(same.train?.headline), "How well it finds flaws when it has the speaker's own clean reading to compare with.", `Score from 0 to 1, where 1 is perfect. On the held-in dev speaker: ${f2(same.dev?.headline)}. On a 1962 speech it never saw: ${f2(same.extra?.headline)}.`],
    [ac ? ac.flags_per_min.toFixed(2) : "–", "False alarms per minute on clean speech recorded in poor conditions: noise, a phone line, an echoey room.", `Lower is better. Our limit was 0.5. The score itself still moves more than we would like under noise (worst case ${ac ? ac.max_shift.toFixed(1) : "–"} points).`],
    [f2(free.train?.headline), "How well it does with no reference at all, only the text and the recording.", `Much weaker: dev ${f2(free.dev?.headline)}, and ${f2(free.extra?.headline)} on the noisy 1962 speech. It is the default in the app because it needs nothing from you.`],
  ].map(([n, p, s]) => `<div class="panel big"><b class="mono">${n}</b><p>${p}</p><small>${s}</small></div>`).join("");

  $("#gloss").innerHTML = `<b>F1</b>: one number from 0 to 1 that is high only when the engine finds the flaws <i>and</i> avoids false alarms. &nbsp; <b>IoU</b>: how much a flagged moment overlaps the true one; we count a hit at half or more. &nbsp; <b>AUC</b>: how well a classifier can tell edited audio from untouched audio using only editing traces; 0.5 means it cannot, and 0.60 is our pass mark.`;

  // full results (collapsed)
  const modes = [["same", "Same speaker", "Compared with the speaker's own clean reading. The validated upper bound."], ["free", "General", "No reference. The app's default."], ["cross", "Another speaker", "Experimental."]];
  const cols = ["train", "dev", "extra", "test"], names = { train: "Train", dev: "Dev", extra: "JFK 1962", test: "Test" };
  let rows = "";
  for (const [id, name, note] of modes) {
    const e = a.eval[id] || {};
    rows += `<tr><td><b>${name}</b><div class="dimmer" style="font-size:12px">${note}</div></td>` + cols.map((c) => `<td class="mono" style="white-space:nowrap">${e[c] ? f3(e[c].headline) : `<span class="dimmer">${c === "test" ? "not run" : "–"}</span>`}</td>`).join("") + `<td class="mono">${(e.train || {}).flaws ?? "–"}</td></tr>`;
  }
  $("#ev").innerHTML = `<div class="dim" style="font-size:13.5px;margin-bottom:10px">Event F1 at IoU 0.5 against the injected truth, headline flaws only. Thresholds are fitted on the train speakers; the test speakers have not been touched.</div>
    <div class="scroll-x"><table class="t"><tr><th>Mode</th>${cols.map((c) => `<th>${names[c]}</th>`).join("")}<th>Flaws</th></tr>${rows}</table></div>
    <div class="dim" style="font-size:13px;margin-top:12px">Experimental, hidden by default: ${Object.keys(a.experimental).join(", ") || "none"}.</div>`;
  $("#acc").innerHTML = ac ? `<div class="dim" style="font-size:13.5px;margin:16px 0 8px">Checks on the score (same speaker)</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
      <div><div class="mono" style="font:700 26px var(--display)">${ac.flags_per_min.toFixed(2)}</div><div class="dim" style="font-size:12.5px">false flags per minute under noise, phone, room, codec (target ≤ 0.5)</div></div>
      <div><div class="mono" style="font:700 26px var(--display)">${ac.locality.toFixed(1)}</div><div class="dim" style="font-size:12.5px">points lost in other areas (target &lt; 5)</div></div>
      <div><div class="mono" style="font:700 26px var(--display);color:var(--coral)">${ac.max_shift.toFixed(1)}</div><div class="dim" style="font-size:12.5px">worst clean-speech score shift (target &lt; 3, not met)</div></div></div>
    <div class="dim" style="font-size:13px;margin-top:12px">Leakage audit: AUC ${f3(a.leak.test)} on held-out windows, ${f3(a.leak.loso)} leave-one-speaker-out. Pass mark 0.60.</div>` : "";

  $("#lim").innerHTML = `<h3>What it does not do</h3><ul class="limits"><li>Nobody has listened to the injected flaws yet: realism is unverified.</li><li>Eight studio voices aged 18–38 plus one 1962 voice. No slang, spontaneous speech or older speakers.</li><li>It scores departure from a chosen yardstick, not how good a speaker is. Do not use it to rank people.</li></ul>`;

  const L = a.links, link = (label, url, hint) => url ? `<a class="btn" href="${esc(url)}" target="_blank" rel="noopener">${label} ↗</a>` : `<span class="btn" style="opacity:.45;cursor:default" title="${hint}">${label} · to be added</span>`;
  const abs = (u) => (u && /^https?:/.test(u) ? u : "");
  $("#links").innerHTML = link("Repository", L.repo, "Public repository link") + link("Technical report", abs(L.report), "docs/technical_report.md in the repository") + link("Datasheet", abs(L.datasheet), "DATASHEET.md in the repository") + link("Dataset download", L.dataset, "Released zip with checksums") + link("Video", L.video, "Demo video");
}
