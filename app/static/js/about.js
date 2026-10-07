import { $, api, esc } from "./main.js";

const f3 = (v) => (v == null ? "–" : v.toFixed(3));
const f2 = (v) => (v == null ? "–" : (v + 1e-9).toFixed(2));

export async function initAbout() {
  const a = await api("/api/about");
  const same = (a.eval.same || {}), free = (a.eval.free || {}), ac = a.accept;

  // three numbers, as things a person can picture (precision and recall at IoU 0.5 on the headline flaws)
  const of10 = (v) => (v == null ? "–" : Math.round(v * 10)), mins = ac ? Math.round(1 / ac.flags_per_min) : null;
  const ICON = {
    find: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l5 5"/></svg>',
    alarm: '<svg viewBox="0 0 24 24"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>',
    solo: '<svg viewBox="0 0 24 24"><path d="M4 12h2M8 7v10M12 4v16M16 8v8M20 11v2"/></svg>',
  };
  $("#big3").innerHTML = [
    [ICON.find, `${of10(same.train?.r)}<small> of 10</small>`, "flaws found", `${of10(same.train?.p)} of 10 flags are real. Needs a clean reading to compare with (yours or the one you upload).`],
    [ICON.alarm, mins ? `1<small> per ${mins} min</small>` : "–", "wrong flag on clean speech", `Even recorded with noise, a phone line or an echoey room. Our limit was one per 2 minutes.`],
    [ICON.solo, `${of10(free.train?.r)}<small> of 10</small>`, "flaws found with no reference", `Only the text and the recording: ${of10(free.train?.p)} of 10 flags are real. Honest and much weaker: that is what the app does by default.`],
  ].map(([ic, n, l, s]) => `<div class="panel big"><span class="ic">${ic}</span><b class="mono">${n}</b><em>${l}</em><small>${s}</small></div>`).join("");

  $("#gloss").innerHTML = `<b>How it is scored.</b> A flag counts as right when it overlaps the true flaw by half or more. “Found” is recall, “real” is precision, F1 is the two combined (${f2(same.train?.headline)} with a reference, ${f2(free.train?.headline)} without).`;

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

  $("#lim").innerHTML = `<h3>What it does not do</h3><ul class="limits"><li>Only one listener has checked the injected flaws by ear, on a handful of clips: realism is only partly verified.</li><li>Eight studio voices aged 18–38 plus one 1962 voice. No slang, spontaneous speech or older speakers.</li><li>It scores departure from a chosen yardstick, not how good a speaker is. Do not use it to rank people.</li></ul>`;

  const L = a.links, link = (label, url, hint) => url ? `<a class="btn" href="${esc(url)}" target="_blank" rel="noopener">${label} ↗</a>` : `<span class="btn" style="opacity:.45;cursor:default" title="${hint}">${label} · to be added</span>`;
  const abs = (u) => (u && /^https?:/.test(u) ? u : "");
  $("#links").innerHTML = link("Repository", L.repo, "Public repository link") + link("Technical report", abs(L.report), "docs/technical_report.md in the repository") + link("Datasheet", abs(L.datasheet), "DATASHEET.md in the repository") + link("Dataset download", L.dataset, "Released zip with checksums") + link("Video", L.video, "Demo video");
}
