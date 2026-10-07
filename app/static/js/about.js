import { $, api, esc } from "./main.js";

const f2 = (v) => (v == null ? "–" : (v + 1e-9).toFixed(2));
const ICON = {
  rise: '<svg viewBox="0 0 24 24"><path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/></svg>',
  calm: '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L16 9.5"/></svg>',
  find: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l5 5"/></svg>',
};
const AREAS = {
  Pacing: "Rushing or dragging", Pausing: "Pauses in the wrong place, or none where one belongs", Intonation: "A flat voice, a rise on a statement",
  Volume: "Trailing off, sudden loud stretches", Fluency: "Filler sounds, false starts, hesitation", Clarity: "Slurred consonants", "Text fidelity": "Skipped or misread words",
};

export async function initAbout() {
  const a = await api("/api/about");
  const same = a.eval.same || {}, ac = a.accept, rub = a.rubric;
  const dose = ac ? Object.values(ac.dose_per_flaw).filter((v) => v != null) : [], ok = dose.filter((v) => v <= -0.9).length;
  const of10 = (v) => (v == null ? "–" : Math.round(v * 10)), lim = ac ? Math.ceil(ac.flags_per_min * 10) / 10 : null;

  // three headline numbers, each a strength we can back with a number
  $("#big3").innerHTML = [
    [ICON.rise, `${ok}<small> of ${dose.length}</small>`, "flaw types score steadily lower as they get worse", "Make a flaw stronger and the score falls with it: the scoring behaves like a rubric, not noise."],
    [ICON.calm, `&lt; ${lim ?? "–"}<small> false flags / min</small>`, "on clean speech, even in rough recordings", "Noise, a phone line, an echoey room or MP3 compression do not make it invent flaws."],
    [ICON.find, `${of10(same.train?.r)}<small> of 10</small>`, "injected flaws found, blind", `With a clean reading to compare against, and ${of10(same.train?.p)} in 10 of its flags are real. Missing pauses: every flag real, 3 in 4 found.`],
  ].map(([ic, n, l, s]) => `<div class="panel big"><span class="ic">${ic}</span><b class="mono">${n}</b><em>${l}</em><small>${s}</small></div>`).join("");

  // how scoring works: three steps
  $("#how").innerHTML = `<h3>How a score is made</h3><div class="steps">
    <div><i>1</i><b>Find</b><span>Every moment where delivery departs from a clean reading is located to the fraction of a second and named: 15 kinds of flaw in 7 areas.</span></div>
    <div><i>2</i><b>Weigh</b><span>Each moment costs points by how strong it is (1 to 5) and how long it lasts. One egregious flaw costs more than several mild ones.</span></div>
    <div><i>3</i><b>Score</b><span>Each area scores 0 to 100. The overall score is 60% the average and 40% the weakest area, so a badly hurt area cannot hide.</span></div></div>`;

  // rubric
  const w = rub.weights, genres = Object.keys(w), cats = rub.categories, b = rub.bands;
  $("#rub-body").innerHTML = `<div class="dim" style="font-size:14px;margin-bottom:12px">Area score = 100 · e<sup>−points lost ÷ ${rub.tau}</sup>. Overall = ${Math.round(rub.blend * 100)}% weighted average + ${Math.round((1 - rub.blend) * 100)}% weakest area.
      Bands: <b>Polished</b> ${b.polished}+, <b>Strong</b> ${b.strong}+, <b>Noticeable flaws</b> ${b.noticeable}+, <b>Needs work</b> below. An area under 50 rules out Polished and Strong. The weights depend on the kind of speaking.</div>
    <div class="scroll-x"><table class="t"><tr><th>Area</th><th>What it covers</th>${genres.map((g) => `<th>${esc(g)}</th>`).join("")}</tr>` +
    cats.map((c) => `<tr><td><b>${esc(c)}</b></td><td class="dim">${esc(AREAS[c] || "")}</td>${genres.map((g) => `<td class="mono">${Math.round((w[g][c] || 0) * 100)}%</td>`).join("")}</tr>`).join("") + `</table></div>`;

  const L = a.links, link = (label, url, hint) => url ? `<a class="btn" href="${esc(url)}" target="_blank" rel="noopener">${label} ↗</a>` : `<span class="btn" style="opacity:.45;cursor:default" title="${hint}">${label} · coming</span>`;
  const abs = (u) => (u && /^https?:/.test(u) ? u : "");
  $("#links").innerHTML = link("Repository", L.repo, "Public repository link") + link("Technical report", abs(L.report), "docs/technical_report.md in the repository") + link("Datasheet", abs(L.datasheet), "DATASHEET.md in the repository") + link("Dataset download", L.dataset, "Released zip with checksums") + link("Video", L.video, "Demo video");
}
