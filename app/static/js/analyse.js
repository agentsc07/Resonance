import { $, $$, api, esc, toast } from "./main.js";
import { Timeline, CAT_COLOR } from "./timeline.js";

const S = { source: "clip", clipId: null, uploadId: null, uploadName: "", mode: "free", genre: "", noFluency: false, experimental: false, rubric: null, textId: "", transcript: "", result: null, truth: false, busy: false, preset: null };
let M, tl, audio;
let stopAt = null, raf = 0;

export function initAnalyse(meta) {
  M = meta; audio = $("#player");
  tl = new Timeline($("#tl"), $("#tip"), { select: (id, play) => selectRegion(id, play), seek: (t) => playFrom(t) });
  buildPresets(); buildAdvanced(); wire();
  runPreset(M.presets[2] || M.presets[0]);
}

// ------------------------------------------------------------------ inputs
function buildPresets() {
  $("#presets").insertAdjacentHTML("beforeend", M.presets.map((p) => `<button class="preset" data-id="${p.id}"><b>${esc(p.title)}</b><span>${esc(p.sub)}</span></button>`).join(""));
  $$(".preset").forEach((b) => b.addEventListener("click", () => runPreset(M.presets.find((p) => p.id === b.dataset.id))));
}

function runPreset(p) {
  S.source = "clip"; S.clipId = p.clip; S.mode = p.mode; S.preset = p.id; S.transcript = ""; S.truth = false;
  $("#transcript").value = ""; setDrop(null); syncAdvanced(); analyse();
}

function wire() {
  const drop = $("#drop"), file = $("#file");
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => { if (ev.dataTransfer.files[0]) upload(ev.dataTransfer.files[0]); });
  drop.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); file.click(); } });
  file.addEventListener("change", () => file.files[0] && upload(file.files[0]));
  $("#run").addEventListener("click", () => { S.transcript = $("#transcript").value.trim(); if (S.source === "clip" && S.transcript) toast("The transcript applies to uploaded recordings; dataset clips already have their text."); analyse(); });
  $("#playall").addEventListener("click", () => (audio.paused ? playFrom(0) : stop()));
  $$("#lanesel button").forEach((b) => b.addEventListener("click", () => { $$("#lanesel button").forEach((x) => x.classList.toggle("on", x === b)); tl.setLane(b.dataset.k); }));
  $("#reveal").addEventListener("click", reveal);
  audio.addEventListener("ended", stop);
  document.addEventListener("keydown", (e) => {
    if (!$("#p-analyse").classList.contains("on") || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.key === " ") { e.preventDefault(); audio.paused ? playFrom(audio.currentTime || 0) : stop(); }
    if (e.key === "ArrowRight") step(1); if (e.key === "ArrowLeft") step(-1);
  });
}

async function upload(f) {
  try {
    $("#drop-t").textContent = "Uploading…";
    const fd = new FormData(); fd.append("file", f);
    const r = await api("/api/upload", { method: "POST", body: fd });
    S.source = "upload"; S.uploadId = r.id; S.uploadName = f.name; S.preset = null; S.truth = false;
    setDrop(f.name + " · " + r.duration + " s");
    S.transcript = $("#transcript").value.trim(); syncAdvanced(); analyse();
  } catch (e) { setDrop(null); toast(e.message); }
}
function setDrop(label) {
  $("#drop").classList.toggle("has", !!label);
  $("#drop-t").textContent = label || "Drop a recording or choose a file";
  $("#drop-s").textContent = label ? "Click to replace" : "WAV, MP3, M4A or FLAC · up to 60 MB";
}

// ------------------------------------------------------------------ analysis
async function analyse() {
  if (S.busy) return;
  S.busy = true; stop(); veil(true);
  const t0 = performance.now(), tick = setInterval(() => { $("#veil-s").textContent = `${Math.round((performance.now() - t0) / 1000)} s · ` + (S.source === "upload" ? "recognising speech, aligning, scoring (about 30 s for a new recording)" : "aligning, measuring, scoring"); }, 500);
  const body = { source: S.source, id: S.source === "clip" ? S.clipId : S.uploadId, mode: S.mode, genre: S.genre || null, no_fluency: S.noFluency, experimental: S.experimental, rubric: S.rubric, text_id: S.textId || null, transcript: S.transcript || null };
  try {
    S.result = await api("/api/analyse", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    render();
  } catch (e) { toast(e.message); }
  finally { clearInterval(tick); veil(false); S.busy = false; }
}
function veil(on) { $("#veil1").classList.toggle("on", on); $("#veil2").classList.toggle("on", on); $("#run").disabled = on; }

// ------------------------------------------------------------------ render
function scoreColors(v) { return v >= 90 ? ["#d6f27a", "#9ad12f", "rgba(182,224,74,.55)"] : v >= 75 ? ["#f8d98a", "#d79a2b", "rgba(240,192,90,.55)"] : v >= 60 ? ["#ffc46b", "#f5a623", "rgba(245,166,35,.55)"] : ["#ff9a8f", "#ff4d3f", "rgba(255,107,94,.55)"]; }

function render() {
  const R = S.result; if (!R) return;
  // gauge
  const [c0, c1, glow] = scoreColors(R.overall);
  $("#gg0").setAttribute("stop-color", c0); $("#gg1").setAttribute("stop-color", c1); $("#arc").style.setProperty("--glow", glow);
  const C = 527.8; $("#arc").style.strokeDashoffset = C; requestAnimationFrame(() => requestAnimationFrame(() => { $("#arc").style.strokeDashoffset = C * (1 - R.overall / 100); }));
  countUp($("#score"), R.overall); $("#band").textContent = R.band;
  // categories
  $("#cats").innerHTML = R.categories.map((c) => `<div class="cat" style="--cc:${CAT_COLOR[c.name]}"><span class="name"><i class="dot"></i>${c.name}</span><span class="bar"><i></i></span><span class="val">${c.score.toFixed(0)}${c.loss >= 0.5 ? `<s>−${c.loss.toFixed(0)}</s>` : ""}</span></div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => $$("#cats .bar i").forEach((el, i) => { el.style.width = R.categories[i].score + "%"; })));
  // summary + chips
  $("#summary").textContent = R.summary;
  const mode = M.modes.find((m) => m.id === R.mode), q = R.quality;
  const qc = { good: "var(--good)", fair: "var(--warn)", poor: "var(--bad)" }[q.badge];
  $("#meta").innerHTML = `<span class="chip">${mode.label}</span><span class="chip" style="color:${qc};border-color:${qc}"><i class="dot"></i>Recording ${q.badge} · SNR ${Math.round(q.snr_db)} dB</span>` +
    (R.text_checks ? "" : `<span class="chip" title="Without the text, skipped and misread words cannot be checked">Text checks off</span>`) +
    (R.mode === "cross" ? `<span class="chip" style="color:var(--warn)">Experimental</span>` : "") + (S.experimental ? `<span class="chip" style="color:var(--warn)">Experimental detectors on</span>` : "");
  // timeline
  tl.set(R.timeline, R.regions); tl.setTruth(null); $("#truth").classList.remove("on");
  loadAudio(R.audio_url);
  $("#lg-ref").hidden = R.mode === "free";
  $("#reveal").hidden = !R.clip; $("#reveal").textContent = "Reveal what was injected";
  // transcript, flaw card, advanced
  renderWords(); const first = R.regions.slice().sort((a, b) => b.points - a.points)[0];
  selectRegion(first ? first.id : null, false);
  renderTable(); syncAdvanced();
  $$(".preset").forEach((b) => b.classList.toggle("on", S.preset === b.dataset.id));
}

function countUp(el, to) {
  const from = parseFloat(el.textContent) || 0, t0 = performance.now(), dur = 900;
  const f = (t) => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = Math.round(from + (to - from) * e); if (k < 1) requestAnimationFrame(f); else el.textContent = Math.round(to); };
  requestAnimationFrame(f);
}

function renderWords() {
  const R = S.result;
  $("#words").innerHTML = R.words.map((w) => {
    const r = w.flag != null ? R.regions[w.flag] : null;
    return `<span class="w${r ? " f" : ""}${w.missed ? " miss" : ""}" data-i="${w.i}" ${r ? `data-r="${r.id}" style="--cc:${CAT_COLOR[r.category]}" title="${esc(r.name)}"` : ""}>${esc(w.w)}</span>`;
  }).join(" ");
  $$("#words .w.f").forEach((el) => el.addEventListener("click", () => selectRegion(+el.dataset.r, true)));
}

function selectRegion(id, play) {
  const R = S.result; S.sel = id; tl.select(id);
  $$("#words .w").forEach((w) => w.classList.toggle("sel", id != null && w.dataset.r == id));
  const el = $("#flaw");
  if (id == null) {
    el.className = "panel flaw clear";
    el.innerHTML = `<div><b style="font-family:var(--display);font-size:20px;color:var(--ink)">Nothing stood out</b><div style="margin-top:4px">Delivery stays inside the expected range from start to finish.</div></div>`;
    return;
  }
  const r = R.regions[id], col = CAT_COLOR[r.category], n = R.regions.length;
  const why = r.why.replace(/\s*\([^()]*severity[^()]*\)\.?$/, ".").replace(/^Words \d+–\d+ /, "");
  el.className = "panel flaw";
  el.innerHTML = `<div class="when"><b class="mono">${r.start.toFixed(1)}s</b><span>${(r.end - r.start).toFixed(1)} s long</span></div>
    <div><h3><span class="chip" style="color:${col};border-color:${col}"><i class="dot"></i>${r.category}</span>${esc(r.name)}</h3><p class="why">${esc(why)}</p>${r.tip ? `<p class="tip-t">${esc(r.tip)}</p>` : ""}</div>
    <div class="cost"><b>−${r.points.toFixed(1)}</b><span>points · severity ${r.severity.toFixed(1)} of 5</span><div style="display:flex;gap:6px;justify-content:flex-end"><button class="btn sm" id="fp" ${n < 2 ? "hidden" : ""} aria-label="Previous flaw">‹</button><button class="btn sm gold" id="fplay">▶ Play this</button><button class="btn sm" id="fn" ${n < 2 ? "hidden" : ""} aria-label="Next flaw">›</button></div></div>`;
  $("#fplay").onclick = () => playRegion(r); $("#fp").onclick = () => step(-1); $("#fn").onclick = () => step(1);
  if (play) playRegion(r);
}
function step(d) {
  const R = S.result; if (!R || !R.regions.length) return;
  const order = R.regions.slice().sort((a, b) => a.start - b.start), i = order.findIndex((r) => r.id === S.sel);
  selectRegion(order[(i + d + order.length) % order.length].id, true);
}

function renderTable() {
  const R = S.result;
  $("#flaws-table").innerHTML = `<tr><th>Time</th><th>Category</th><th>Flaw</th><th>Severity</th><th>Points lost</th><th>Why</th></tr>` +
    (R.regions.length ? R.regions.slice().sort((a, b) => b.points - a.points).map((r) => `<tr><td class="mono">${r.start.toFixed(1)}–${r.end.toFixed(1)} s</td><td>${r.category}</td><td>${esc(r.name)}</td><td class="mono">${r.severity.toFixed(1)}</td><td class="mono">${r.points.toFixed(1)}</td><td>${esc(r.why.replace(/^Words \d+–\d+ /, ""))}</td></tr>`).join("") : `<tr><td colspan="6" class="dim">Nothing flagged.</td></tr>`);
}

// ------------------------------------------------------------------ playback
let blobUrl = null;
async function loadAudio(url) {                          // a blob URL is fully seekable (a streamed response without Range support is not)
  try {
    const b = await (await fetch(url)).blob();
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(b); audio.src = blobUrl;
  } catch { audio.src = url; }
}
function playRegion(r) { playSpan(Math.max(0, r.start - 0.4), r.end + 0.4); }
function playFrom(t) { playSpan(t, null); }
function ready() {                                       // seeking before the audio has loaded is ignored by browsers
  if (audio.readyState >= 2 && audio.src.startsWith("blob:")) return Promise.resolve();
  return new Promise((res) => { const f = () => { audio.removeEventListener("canplay", f); res(); }; audio.addEventListener("canplay", f); audio.load(); });
}
async function playSpan(a, b) {
  stopAt = b; await ready(); audio.currentTime = a;
  audio.play().then(() => { cancelAnimationFrame(raf); loop(); $("#playall").textContent = "■ Stop"; }).catch(() => toast("The browser blocked playback. Click again to start the sound."));
}
function loop() {
  const t = audio.currentTime; tl.setPlayhead(t);
  const w = S.result?.words.find((x) => x.s <= t && t <= x.e); $$("#words .w.now").forEach((x) => x.classList.remove("now")); if (w) $(`#words .w[data-i="${w.i}"]`)?.classList.add("now");
  if (stopAt !== null && t >= stopAt) return stop();
  if (!audio.paused) raf = requestAnimationFrame(loop);
}
function stop() { audio.pause(); cancelAnimationFrame(raf); stopAt = null; tl.setPlayhead(null); $("#playall").textContent = "▶ Play all"; $$("#words .w.now").forEach((x) => x.classList.remove("now")); }

async function reveal() {
  const on = !S.truth; S.truth = on;
  if (on) {
    const t = await api("/api/truth/" + S.result.clip);
    tl.setTruth(t); const box = $("#truth"); box.classList.add("on");
    box.innerHTML = `<span class="eyebrow" style="align-self:center">Injected</span>` + (t.length ? t.map((x) => `<span class="chip">${esc(x.name)}${x.level ? " · level " + x.level : ""} · ${x.start.toFixed(1)}–${x.end.toFixed(1)} s</span>`).join("") : `<span class="chip">Nothing: this is a clean reading</span>`);
  } else { tl.setTruth(null); $("#truth").classList.remove("on"); }
  $("#reveal").textContent = on ? "Hide what was injected" : "Reveal what was injected";
}

// ------------------------------------------------------------------ advanced
function buildAdvanced() {
  $("#modes").innerHTML = M.modes.map((m) => `<button class="mode" data-m="${m.id}"><b>${m.label}<span class="mono dimmer" style="font-weight:500;font-size:11px">F1 ${m.f1}</span></b><span>${m.note}</span></button>`).join("");
  $$(".mode").forEach((b) => b.addEventListener("click", () => { S.mode = b.dataset.m; syncAdvanced(); analyse(); }));
  $("#genre").innerHTML = `<option value="">Automatic</option>` + M.genres.map((g) => `<option>${g}</option>`).join("");
  $("#genre").addEventListener("change", (e) => { S.genre = e.target.value; analyse(); });
  $("#textid").innerHTML = `<option value="">Automatic (use what is heard)</option>` + M.texts.map((t) => `<option value="${t.id}">${esc(t.label)}</option>`).join("");
  $("#textid").addEventListener("change", (e) => { S.textId = e.target.value; if (S.source === "upload") analyse(); });
  $("#nofl").addEventListener("change", (e) => { S.noFluency = e.target.checked; analyse(); });
  $("#exp").addEventListener("change", (e) => { S.experimental = e.target.checked; analyse(); });
  $("#rub").addEventListener("change", async (e) => { const f = e.target.files[0]; S.rubric = f ? await f.text() : null; $("#rub-n").textContent = f ? f.name : ""; analyse(); });
  const kinds = { "": "All clips", clean: "Clean take", flaw: "One flaw", multi: "Several flaws", condition: "Noisy / phone / room" };
  $("#d-take").innerHTML = M.takes.map((t) => `<option value="${t.take}">${t.baseline} · ${esc(t.accent)}, ${t.gender === "F" ? "female" : "male"}</option>`).join("");
  $("#d-kind").innerHTML = Object.entries(kinds).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  const fillClips = () => {
    const t = M.takes.find((x) => x.take === $("#d-take").value), k = $("#d-kind").value;
    $("#d-clip").innerHTML = t.clips.filter((c) => !k || c.kind === k).map((c) => `<option value="${c.id}">${esc(c.kind === "clean" ? "clean take" : c.kind === "condition" ? c.cond : c.flaw + (c.level ? " L" + c.level : ""))}</option>`).join("");
  };
  $("#d-take").addEventListener("change", fillClips); $("#d-kind").addEventListener("change", fillClips); fillClips();
  $("#d-go").addEventListener("click", () => { const id = $("#d-clip").value; if (!id) return; S.source = "clip"; S.clipId = id; S.preset = null; S.truth = false; setDrop(null); analyse(); });
  $("#ex-json").addEventListener("click", () => download(`${S.result.clip || "upload"}.score.json`, JSON.stringify(S.result.pred, null, 2), "application/json"));
  $("#ex-html").addEventListener("click", () => download(`${S.result.clip || "upload"}.report.html`, reportHtml(S.result), "text/html"));
}
function syncAdvanced() {
  $$(".mode").forEach((b) => b.classList.toggle("on", b.dataset.m === S.mode));
  $("#genre").value = S.genre || ""; $("#textid").value = S.textId || ""; $("#nofl").checked = S.noFluency; $("#exp").checked = S.experimental;
}
function download(name, text, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function reportHtml(R) {
  const rows = R.regions.map((r) => `<tr><td>${r.start.toFixed(1)}–${r.end.toFixed(1)}</td><td>${r.category}</td><td>${esc(r.name)}</td><td>${r.severity.toFixed(1)}</td><td>${r.points.toFixed(1)}</td><td>${esc(r.why)}</td></tr>`).join("");
  return `<!doctype html><meta charset="utf-8"><title>Delivery report</title><body style="font:14px system-ui;max-width:900px;margin:2rem auto"><h1>Delivery report</h1><h2>${R.overall} / 100 · ${R.band}</h2><p>${esc(R.summary)}</p><ul>${R.categories.map((c) => `<li>${c.name}: ${c.score}</li>`).join("")}</ul><table border="1" cellpadding="4" cellspacing="0"><tr><th>Time</th><th>Category</th><th>Flaw</th><th>Severity</th><th>Points lost</th><th>Why</th></tr>${rows}</table></body>`;
}
