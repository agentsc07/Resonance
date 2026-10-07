// Listening study: SPACE marks a slip at the playhead, each mark gets an area, then an overall 1-10 score. Engine output is never shown or requested here.
const $ = (s) => document.querySelector(s);
const api = async (p, o) => { const r = await fetch(p, o); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); };
const fmt = (t) => { t = Math.max(0, t); return Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0"); };
let rater = "", areas = [], clips = [], idx = 0, marks = [], score = 0, peaks = null, dur = 0, raf = 0;
const au = new Audio(); au.preload = "auto";
const cv = $("#wave"), cx = cv.getContext("2d");

$("#nm").value = (() => { try { return localStorage.getItem("flawline_rater") || ""; } catch { return ""; } })();
$("#start").onclick = async () => {
  try {
    const d = await api("/api/human/clips?rater=" + encodeURIComponent($("#nm").value));
    rater = d.rater; areas = d.areas; clips = d.clips;
    try { localStorage.setItem("flawline_rater", $("#nm").value); } catch { /* optional */ }
    $("#who").textContent = "Rater: " + rater;
    $("#s-name").classList.add("hide");
    idx = Math.max(0, clips.findIndex((c) => !c.done));
    if (clips.every((c) => c.done)) return done();
    load();
  } catch (e) { $("#e0").textContent = e.message; }
};
$("#nm").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#start").click(); });

function done() { $("#s-clip").classList.add("hide"); $("#s-done").classList.remove("hide"); }

async function load() {
  au.pause(); cancelAnimationFrame(raf);
  $("#s-clip").classList.remove("hide");
  marks = []; score = 0; $("#note").value = ""; $("#e1").textContent = ""; peaks = null;
  $("#prog").textContent = `Reading ${idx + 1} of ${clips.length}`;
  $("#barfill").style.width = (100 * idx / clips.length) + "%";
  $("#play").textContent = "▶ Play";
  drawScores(); drawMarks();
  const url = "/api/human/audio/" + encodeURIComponent(clips[idx].clip);
  au.src = url;
  try {   // waveform from the decoded file
    const buf = await (await fetch(url)).arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const a = await ac.decodeAudioData(buf); ac.close();
    const ch = a.getChannelData(0), n = 900, step = Math.floor(ch.length / n);
    peaks = Array.from({ length: n }, (_, i) => { let m = 0; for (let k = i * step; k < (i + 1) * step; k++) m = Math.max(m, Math.abs(ch[k])); return m; });
    const mx = Math.max(...peaks, 1e-6); peaks = peaks.map((p) => p / mx);
    dur = a.duration;
  } catch { dur = 0; }
  au.onloadedmetadata = () => { dur = au.duration || dur; draw(); };
  draw();
}

function size() { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; if (cv.width !== Math.round(r.width * d)) { cv.width = Math.round(r.width * d); cv.height = Math.round(150 * d); } return d; }
function draw() {
  const d = size(), W = cv.width, H = cv.height, D = dur || au.duration || 1;
  cx.clearRect(0, 0, W, H);
  if (peaks) {
    cx.fillStyle = "rgba(240,192,90,.45)";
    const bw = W / peaks.length;
    peaks.forEach((p, i) => { const h = Math.max(2 * d, p * (H * 0.86)); cx.fillRect(i * bw, (H - h) / 2, Math.max(1, bw - 1), h); });
  }
  cx.strokeStyle = "#ff6b5e"; cx.lineWidth = 2 * d;
  marks.forEach((m) => { const x = m.t_s / D * W; cx.beginPath(); cx.moveTo(x, 0); cx.lineTo(x, H); cx.stroke(); });
  const x = (au.currentTime / D) * W;
  cx.strokeStyle = "#f2f0ea"; cx.lineWidth = 2 * d; cx.beginPath(); cx.moveTo(x, 0); cx.lineTo(x, H); cx.stroke();
  $("#clock").textContent = fmt(au.currentTime) + " / " + fmt(D);
}
function loop() { draw(); if (!au.paused) raf = requestAnimationFrame(loop); }
au.onplay = () => { $("#play").textContent = "❚❚ Pause"; loop(); };
au.onpause = () => { $("#play").textContent = "▶ Play"; draw(); };
au.onended = () => { $("#play").textContent = "▶ Play"; draw(); };
const toggle = () => (au.paused ? au.play() : au.pause());
$("#play").onclick = toggle;
cv.onclick = (e) => { const r = cv.getBoundingClientRect(); au.currentTime = ((e.clientX - r.left) / r.width) * (dur || au.duration || 0); draw(); };
addEventListener("resize", draw);

function addMark() {
  const t = au.currentTime;
  marks.push({ t_s: +t.toFixed(2), pressed_s: +t.toFixed(3), area: "" });
  marks.sort((a, b) => a.t_s - b.t_s);
  drawMarks(); draw();
}
function drawMarks() {
  const box = $("#marks"); box.innerHTML = "";
  $("#nomarks").style.display = marks.length ? "none" : "";
  marks.forEach((m, i) => {
    const row = document.createElement("div"); row.className = "mk";
    row.innerHTML = `<span class="t">${fmt(m.t_s)}</span>
      <select>${'<option value="">Area…</option>' + areas.map((a) => `<option ${a === m.area ? "selected" : ""}>${a}</option>`).join("")}</select>
      <button data-d="-0.25" title="Earlier by 0.25 s">−0.25 s</button><button data-d="0.25" title="Later by 0.25 s">+0.25 s</button>
      <button data-x="1" title="Delete this mark">Delete</button>`;
    const sel = row.querySelector("select");
    sel.onchange = () => { m.area = sel.value; sel.classList.toggle("bad", !m.area); };
    sel.classList.toggle("bad", !m.area && m.touched);
    row.querySelector(".t").onclick = () => { au.currentTime = Math.max(0, m.t_s - 1.5); au.play(); };
    row.querySelectorAll("button").forEach((b) => b.onclick = () => {
      if (b.dataset.x) marks.splice(i, 1); else m.t_s = Math.max(0, +(m.t_s + parseFloat(b.dataset.d)).toFixed(2));
      marks.sort((a, c) => a.t_s - c.t_s); drawMarks(); draw();
    });
    box.appendChild(row);
  });
}
function drawScores() {
  const box = $("#scores"); box.innerHTML = "";
  for (let k = 1; k <= 10; k++) { const b = document.createElement("button"); b.textContent = k; b.className = k === score ? "on" : ""; b.onclick = () => { score = k; drawScores(); }; box.appendChild(b); }
}

addEventListener("keydown", (e) => {
  if ($("#s-clip").classList.contains("hide")) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input") return;
  if (e.code === "Space") { e.preventDefault(); if (tag === "select" || tag === "button") e.target.blur(); if (!e.repeat) addMark(); }
  else if (e.key === "p" || e.key === "P") { e.preventDefault(); toggle(); }
  else if (e.key === "ArrowRight") { au.currentTime = Math.min((dur || 1e9), au.currentTime + 2); draw(); }
  else if (e.key === "ArrowLeft") { au.currentTime = Math.max(0, au.currentTime - 2); draw(); }
});

$("#next").onclick = async () => {
  const err = $("#e1");
  if (marks.some((m) => !m.area)) { marks.forEach((m) => m.touched = true); drawMarks(); err.textContent = "Choose an area for every mark (or delete the mark)."; return; }
  if (!score) { err.textContent = "Give the reading an overall score from 1 to 10."; return; }
  try {
    await api("/api/human/save", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rater, clip: clips[idx].clip, marks: marks.map(({ t_s, area, pressed_s }) => ({ t_s, area, pressed_s })), score_1_10: score, note: $("#note").value }) });
    clips[idx].done = true; au.pause();
    idx++;
    if (idx >= clips.length) { $("#barfill").style.width = "100%"; done(); } else load();
  } catch (e) { err.textContent = e.message; }
};
