// "Try that part again": record a retake of the flagged phrase in the browser, send it as a 16 kHz WAV, and compare the one feature behind the flag.
import { $, api, esc, toast } from "./main.js";

export const RETAKE_FLAWS = new Set(["PACE_FAST", "PACE_SLOW", "PAUSE_BAD", "PAUSE_LOST", "FADE", "SHOUT", "SLUR", "MONOTONE", "FILLER", "RARE_HESIT", "REPEAT", "WORD_SKIP", "WORD_SWAP"]);
const history = {};                       // "<result key>:<region>" -> attempts, kept for the whole session
let rec = null, playing = null;

const fmt = (v, unit) => (v == null ? "—" : unit === "count" ? String(Math.round(v)) : unit === "share" ? Math.round(v * 100) + "%" : v.toFixed(unit === "seconds" ? 2 : 1));
const rng = (p) => {
  const u = p.unit === "share" ? (x) => Math.round(x * 100) + "%" : (x) => (p.unit === "seconds" ? x.toFixed(2) : x.toFixed(1));
  if (p.unit === "count") return "none";
  if (p.lo != null && p.hi != null) return `${u(p.lo)} to ${u(p.hi)}`;
  return p.lo != null ? `${u(p.lo)} or more` : `${u(p.hi)} or less`;
};

async function toWav16k(blob) {
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ac.decodeAudioData(await blob.arrayBuffer()); ac.close();
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(buf.duration * 16000)), 16000);
  const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
  const x = (await off.startRendering()).getChannelData(0);
  const out = new DataView(new ArrayBuffer(44 + x.length * 2)), w = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); out.setUint32(4, 36 + x.length * 2, true); w(8, "WAVEfmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, 16000, true); out.setUint32(28, 32000, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true); w(36, "data"); out.setUint32(40, x.length * 2, true);
  x.forEach((v, i) => out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return new Blob([out], { type: "audio/wav" });
}

function playBlob(url) { if (playing) playing.pause(); playing = new Audio(url); playing.play(); }

export function mountRetake(host, R, r, playOriginal) {
  const id = `${R.key}:${r.id}`, H = (history[id] ||= []);
  host.hidden = false;
  host.innerHTML = `<div class="rt-head"><b>Try that part again</b><button class="btn sm" id="rt-close" aria-label="Close">✕</button></div><div class="dim" id="rt-load">Getting ready…</div>`;
  $("#rt-close").onclick = () => { host.hidden = true; host.innerHTML = ""; };
  api(`/api/retake/prepare?key=${R.key}&region=${r.id}`).then((p) => draw(host, R, r, p, H, playOriginal)).catch((e) => { $("#rt-load").textContent = e.message; });
}

function draw(host, R, r, p, H, playOriginal) {
  const rows = [{ label: "First attempt", value: p.first, inside: p.first != null && inRange(p, p.first), pts: p.points_before, orig: true }, ...H.map((h, i) => ({ label: `Retake ${i + 1}`, value: h.value, inside: h.inside, pts: h.points_after, url: h.url }))];
  host.innerHTML = `<div class="rt-head"><b>Try that part again</b><button class="btn sm" id="rt-close" aria-label="Close">✕</button></div>
    <p class="rt-say">“${esc(p.text)}”</p>
    <p class="dim rt-hint">${esc(p.hint)} Measured: <b>${esc(p.name.toLowerCase())}</b> (${esc(p.unit)}). Expected: <b>${rng(p)}</b>.</p>
    <div class="rt-act"><button class="btn gold" id="rt-rec">● Record</button><span class="dim" id="rt-st"></span></div>
    <table class="rt-t"><tr><th></th><th>${esc(p.name)}</th><th>In range</th><th>Points lost</th><th></th></tr>
    ${rows.map((x, i) => `<tr class="${x.inside ? "ok" : ""}"><td>${x.label}</td><td class="mono">${fmt(x.value, p.unit)}</td><td class="tick">${x.value == null ? "—" : x.inside ? "✓" : "✗"}</td><td class="mono">${x.orig ? "−" + x.pts.toFixed(1) : x.pts === 0 ? "0" : "−" + x.pts.toFixed(1)}</td><td><button class="btn sm" data-p="${i}">▶</button></td></tr>`).join("")}</table>`;
  $("#rt-close").onclick = () => { host.hidden = true; host.innerHTML = ""; if (playing) playing.pause(); if (rec) rec.stop(); };
  host.querySelectorAll("[data-p]").forEach((b) => b.onclick = () => { const x = rows[+b.dataset.p]; x.orig ? playOriginal(p.span[0], p.span[1]) : playBlob(x.url); });
  const btn = $("#rt-rec"), st = $("#rt-st");
  btn.onclick = async () => {
    if (rec) { rec.stop(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } }); }
    catch { toast("Microphone not available. Allow microphone access for this page (it only works on localhost or https)."); return; }
    const chunks = []; rec = new MediaRecorder(stream); const t0 = Date.now();
    const tick = setInterval(() => { st.textContent = `Recording… ${((Date.now() - t0) / 1000).toFixed(0)} s`; }, 250);
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      clearInterval(tick); stream.getTracks().forEach((t) => t.stop()); rec = null; btn.textContent = "● Record"; st.textContent = "Scoring…";
      try {
        const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" }), wav = await toWav16k(blob), fd = new FormData(); fd.append("file", wav, "retake.wav");
        const res = await api(`/api/retake?key=${R.key}&region=${r.id}`, { method: "POST", body: fd });
        H.push({ ...res, url: URL.createObjectURL(wav) });
        draw(host, R, r, p, H, playOriginal);
      } catch (e) { st.textContent = ""; toast(e.message); }
    };
    rec.start(); btn.textContent = "■ Stop";
  };
}

function inRange(p, v) { return (p.lo == null || v >= p.lo) && (p.hi == null || v <= p.hi); }
