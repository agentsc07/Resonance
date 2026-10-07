// The room: a black space with a floor grid receding to the horizon and a hologram waveform that drifts through it, with a slow gold glint
// passing along the trace and a few motes of light. Pure decoration behind the page; paused when hidden or reduced-motion is set.
const cv = document.getElementById("room"), c = cv.getContext("2d");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
let W = 0, H = 0, dpr = 1, t0 = performance.now();
const motes = Array.from({ length: 46 }, () => ({ x: Math.random(), y: Math.random(), z: 0.3 + Math.random() * 0.7, p: Math.random() * 6.28 }));

function size() { dpr = Math.min(devicePixelRatio || 1, 2); W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px"; }
addEventListener("resize", size); size();

function env(x, t) {                                         // a speech-like envelope: phrases of loud and quiet
  return 0.35 + 0.65 * Math.abs(Math.sin(x * 0.0042 + t * 0.00022) * Math.sin(x * 0.0113 - t * 0.00031) + 0.35 * Math.sin(x * 0.027 + t * 0.0004));
}
function frame(now) {
  const t = now - t0;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  const hz = H * 0.62;                                       // horizon

  // floor grid in perspective, fading with distance
  c.lineWidth = 1;
  for (let i = 0; i < 16; i++) {
    const k = ((i + (t * 0.00018) % 1) / 16), y = hz + (H - hz) * k * k, a = 0.02 + 0.09 * k;
    c.strokeStyle = `rgba(240,192,90,${a})`; c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke();
  }
  for (let i = -14; i <= 14; i++) {
    const a = 0.05 * (1 - Math.abs(i) / 16);
    c.strokeStyle = `rgba(240,192,90,${a})`; c.beginPath(); c.moveTo(W / 2 + i * 12, hz); c.lineTo(W / 2 + i * W * 0.11, H); c.stroke();
  }
  const fl = c.createLinearGradient(0, hz - 40, 0, hz + 30); fl.addColorStop(0, "rgba(240,192,90,0)"); fl.addColorStop(.5, "rgba(240,192,90,.10)"); fl.addColorStop(1, "rgba(240,192,90,0)");
  c.fillStyle = fl; c.fillRect(0, hz - 40, W, 70);

  // hologram waveform: mirrored bars + a glowing spine, scrolling slowly to the left
  const bars = Math.ceil(W / 7), mid = H * 0.17, sweep = ((t / 9000) % 1.3 - 0.15) * W;
  for (let i = 0; i < bars; i++) {
    const x = i * 7, e = env(x + t * 0.03, t), n = 0.55 + 0.45 * Math.sin(i * 1.7 + t * 0.002 * (1 + (i % 5) * 0.15)), h = e * n * H * 0.10;
    const near = Math.max(0, 1 - Math.abs(x - sweep) / 190);
    const a = 0.55 * (0.08 + 0.18 * e) + 0.4 * near * near;
    c.fillStyle = near > 0.02 ? `rgba(255,${222 + 20 * near | 0},${150 + 90 * near | 0},${a})` : `rgba(240,192,90,${a})`;
    c.fillRect(x, mid - h, 3.4, h * 2);
  }
  c.save(); c.shadowColor = "rgba(240,192,90,.55)"; c.shadowBlur = 22; c.strokeStyle = "rgba(255,226,160,.35)"; c.lineWidth = 1.4; c.beginPath();
  for (let x = 0; x <= W; x += 6) { const y = mid + Math.sin(x * 0.012 + t * 0.0011) * env(x + t * 0.03, t) * H * 0.07; x ? c.lineTo(x, y) : c.moveTo(x, y); }
  c.stroke(); c.restore();
  // reflection in the floor
  c.save(); c.globalAlpha = 0.10; c.translate(0, hz * 2 - mid * 0.5); c.scale(1, -0.42);
  for (let i = 0; i < bars; i += 2) { const x = i * 7, e = env(x + t * 0.03, t), h = e * (0.55 + 0.45 * Math.sin(i * 1.7 + t * 0.002)) * H * 0.13; c.fillStyle = "rgba(240,192,90,1)"; c.fillRect(x, mid - h, 3.4, h * 2); }
  c.restore();

  // motes
  for (const m of motes) {
    const y = (m.y - t * 0.00001 * m.z + 1) % 1, x = (m.x + Math.sin(t * 0.0003 + m.p) * 0.01 + 1) % 1, a = (0.15 + 0.35 * Math.abs(Math.sin(t * 0.0008 + m.p))) * m.z;
    c.fillStyle = `rgba(255,236,190,${a})`; c.beginPath(); c.arc(x * W, y * H * 0.8, 0.7 + 1.3 * m.z, 0, 6.28); c.fill();
  }
  if (!reduce && !document.hidden) requestAnimationFrame(frame);
  else if (!reduce) document.addEventListener("visibilitychange", () => requestAnimationFrame(frame), { once: true });
}
requestAnimationFrame(frame);
