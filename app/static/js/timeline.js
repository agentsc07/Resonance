// Timeline canvas: waveform (hero), a second lane (pitch / loudness / rate) with its expected-range band, one strip per flagged moment, time axis.
// Click a strip or a shaded region to select it; click anywhere else to play from there.
export const CAT_COLOR = {
  "Pacing": "#5aa9ff", "Pausing": "#a78bfa", "Intonation": "#f59e0b", "Volume": "#4ade80",
  "Fluency": "#fb7185", "Clarity": "#2dd4bf", "Text fidelity": "#cbd5e1",
};

const LANES = {
  pitch: { label: "PITCH · semitones re your median", lo: -12, hi: 12, key: "f0", band: "f0_band", ref: "ref_f0" },
  level: { label: "LOUDNESS · dB re your speech level", lo: -40, hi: 10, key: "level", band: "level_band", ref: "ref_level" },
  rate: { label: "SPEAKING RATE · syllables per second", lo: 0, hi: 12, key: "rate", band: "rate_band", ref: null },
};

export class Timeline {
  constructor(canvas, tip, cb) {
    this.cv = canvas; this.tip = tip; this.cb = cb;
    this.data = null; this.regions = []; this.sel = null; this.hover = null; this.lane = "pitch"; this.truth = null; this.playhead = null;
    this.layout = {};
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener("mousemove", (e) => this.onMove(e));
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.style.opacity = 0; this.draw(); });
    canvas.addEventListener("click", (e) => this.onClick(e));
  }

  set(data, regions) { this.data = data; this.regions = regions; this.sel = null; this.hover = null; this.rows(); this.draw(); }
  setLane(k) { this.lane = k; this.draw(); }
  setTruth(t) { this.truth = t; this.draw(); }
  select(id) { this.sel = id; this.draw(); }
  setPlayhead(t) { this.playhead = t; this.draw(); }

  resize() {
    const dpr = window.devicePixelRatio || 1, w = this.cv.clientWidth, h = this.cv.clientHeight;
    if (!w) return;
    this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
    this.W = w; this.H = h; this.dpr = dpr; this.draw();
  }

  rows() {                                   // greedy row assignment so overlapping strips stack
    const ends = [];
    this.strip = this.regions.map((r) => {
      let k = ends.findIndex((e) => e <= r.start - 0.05);
      if (k < 0) { k = ends.length; ends.push(0); }
      ends[k] = r.end;
      return k;
    });
    this.nrows = Math.max(1, ends.length);
    const rowH = Math.min(22, Math.floor(76 / this.nrows));
    this.cv.style.height = 204 + this.nrows * rowH + 30 + "px";               // no empty band under the strips
  }

  geo() {
    const pad = 10, W = this.W || this.cv.clientWidth, H = this.H || this.cv.clientHeight;
    const rowH = Math.min(22, Math.floor(76 / this.nrows));
    return { pad, W, H, wave: [6, 112], lane: [122, 196], stripTop: 204, rowH, axisY: H - 14 };
  }

  x(t) { const g = this.geo(); return g.pad + (t / this.data.dur) * (g.W - 2 * g.pad); }
  t(x) { const g = this.geo(); return Math.max(0, Math.min(this.data.dur, ((x - g.pad) / (g.W - 2 * g.pad)) * this.data.dur)); }

  hit(px, py) {
    if (!this.data) return null;
    const g = this.geo(); const tt = this.t(px);
    for (let i = this.regions.length - 1; i >= 0; i--) {
      const r = this.regions[i], x0 = this.x(r.start), x1 = Math.max(this.x(r.end), x0 + 7);
      const sy = g.stripTop + this.strip[i] * g.rowH;
      const onStrip = py >= sy && py <= sy + g.rowH - 3 && px >= x0 - 2 && px <= x1 + 2;
      const onBand = py < g.stripTop - 4 && px >= x0 - 2 && px <= x1 + 2;
      if (onStrip || onBand) return i;
    }
    return null;
  }

  onMove(e) {
    const r = this.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    const h = this.hit(px, py);
    if (h !== this.hover) { this.hover = h; this.draw(); }
    if (h !== null) {
      const R = this.regions[h];
      this.tip.innerHTML = `<b>${R.name}</b><br><span style="color:${CAT_COLOR[R.category]}">${R.category}</span> · −${R.points.toFixed(1)} pts · ${R.start.toFixed(1)} s<br><span class="dimmer">Click to select and play</span>`;
      this.tip.style.left = Math.min(px + 14, this.W - 290) + "px"; this.tip.style.top = Math.max(0, py - 54) + "px"; this.tip.style.opacity = 1;
    } else this.tip.style.opacity = 0;
  }

  onClick(e) {
    const r = this.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    const h = this.hit(px, py);
    if (h !== null) this.cb.select(this.regions[h].id, true);
    else this.cb.seek(this.t(px));
  }

  draw() {
    const d = this.data, c = this.cv.getContext("2d");
    if (!c) return;
    if (!this.W) this.resize();
    const dpr = this.dpr || 1;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, this.W, this.H);
    if (!d) return;
    const g = this.geo(), W = g.W;
    const px = this.playhead !== null ? this.x(this.playhead) : -1;

    // ---- region bands over the two lanes
    this.regions.forEach((r, i) => {
      const x0 = this.x(r.start), x1 = Math.max(this.x(r.end), x0 + 4), col = CAT_COLOR[r.category];
      const on = this.sel === r.id, hv = this.hover === i;
      c.fillStyle = hexA(col, on ? 0.30 : hv ? 0.24 : 0.14);
      c.fillRect(x0, g.wave[0], x1 - x0, g.lane[1] - g.wave[0]);
      c.fillStyle = hexA(col, on ? 0.95 : 0.55);
      c.fillRect(x0, g.wave[0], x1 - x0, 2);
    });

    // ---- waveform
    const mid = (g.wave[0] + g.wave[1]) / 2, amp = (g.wave[1] - g.wave[0]) / 2 - 3, n = d.wave.length;
    const gr = c.createLinearGradient(0, g.wave[0], 0, g.wave[1]);
    gr.addColorStop(0, "#f8d98a"); gr.addColorStop(0.5, "#e7ae4a"); gr.addColorStop(1, "#d79a2b");
    const bw = (W - 2 * g.pad) / n;
    for (let pass = 0; pass < 2; pass++) {
      c.save();
      c.beginPath();
      if (px >= 0) { if (pass === 0) c.rect(0, 0, px, this.H); else c.rect(px, 0, W - px, this.H); } else c.rect(0, 0, W, this.H);
      c.clip();
      c.globalAlpha = pass === 0 || px < 0 ? 0.95 : 0.5;
      c.fillStyle = gr;
      for (let i = 0; i < n; i++) {
        const lo = d.wave[i][0], hi = d.wave[i][1], x = g.pad + i * bw;
        const y0 = mid - hi * amp, y1 = mid - lo * amp;
        c.fillRect(x, y0, Math.max(bw - 0.4, 1), Math.max(y1 - y0, 1));
      }
      c.restore();
      if (px < 0) break;
    }
    c.globalAlpha = 1;

    // ---- second lane
    const L = LANES[this.lane];
    const [ly0, ly1] = g.lane;
    c.fillStyle = "rgba(255,255,255,.035)"; roundRect(c, g.pad, ly0, W - 2 * g.pad, ly1 - ly0, 8); c.fill();
    c.fillStyle = "#7d7a73"; c.font = "600 10px 'JetBrains Mono', monospace"; c.textBaseline = "top"; c.fillText(L.label, g.pad + 8, ly0 + 6);
    const yv = (v) => ly1 - 8 - ((Math.max(L.lo, Math.min(L.hi, v)) - L.lo) / (L.hi - L.lo)) * (ly1 - ly0 - 22);
    const band = d[L.band];
    if (band) {
      const yb0 = yv(band[1]), yb1 = yv(band[0]);
      c.fillStyle = "rgba(240,192,90,.13)"; c.fillRect(g.pad, yb0, W - 2 * g.pad, yb1 - yb0);
      c.strokeStyle = "rgba(240,192,90,.5)"; c.setLineDash([4, 4]); c.lineWidth = 1;
      c.beginPath(); c.moveTo(g.pad, yb0); c.lineTo(W - g.pad, yb0); c.moveTo(g.pad, yb1); c.lineTo(W - g.pad, yb1); c.stroke(); c.setLineDash([]);
    }
    const track = (arr, color, dash, width) => {
      c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash); c.lineJoin = "round"; c.beginPath();
      let pen = false;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v === null) { pen = false; continue; }
        const x = this.x(i * d.step), y = yv(v);
        if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y);
      }
      c.stroke(); c.setLineDash([]);
    };
    if (this.lane === "rate") {
      c.strokeStyle = "#9bd5ff"; c.lineWidth = 1.6; c.beginPath();
      d.rate.forEach(([a, b, v], i) => { const y = yv(Math.min(v, 12)); if (i === 0) c.moveTo(this.x(a), y); else c.lineTo(this.x(a), y); c.lineTo(this.x(b), y); });
      c.stroke();
    } else {
      if (L.ref && d[L.ref]) track(d[L.ref], "#ffb15e", [3, 4], 1.2);
      track(d[L.key], "#9bd5ff", [], 1.5);
    }

    // ---- flaw strips
    c.textBaseline = "middle";
    this.regions.forEach((r, i) => {
      const x0 = this.x(r.start), x1 = Math.max(this.x(r.end), x0 + 7), y = g.stripTop + this.strip[i] * g.rowH, h = g.rowH - 4;
      const col = CAT_COLOR[r.category], on = this.sel === r.id, hv = this.hover === i;
      const sg = c.createLinearGradient(x0, 0, x1, 0);
      sg.addColorStop(0, hexA(col, on ? 1 : hv ? 0.95 : 0.78)); sg.addColorStop(1, hexA(col, on ? 0.85 : 0.5));
      if (on) { c.shadowColor = col; c.shadowBlur = 14; }
      c.fillStyle = sg; roundRect(c, x0, y, x1 - x0, h, 6); c.fill(); c.shadowBlur = 0;
      if (on) { c.strokeStyle = "#fff"; c.lineWidth = 1.5; roundRect(c, x0, y, x1 - x0, h, 6); c.stroke(); }
      if (x1 - x0 > 60 && h >= 14) { c.fillStyle = "#0b0b10"; c.font = "600 11px 'Hanken Grotesk', sans-serif"; c.save(); c.beginPath(); c.rect(x0 + 2, y, x1 - x0 - 4, h); c.clip(); c.fillText(r.name, x0 + 8, y + h / 2 + 0.5); c.restore(); }
    });

    // ---- what was injected (dataset clips, on request)
    if (this.truth) {
      c.setLineDash([5, 4]); c.strokeStyle = "#f0c05a"; c.lineWidth = 1.5; c.fillStyle = "#f0c05a"; c.font = "600 10.5px 'JetBrains Mono', monospace"; c.textBaseline = "top";
      this.truth.forEach((t, i) => {
        const x0 = this.x(t.start), x1 = Math.max(this.x(t.end), x0 + 6);
        c.strokeRect(x0, g.wave[0] + 1, x1 - x0, g.wave[1] - g.wave[0] - 2);
        c.fillText(`${t.name}${t.level ? " L" + t.level : ""}`, Math.min(x0 + 3, W - 140), g.wave[0] + 6 + (i % 2) * 12);
      });
      c.setLineDash([]);
    }

    // ---- axis
    const step = d.dur > 90 ? 15 : d.dur > 40 ? 5 : 2;
    c.fillStyle = "#7d7a73"; c.font = "500 10px 'JetBrains Mono', monospace"; c.textBaseline = "top";
    for (let t = 0; t <= d.dur; t += step) { const x = this.x(t); c.fillRect(x, g.axisY - 5, 1, 4); c.fillText(t + "s", x + 3, g.axisY - 2); }

    // ---- playhead
    if (px >= 0) {
      c.fillStyle = "#fff"; c.fillRect(px - 0.75, g.wave[0], 1.5, g.axisY - g.wave[0] - 6);
      c.beginPath(); c.moveTo(px - 5, g.wave[0] - 2); c.lineTo(px + 5, g.wave[0] - 2); c.lineTo(px, g.wave[0] + 6); c.fill();
    }
  }
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function roundRect(c, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}
