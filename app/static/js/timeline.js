// Timeline canvas. Colour roles: neutral = your data, blue = the pitch line, coral = every flagged moment (points lost), gold = the selected one.
// Flagged moments are numbered (in time order); click a number or a shaded region to select and play it, click anywhere else to play from there.
const CORAL = "255,107,94", GOLD = "240,192,90";

const LANES = {
  pitch: { label: "PITCH · how high or low your voice goes (semitones from your usual)", lo: -12, hi: 12, key: "f0", band: "f0_band", ref: "ref_f0" },
  level: { label: "LOUDNESS · how loud you are (dB from your usual level)", lo: -40, hi: 10, key: "level", band: "level_band", ref: "ref_level" },
  rate: { label: "SPEED · syllables per second", lo: 0, hi: 12, key: "rate", band: "rate_band", ref: null },
};

export class Timeline {
  constructor(canvas, tip, cb) {
    this.cv = canvas; this.tip = tip; this.cb = cb;
    this.data = null; this.regions = []; this.sel = null; this.hover = null; this.lane = "pitch"; this.truth = null; this.playhead = null; this.num = {};
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener("mousemove", (e) => this.onMove(e));
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.tip.style.opacity = 0; this.draw(); });
    canvas.addEventListener("click", (e) => this.onClick(e));
    this.phase = 0; this.last = performance.now();
    const loop = (t) => {                                    // a slow glint travels along the trace; paused when the tab is hidden or reduced motion is set
      if (this.data && !document.hidden && !matchMedia("(prefers-reduced-motion: reduce)").matches) { this.phase += (t - this.last) / 7000; this.draw(); }
      this.last = t; requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  set(data, regions) { this.data = data; this.regions = regions; this.sel = null; this.hover = null; this.rows(); this.draw(); }
  setLane(k) { this.lane = k; this.draw(); }
  setTruth(t) { this.truth = t; this.draw(); }
  select(id) { this.sel = id; this.draw(); }
  setPlayhead(t) { this.playhead = t; this.draw(); }
  numberOf(id) { return this.num[id]; }

  resize() {
    const dpr = window.devicePixelRatio || 1, w = this.cv.clientWidth, h = this.cv.clientHeight;
    if (!w) return;
    this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
    this.W = w; this.H = h; this.dpr = dpr; this.draw();
  }

  rows() {                                   // number the moments in time order, then stack overlapping badges in rows
    const order = this.regions.map((r, i) => [r.start, i]).sort((a, b) => a[0] - b[0]);
    this.num = {}; order.forEach(([, i], k) => { this.num[this.regions[i].id] = k + 1; });
    const ends = []; this.strip = new Array(this.regions.length);
    order.forEach(([, i]) => {
      const r = this.regions[i], need = r.start - 0.03 * (this.data.dur / 10);
      let k = ends.findIndex((e) => e <= need);
      if (k < 0) { k = ends.length; ends.push(0); }
      ends[k] = Math.max(r.end, r.start + this.data.dur * 0.02);
      this.strip[i] = k;
    });
    this.nrows = Math.max(1, ends.length);
    this.cv.style.height = 204 + this.nrows * this.rowH() + 30 + "px";
  }
  rowH() { return Math.min(24, Math.floor(84 / this.nrows)); }

  geo() {
    const pad = 12, W = this.W || this.cv.clientWidth, H = this.H || this.cv.clientHeight;
    return { pad, W, H, wave: [10, 196], lane: [10, 196], stripTop: 204, rowH: this.rowH(), axisY: H - 14 };
  }

  x(t) { const g = this.geo(); return g.pad + (t / this.data.dur) * (g.W - 2 * g.pad); }
  t(x) { const g = this.geo(); return Math.max(0, Math.min(this.data.dur, ((x - g.pad) / (g.W - 2 * g.pad)) * this.data.dur)); }

  hit(px, py) {
    if (!this.data) return null;
    const g = this.geo();
    for (let i = this.regions.length - 1; i >= 0; i--) {
      const r = this.regions[i], x0 = this.x(r.start), x1 = Math.max(this.x(r.end), x0 + 10);
      const sy = g.stripTop + this.strip[i] * g.rowH;
      const onBadge = py >= sy - 1 && py <= sy + g.rowH && px >= x0 - 12 && px <= x0 + 12;
      const onStrip = py >= sy && py <= sy + g.rowH - 3 && px >= x0 && px <= x1 + 2;
      const onBand = py < g.stripTop - 4 && px >= x0 - 2 && px <= x1 + 2;
      if (onBadge || onStrip || onBand) return i;
    }
    return null;
  }

  onMove(e) {
    const r = this.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    const h = this.hit(px, py);
    if (h !== this.hover) { this.hover = h; this.draw(); }
    if (h !== null) {
      const R = this.regions[h];
      this.tip.innerHTML = `<b>${this.num[R.id]}. ${R.name}</b><br><span style="color:var(--coral)">−${R.points.toFixed(1)} points</span> · at ${R.start.toFixed(1)} s<br><span class="dimmer">Click to select and play</span>`;
      this.tip.style.left = Math.min(px + 14, this.W - 300) + "px"; this.tip.style.top = Math.max(0, py - 58) + "px"; this.tip.style.opacity = 1;
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

    // ---- flagged regions: one coral tint, gold when selected
    this.regions.forEach((r, i) => {
      const x0 = this.x(r.start), x1 = Math.max(this.x(r.end), x0 + 4), on = this.sel === r.id, hv = this.hover === i;
      c.fillStyle = on ? `rgba(${GOLD},.20)` : `rgba(${CORAL},${hv ? 0.22 : 0.13})`;
      c.fillRect(x0, g.wave[0], x1 - x0, g.lane[1] - g.wave[0]); if (on) { c.shadowColor = `rgba(${GOLD},.8)`; c.shadowBlur = 18; c.fillRect(x0, g.wave[0], x1 - x0, 2); c.shadowBlur = 0; }
      c.fillStyle = on ? `rgba(${GOLD},.95)` : `rgba(${CORAL},.6)`;
      c.fillRect(x0, g.wave[0], x1 - x0, 2);
    });

    // ---- waveform: a faint backdrop, never the subject (gold-tinted, brighter where it has already played)
    const mid = (g.wave[0] + g.wave[1]) / 2, amp = (g.wave[1] - g.wave[0]) / 2 - 4, n = d.wave.length, bw = (W - 2 * g.pad) / n;
    for (let i = 0; i < n; i++) {
      const lo = d.wave[i][0], hi = d.wave[i][1], x = g.pad + i * bw, y0 = mid - hi * amp * 0.8, y1 = mid - lo * amp * 0.8;
      c.fillStyle = px >= 0 && x < px ? "rgba(240,192,90,.34)" : "rgba(240,192,90,.17)";
      c.fillRect(x, y0, Math.max(bw - 0.5, 1), Math.max(y1 - y0, 1));
    }

    // ---- one lane at a time: expected range (soft band) + your line, drawn as a glowing trace
    const L = LANES[this.lane], [ly0, ly1] = g.lane;
    c.fillStyle = "#9a978d"; c.font = "500 11px 'Hanken Grotesk', sans-serif"; c.textBaseline = "top"; c.fillText(L.label, g.pad + 8, ly0 + 4);
    const yv = (v) => ly1 - 10 - ((Math.max(L.lo, Math.min(L.hi, v)) - L.lo) / (L.hi - L.lo)) * (ly1 - ly0 - 40);
    const band = d[L.band];
    if (band) {
      const yb0 = yv(band[1]), yb1 = yv(band[0]), bg = c.createLinearGradient(0, yb0, 0, yb1);
      bg.addColorStop(0, "rgba(255,255,255,.02)"); bg.addColorStop(.5, "rgba(255,255,255,.09)"); bg.addColorStop(1, "rgba(255,255,255,.02)");
      c.fillStyle = bg; c.fillRect(g.pad, yb0, W - 2 * g.pad, yb1 - yb0);
    }
    const sweepX = g.pad + (((this.phase || 0) % 1.25) - 0.1) * (W - 2 * g.pad);
    const trace = (pts) => {                                    // pts: [[x,y]|null,...]
      const path = () => { c.beginPath(); let pen = false; for (const p of pts) { if (!p) { pen = false; continue; } if (!pen) { c.moveTo(p[0], p[1]); pen = true; } else c.lineTo(p[0], p[1]); } };
      c.lineJoin = "round"; c.lineCap = "round";
      c.shadowColor = "rgba(124,183,255,.9)"; c.shadowBlur = 16; c.strokeStyle = "rgba(124,183,255,.35)"; c.lineWidth = 5; path(); c.stroke();
      c.shadowBlur = 6; c.strokeStyle = "#8cc2ff"; c.lineWidth = 1.7; path(); c.stroke(); c.shadowBlur = 0;
      const sg = c.createLinearGradient(sweepX - 110, 0, sweepX + 110, 0);                         // the passing glint
      sg.addColorStop(0, "rgba(255,244,214,0)"); sg.addColorStop(.5, "rgba(255,244,214,.95)"); sg.addColorStop(1, "rgba(255,244,214,0)");
      c.strokeStyle = sg; c.lineWidth = 2.4; path(); c.stroke();
    };
    if (this.lane === "rate") {
      const pts = []; d.rate.forEach(([a, b, v]) => { const y = yv(Math.min(v, 12)); pts.push([this.x(a), y], [this.x(b), y]); }); trace(pts);
    } else {
      if (L.ref && d[L.ref]) { c.strokeStyle = "rgba(217,214,204,.55)"; c.setLineDash([3, 5]); c.lineWidth = 1.2; c.beginPath(); let pen = false; d[L.ref].forEach((v, i) => { if (v === null) { pen = false; return; } const X = this.x(i * d.step), Y = yv(v); if (!pen) { c.moveTo(X, Y); pen = true; } else c.lineTo(X, Y); }); c.stroke(); c.setLineDash([]); }
      trace(d[L.key].map((v, i) => (v === null ? null : [this.x(i * d.step), yv(v)])));
    }

    // ---- numbered flagged moments
    c.textBaseline = "middle";
    this.regions.forEach((r, i) => {
      const x0 = this.x(r.start), x1 = Math.max(this.x(r.end), x0 + 10), y = g.stripTop + this.strip[i] * g.rowH, h = g.rowH - 4;
      const on = this.sel === r.id, hv = this.hover === i, cy = y + h / 2;
      c.fillStyle = on ? `rgba(${GOLD},.85)` : `rgba(${CORAL},${hv ? 0.9 : 0.55})`;
      if (on) { c.shadowColor = `rgba(${GOLD},.9)`; c.shadowBlur = 12; }
      roundRect(c, x0, y + h / 2 - 3, Math.max(x1 - x0, 6), 6, 3); c.fill(); c.shadowBlur = 0;
      const rad = Math.min(10, h / 2 + 1);
      c.fillStyle = on ? `rgb(${GOLD})` : `rgb(${CORAL})`; c.beginPath(); c.arc(x0 + 1, cy, rad, 0, Math.PI * 2); c.fill();
      c.fillStyle = "#1a0a08"; c.font = `700 ${rad > 8 ? 11 : 10}px 'JetBrains Mono', monospace`; c.textAlign = "center"; c.fillText(String(this.num[r.id]), x0 + 1, cy + 0.5); c.textAlign = "left";
    });

    // ---- what was injected (dataset clips, on request)
    if (this.truth) {
      c.setLineDash([5, 4]); c.strokeStyle = `rgb(${GOLD})`; c.lineWidth = 1.5; c.fillStyle = `rgb(${GOLD})`; c.font = "600 10.5px 'JetBrains Mono', monospace"; c.textBaseline = "top";
      this.truth.forEach((t, i) => {
        const x0 = this.x(t.start), x1 = Math.max(this.x(t.end), x0 + 6);
        c.strokeRect(x0, g.wave[0] + 1, x1 - x0, g.wave[1] - g.wave[0] - 2);
        c.fillText(`${t.name}${t.level ? " L" + t.level : ""}`, Math.min(x0 + 3, W - 140), g.wave[0] + 6 + (i % 2) * 12);
      });
      c.setLineDash([]);
    }

    // ---- axis
    const step = d.dur > 90 ? 15 : d.dur > 40 ? 5 : 2;
    c.fillStyle = "#7b7870"; c.font = "500 10px 'JetBrains Mono', monospace"; c.textBaseline = "top";
    for (let t = 0; t <= d.dur; t += step) { const x = this.x(t); c.fillRect(x, g.axisY - 5, 1, 4); c.fillText(t + "s", x + 3, g.axisY - 2); }

    // ---- playhead
    if (px >= 0) {
      c.fillStyle = "#fff"; c.fillRect(px - 0.75, g.wave[0], 1.5, g.axisY - g.wave[0] - 6);
      c.beginPath(); c.moveTo(px - 5, g.wave[0] - 2); c.lineTo(px + 5, g.wave[0] - 2); c.lineTo(px, g.wave[0] + 6); c.fill();
    }
  }
}

function roundRect(c, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}
