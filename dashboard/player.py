"""Synced A/B spectrogram player (custom Streamlit component, plain HTML/JS, no dependencies).

Two lanes: baseline (A) and altered (B). Each lane is a spectrogram computed in the browser, with the altered
regions drawn on it and one playhead that follows the audio on BOTH lanes (time is mapped through the edit
regions, because stretching/insertion shifts everything after them).

Interactions
  click inside a region        play only that change (plus optional context)
  click elsewhere              play from there
  drag                         play just the selection (shown on both lanes)
  buttons / keys               A, B, switch A<->B at the same moment, changed-only, stop, zoom to region
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit.components.v1 as components


def synced_player(base_path: Path, alt_path: Path, regions: list[dict], words: list[dict], flagged: dict[int, str],
                  height: int = 600):
    b64 = lambda p: base64.b64encode(Path(p).read_bytes()).decode()
    data = {
        "a": b64(base_path), "b": b64(alt_path),
        "regions": [{"label": f'{r["flaw"]} L{r["level"]}' + (f' · {r["params"].get("original", "")}→{r["params"].get("replaced_with", "")}'
                                                              if r["flaw"] == "WORD_SWAP" else ""),
                     "kind": r["kind"], "bs": r["baseline_start_s"], "be": r["baseline_end_s"], "cs": r["start_s"], "ce": r["end_s"]}
                    for r in regions],
        "words": [{"t": w["w"], "s": w["start_s"], "e": w["end_s"], "f": flagged.get(w["i"], "")} for w in words],
    }
    components.html(_HTML.replace("__DATA__", json.dumps(data)), height=height, scrolling=False)


_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><style>
:root{--ink:#111827;--mut:#6b7280;--line:#e5e7eb;--acc:#d9480f;--bg:#fff}
*{box-sizing:border-box}body{margin:0;font:13px -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
button{font:inherit;padding:5px 10px;border:1px solid #d1d5db;background:#f9fafb;border-radius:7px;cursor:pointer}
button:hover{background:#eef2f7}button.p{background:#1f2937;color:#fff;border-color:#1f2937}button.p:hover{background:#111827}
button.a{background:#fff4ec;border-color:#f5c3a3;color:#9c3a08}
select{font:inherit;padding:4px 6px;border:1px solid #d1d5db;border-radius:7px;background:#fff}
label{color:var(--mut);display:flex;gap:4px;align-items:center}
.lane{position:relative;margin:2px 0 6px}.lane h4{margin:0 0 3px;font-size:12px;font-weight:600;color:var(--mut)}
canvas{width:100%;display:block;border-radius:6px;cursor:crosshair;background:#0b1020}
#hint{color:var(--mut);margin-top:4px;font-size:12px}#tt{position:absolute;pointer-events:none;background:#111827;color:#fff;padding:3px 7px;border-radius:5px;font-size:12px;display:none;z-index:5;white-space:nowrap}
#st{color:var(--mut);margin-left:auto;font-variant-numeric:tabular-nums}
</style></head><body>
<div class="bar">
  <button class="p" id="pA">▶ Baseline</button><button class="p" id="pB">▶ Altered</button>
  <button class="a" id="rB">▶ Change only (altered)</button><button class="a" id="rA">▶ Same span (baseline)</button>
  <button id="sw">⇄ Switch A/B here</button><button id="stop">■ Stop</button>
  <label>Region <select id="reg"></select></label>
  <label>Context <select id="pad"><option value="0">none</option><option value="0.3" selected>±0.3 s</option><option value="1">±1 s</option></select></label>
  <label>View <select id="zoom"><option value="full">full clip</option></select></label>
  <label><input type="checkbox" id="loop"> loop</label>
  <span id="st">loading…</span>
</div>
<div class="lane"><h4>A · BASELINE</h4><canvas id="cA"></canvas></div>
<div class="lane"><h4>B · ALTERED</h4><canvas id="cB"></canvas><div id="tt"></div></div>
<div id="hint">Click a shaded region to hear only that change · click elsewhere to play from there · drag to play a selection · space = stop/resume · A / B / S keys</div>
<script>
const D = __DATA__;
const AC = new (window.AudioContext || window.webkitAudioContext)();
const R = D.regions;
const lanes = {A:{key:'A',cv:document.getElementById('cA'),view:null,buf:null,off:null},
               B:{key:'B',cv:document.getElementById('cB'),view:null,buf:null,off:null}};
let st = {src:null,lane:null,a:0,b:null,t0:0,playing:false,loop:false}, sel=null, curRegion=0, hover=null;
const $ = id => document.getElementById(id);
const rs = (L,r)=> L.key==='A' ? [r.bs,r.be] : [r.cs,r.ce];

/* ---------- time mapping through the edit regions (clip<->baseline) ---------- */
const knots = (()=>{const k=[[0,0]]; R.forEach(r=>{k.push([r.cs,r.bs],[r.ce,r.be]);}); return k;})();
function interp(x, from, to){
  const K=knots; if (x<=K[0][from]) return K[0][to]+(x-K[0][from]);
  for(let i=0;i<K.length-1;i++){const x0=K[i][from],x1=K[i+1][from];
    if(x>=x0 && x<=x1){ if(x1-x0<1e-9) return K[i][to]; return K[i][to]+(x-x0)/(x1-x0)*(K[i+1][to]-K[i][to]); }}
  const l=K[K.length-1]; return l[to]+(x-l[from]);
}
const toA = t => interp(t,0,1), toB = t => interp(t,1,0);
const mapTo = (laneKey,t)=> laneKey==='A' ? toB(t) : toA(t);   // time in the OTHER lane

/* ---------- decode + spectrogram ---------- */
function b64buf(s){const bin=atob(s),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;}
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
  for(let len=2;len<=n;len<<=1){const ang=-2*Math.PI/len,wr=Math.cos(ang),wi=Math.sin(ang);
    for(let i=0;i<n;i+=len){let cr=1,ci=0;for(let j=0;j<len/2;j++){const ur=re[i+j],ui=im[i+j],vr=re[i+j+len/2]*cr-im[i+j+len/2]*ci,vi=re[i+j+len/2]*ci+im[i+j+len/2]*cr;
      re[i+j]=ur+vr;im[i+j]=ui+vi;re[i+j+len/2]=ur-vr;im[i+j+len/2]=ui-vi;const t=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=t;}}}}
const STOPS=[[11,16,32],[44,32,110],[120,40,140],[200,70,110],[245,140,60],[252,235,160]];
function cmap(v){v=Math.max(0,Math.min(1,v))*(STOPS.length-1);const i=Math.min(STOPS.length-2,Math.floor(v)),f=v-i,a=STOPS[i],b=STOPS[i+1];
  return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}
function spectrogram(buf){
  const x=buf.getChannelData(0),sr=buf.sampleRate;let N=256;while(N<sr*0.023)N<<=1;const H=N>>2,cols=Math.max(1,Math.floor((x.length-N)/H));
  const nb=Math.floor(8000/(sr/N)),win=new Float64Array(N).map((_,i)=>0.5-0.5*Math.cos(2*Math.PI*i/(N-1)));
  const off=document.createElement('canvas');off.width=cols;off.height=nb;const c=off.getContext('2d'),img=c.createImageData(cols,nb);
  const re=new Float64Array(N),im=new Float64Array(N);
  for(let t=0;t<cols;t++){for(let i=0;i<N;i++){re[i]=x[t*H+i]*win[i];im[i]=0;}fft(re,im);
    for(let k=0;k<nb;k++){const db=20*Math.log10(Math.hypot(re[k],im[k])/(N/4)+1e-9),[r,g,b]=cmap((db+95)/65),p=((nb-1-k)*cols+t)*4;
      img.data[p]=r;img.data[p+1]=g;img.data[p+2]=b;img.data[p+3]=255;}}
  c.putImageData(img,0,0);return {off,cols,dur:x.length/sr};
}
async function load(){
  for(const [k,s] of [['A',D.a],['B',D.b]]){const L=lanes[k];L.buf=await AC.decodeAudioData(b64buf(s));
    const sp=spectrogram(L.buf);L.sp=sp;L.dur=sp.dur;L.view=[0,L.buf.duration];}
  const rg=$("reg");R.forEach((r,i)=>{const o=document.createElement("option");o.value=i;o.textContent=`${i+1}: ${r.label}`;rg.appendChild(o);});
  const z=$('zoom');R.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`region ${i+1}`;z.appendChild(o);});
  $('st').textContent='ready';resize();
}

/* ---------- drawing ---------- */
function resize(){for(const L of Object.values(lanes)){const w=L.cv.parentElement.clientWidth,dpr=window.devicePixelRatio||1;
  L.cv.width=w*dpr;L.cv.height=150*dpr;L.cv.style.height='150px';L.dpr=dpr;}draw();}
function x2t(L,x){const [v0,v1]=L.view;return v0+(x/L.cv.clientWidth)*(v1-v0);}
function t2x(L,t){const [v0,v1]=L.view;return (t-v0)/(v1-v0)*L.cv.width;}
function draw(){
  for(const L of Object.values(lanes)){if(!L.sp)continue;const c=L.cv.getContext('2d'),W=L.cv.width,Hh=L.cv.height,d=L.dpr,[v0,v1]=L.view,sp=L.sp;
    c.clearRect(0,0,W,Hh);const sx=Math.max(0,v0/sp.dur*sp.cols),sw=Math.min(sp.cols-sx,(v1-v0)/sp.dur*sp.cols);
    c.imageSmoothingEnabled=true;c.drawImage(sp.off,sx,0,sw,sp.off.height,0,0,W,Hh-16*d);
    // time axis
    c.fillStyle='#9ca3af';c.font=`${10*d}px sans-serif`;const span=v1-v0,step=span>20?5:span>8?2:span>3?1:0.5;
    for(let t=Math.ceil(v0/step)*step;t<=v1;t+=step){const x=t2x(L,t);c.fillRect(x,Hh-16*d,1,4*d);c.fillText(t.toFixed(step<1?1:0)+'s',x+3*d,Hh-4*d);}
    // regions
    R.forEach((r,i)=>{const [a,b]=rs(L,r),x0=t2x(L,a),x1=t2x(L,b),pt=(b-a)<0.05;
      c.fillStyle=i===curRegion?'rgba(255,120,40,.38)':'rgba(255,120,40,.22)';c.strokeStyle='#ff7a2f';c.lineWidth=(i===curRegion?2:1)*d;
      if(pt){c.beginPath();c.moveTo(x0,0);c.lineTo(x0,Hh-16*d);c.stroke();c.fillStyle='#ff7a2f';c.beginPath();c.moveTo(x0-5*d,0);c.lineTo(x0+5*d,0);c.lineTo(x0,8*d);c.fill();}
      else{c.fillRect(x0,0,x1-x0,Hh-16*d);c.strokeRect(x0,0,x1-x0,Hh-16*d);}
      if(x1>-50&&x0<W){c.fillStyle='#fff';c.font=`bold ${10*d}px sans-serif`;c.fillText(`${i+1}`,Math.max(x0+3*d,2),12*d);}});
    // words (baseline lane only; label when they fit)
    if(L.key==='A'){c.font=`${10*d}px sans-serif`;D.words.forEach(w=>{const x0=t2x(L,w.s),x1=t2x(L,w.e);if(x1<0||x0>W)return;
      c.fillStyle=w.f?'#ffb48a':'rgba(255,255,255,.55)';c.fillRect(x0,Hh-34*d,Math.max(1,x1-x0-1),2*d);
      if(x1-x0>c.measureText(w.t).width+4*d){c.fillStyle=w.f?'#ffd2b8':'rgba(255,255,255,.8)';c.fillText(w.t,x0+2*d,Hh-22*d);}});}
    // selection
    if(sel){const [sa,sb]=sel.lane===L.key?[sel.a,sel.b]:[mapTo(sel.lane,sel.a),mapTo(sel.lane,sel.b)];
      c.fillStyle='rgba(255,255,255,.2)';c.fillRect(t2x(L,sa),0,t2x(L,sb)-t2x(L,sa),Hh-16*d);}
    // playhead (mapped to this lane)
    if(st.playing){let t=st.a+(AC.currentTime-st.t0);const tl=st.lane===L.key?t:mapTo(st.lane,t);
      const x=t2x(L,tl);c.fillStyle='#fff';c.fillRect(x-d,0,2*d,Hh-16*d);}
  }
}
function tick(){draw();if(st.playing){const t=st.a+(AC.currentTime-st.t0);$('st').textContent=`${st.lane==='A'?'baseline':'altered'} ${t.toFixed(2)} s`;}requestAnimationFrame(tick);}

/* ---------- playback ---------- */
function stop(keep){if(st.src){st.src.onended=null;try{st.src.stop();}catch(e){}}st.src=null;st.playing=false;if(!keep)$('st').textContent='stopped';}
function play(laneKey,a,b){
  AC.resume();stop(true);const L=lanes[laneKey];a=Math.max(0,a);if(b!=null)b=Math.min(L.buf.duration,b);if(b!=null&&b-a<0.05)b=a+0.05;
  const s=AC.createBufferSource();s.buffer=L.buf;s.connect(AC.destination);
  const gen={a,b,lane:laneKey};
  if($('loop').checked&&b!=null){s.loop=true;s.loopStart=a;s.loopEnd=b;s.start(0,a);}else s.start(0,a,b==null?undefined:b-a);
  st={src:s,lane:laneKey,a,b,t0:AC.currentTime,playing:true};
  s.onended=()=>{if(st.src===s){st.playing=false;$('st').textContent='ready';}};
}
function regionSpan(laneKey,i){const r=R[i],pad=parseFloat($('pad').value);let [a,b]=rs(lanes[laneKey],r);
  if(b-a<0.05){a-=Math.max(pad,0.3);b+=Math.max(pad,0.3);}else{a-=pad;b+=pad;}return [Math.max(0,a),b];}
function playRegion(laneKey,i){if(!R.length)return;curRegion=i;$('reg').value=i;const [a,b]=regionSpan(laneKey,i);sel={lane:laneKey,a,b};play(laneKey,a,b);}
function switchLane(){if(!st.playing){return;}const t=st.a+(AC.currentTime-st.t0),o=st.lane==='A'?'B':'A';
  const nt=mapTo(st.lane,t),nb=st.b==null?null:mapTo(st.lane,st.b);play(o,nt,nb);}

/* ---------- interaction ---------- */
for(const L of Object.values(lanes)){
  let down=null;
  L.cv.addEventListener('pointerdown',e=>{down={x:e.offsetX};L.cv.setPointerCapture(e.pointerId);});
  L.cv.addEventListener('pointermove',e=>{
    if(down&&Math.abs(e.offsetX-down.x)>4){sel={lane:L.key,a:Math.min(x2t(L,down.x),x2t(L,e.offsetX)),b:Math.max(x2t(L,down.x),x2t(L,e.offsetX))};}
    const t=x2t(L,e.offsetX);let tip='';R.forEach((r,i)=>{const [a,b]=rs(L,r);if(t>=a-0.05&&t<=b+0.05)tip=`${i+1}: ${r.label} (${r.kind}) — click to hear only this`;});
    const tt=$('tt');if(tip){tt.style.display='block';tt.textContent=tip;tt.style.left=Math.min(e.clientX-8,L.cv.clientWidth-300)+'px';tt.style.top=(L.key==='A'?-4:e.offsetY+70)+'px';}else tt.style.display='none';});
  L.cv.addEventListener('pointerup',e=>{if(!down)return;const moved=Math.abs(e.offsetX-down.x)>4,t=x2t(L,e.offsetX);down=null;
    if(moved){play(L.key,sel.a,sel.b);return;}
    const hit=R.findIndex(r=>{const [a,b]=rs(L,r);return t>=a-0.05&&t<=b+0.05;});
    if(hit>=0){playRegion(L.key,hit);}else{sel=null;play(L.key,t,null);}});
}
$('pA').onclick=()=>play('A',lanes.A.view[0],null);$('pB').onclick=()=>play('B',lanes.B.view[0],null);
$('rB').onclick=()=>playRegion('B',curRegion);$('rA').onclick=()=>playRegion('A',curRegion);
$('sw').onclick=switchLane;$('stop').onclick=()=>{stop();sel=null;};
$('reg').onchange=e=>{curRegion=+e.target.value;};
$('zoom').onchange=e=>{const v=e.target.value;
  if(v==='full'){for(const L of Object.values(lanes))L.view=[0,L.buf.duration];}
  else{const r=R[+v],m=1.5;lanes.A.view=[Math.max(0,r.bs-m),Math.min(lanes.A.buf.duration,r.be+m)];lanes.B.view=[Math.max(0,r.cs-m),Math.min(lanes.B.buf.duration,r.ce+m)];curRegion=+v;$('reg').value=v;}};
addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;
  if(e.code==='Space'){e.preventDefault();st.playing?stop():play(st.lane||'B',st.a||0,st.b);}
  else if(e.key==='a'||e.key==='A')play('A',sel&&sel.lane==='A'?sel.a:lanes.A.view[0],sel&&sel.lane==='A'?sel.b:null);
  else if(e.key==='b'||e.key==='B')play('B',sel&&sel.lane==='B'?sel.a:lanes.B.view[0],sel&&sel.lane==='B'?sel.b:null);
  else if(e.key==='s'||e.key==='S')switchLane();});
addEventListener('resize',resize);
load().then(()=>requestAnimationFrame(tick)).catch(e=>{$('st').textContent='audio decode failed: '+e;});
</script></body></html>
"""
