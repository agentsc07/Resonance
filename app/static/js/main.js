import "./room.js";
import { initAnalyse } from "./analyse.js";
import { initDataset } from "./dataset.js";
import { initAbout } from "./about.js";

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return r.json();
}

let toastT;
export function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("on");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("on"), 6000);
}

let BRAND = { product: "Resonance", tagline: "Detect. Explain. Improve." };   // replaced by /api/meta (app/brand.json: the one place the product name lives)
const PAGES = ["analyse", "dataset", "about"];
const loaded = {};
function show(name) {
  if (!PAGES.includes(name)) name = "analyse";
  $$(".page").forEach((p) => p.classList.toggle("on", p.id === "p-" + name));
  $$("#nav a[data-page]").forEach((a) => a.classList.toggle("on", a.dataset.page === name));
  document.body.dataset.page = name;
  const a = $(`#nav a[data-page="${name}"]`), pill = $("#navpill");
  pill.style.left = a.offsetLeft + "px"; pill.style.width = a.offsetWidth + "px";
  if (name === "dataset" && !loaded.dataset) { loaded.dataset = true; initDataset(); }
  if (name === "about" && !loaded.about) { loaded.about = true; initAbout(); }
  document.title = (BRAND.product || "Resonance") + " · " + name[0].toUpperCase() + name.slice(1);
}

const meta = await api("/api/meta");
BRAND = meta.brand || BRAND;
$$("[data-brand]").forEach((e) => { e.textContent = BRAND.product; });
$$("[data-tagline]").forEach((e) => { e.textContent = BRAND.tagline; });
if (meta.lab) { const l = $("#labnav"); l.hidden = false; l.href = meta.lab_url; }
initAnalyse(meta);
const go = () => show(location.hash.replace("#", "") || "analyse");
addEventListener("hashchange", go);
let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(go, 120); });
document.fonts?.ready.then(go);
go();
