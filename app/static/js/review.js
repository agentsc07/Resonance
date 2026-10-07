// Listening confirmation: play an engine flag that no human mark matched (1 s before to 1 s after) and mark Agree / Disagree / Unsure.
const $ = (s) => document.querySelector(s);
const api = async (p, o) => { const r = await fetch(p, o); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); };
let reviewer = "", flags = [], au = new Audio(), stopAt = 0;
au.ontimeupdate = () => { if (au.currentTime >= stopAt) au.pause(); };
$("#nm").value = (() => { try { return localStorage.getItem("flawline_rater") || ""; } catch { return ""; } })();
$("#go").onclick = async () => {
  try {
    const d = await api("/api/human/review/flags?reviewer=" + encodeURIComponent($("#nm").value));
    reviewer = d.reviewer; flags = d.flags; $("#s0").classList.add("hide"); draw();
  } catch (e) { $("#e").textContent = e.message; }
};
function draw() {
  const done = flags.filter((f) => f.verdict).length, ag = flags.filter((f) => f.verdict === "agree").length;
  $("#sum").classList.remove("hide");
  $("#sum").textContent = `${done} of ${flags.length} reviewed · agree ${ag}, disagree ${flags.filter((f) => f.verdict === "disagree").length}, unsure ${flags.filter((f) => f.verdict === "unsure").length}`;
  const box = $("#list"); box.innerHTML = "";
  flags.forEach((f) => {
    const row = document.createElement("div"); row.className = "panel";
    row.innerHTML = `<button class="go">▶ Play</button><div class="what"><b>${f.name}</b><span>${f.category} · ${f.clip} · ${f.start_s.toFixed(1)} s${f.unedited ? " · unedited reading" : ""}</span></div>
      ${["agree", "disagree", "unsure"].map((v) => `<button class="v ${v} ${f.verdict === v ? "on" : ""}" data-v="${v}">${v[0].toUpperCase() + v.slice(1)}</button>`).join("")}`;
    row.querySelector(".go").onclick = () => {
      au.src = "/api/human/audio/" + encodeURIComponent(f.clip);
      const a = Math.max(0, f.start_s - 1), b = f.end_s + 1; stopAt = b;
      au.onloadedmetadata = () => { au.currentTime = a; au.play(); };
      if (au.readyState >= 1) { au.currentTime = a; au.play(); }
    };
    row.querySelectorAll(".v").forEach((b) => b.onclick = async () => {
      await api("/api/human/review/verdict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer, flag: f.flag, verdict: b.dataset.v }) });
      f.verdict = b.dataset.v; draw();
    });
    box.appendChild(row);
  });
}
