// Render every view × model (plus engine / framework / phase variants) and dump the visible text and labels.
// A refactor that must not change what the page says (e.g. moving text into locale files) is checked by diffing two dumps:
//   node snapshot.js ../index.html out.json              write a dump
//   node snapshot.js ../index.html out.json base.json    write a dump and compare it with base.json
const fs = require("fs");
const load = require("./load");
(async () => {
  const [page = "../index.html", out, base] = process.argv.slice(2);
  const { w, d, errs } = await load(page, "#overview");
  const go = h => { w.location.hash = h; w.dispatchEvent(new w.HashChangeEvent("hashchange")); };
  const setSel = (sel, v) => { const el = d.querySelector(sel); if (!el) return; el.value = v; el.dispatchEvent(new w.Event("change", { bubbles: true })); };
  const norm = s => String(s || "").replace(/\s+/g, " ").trim();
  const attrs = root => Array.from(root.querySelectorAll("[aria-label],[title],[placeholder],[alt]"))
    .map(e => ["aria-label", "title", "placeholder", "alt"].map(a => e.getAttribute(a)).filter(Boolean).join("|")).join(" ¦ ");
  const pane = key => d.querySelector(`section[data-pane="${key}"]`) || d.querySelector(`section[data-pane="${key.split(" ")[0]}"]`);
  const snap = {};
  const take = (label, key) => { const p = pane(key); const g = d.getElementById("g-warn");
    snap[label] = { text: norm(p && p.textContent), attrs: p ? attrs(p) : "", warn: g && !g.hidden ? norm(g.textContent) : "" }; };
  const VIEWS = Array.from(d.querySelectorAll("section[data-pane]")).map(s => s.dataset.pane);
  snap.shell = { title: d.title, text: norm(d.querySelector("header") && d.querySelector("header").textContent) + " ‖ " + Array.from(d.querySelectorAll("nav")).map(n => norm(n.textContent)).join(" ‖ "),
    attrs: Array.from(d.querySelectorAll("nav")).map(attrs).join(" ‖ ") };
  for (const m of ["qwen35", "qwen38_27b", "glm53f", "musespark", "kimik3"]) {
    setSel("#g-model", m);
    const rack = d.querySelector("aside.rack");
    snap[m + " rack"] = { text: norm(rack.textContent), attrs: attrs(rack) };
    for (const v of VIEWS) { go(v); take(m + " " + v, v); }
    for (const fw of ["megatron", "torchtitan"]) {
      go("train/step"); setSel("[data-bind='trn.fw']", fw); take(m + " train/step " + fw, "train/step");
      snap[m + " rack " + fw] = { text: norm(rack.textContent) };
      go("train/par"); take(m + " train/par " + fw, "train/par");
    }
    for (const eng of ["sglang", "vllm", "tokenspeed", "titan"]) {
      go("infer/step"); setSel("[data-bind='inf.engine']", eng);
      for (const ph of ["decode", "prefill"]) { setSel("[data-bind='inf.phase']", ph); go("infer/step"); take(`${m} infer/step ${eng} ${ph}`, "infer/step"); }
      setSel("[data-bind='inf.phase']", "decode");
      snap[m + " rack " + eng] = { text: norm(rack.textContent) };
      go("infer/deploy"); take(`${m} infer/deploy ${eng}`, "infer/deploy");
      go("infer/spec"); take(`${m} infer/spec ${eng}`, "infer/spec");
    }
    setSel("[data-bind='inf.engine']", "sglang");
  }
  if (errs.length) snap.errors = errs.slice(0, 5);
  if (out) fs.writeFileSync(out, JSON.stringify(snap, null, 1));
  if (base) {
    const b = JSON.parse(fs.readFileSync(base, "utf8")); let n = 0;
    for (const k of new Set([...Object.keys(b), ...Object.keys(snap)])) {
      for (const f of ["title", "text", "attrs", "warn"]) {
        const x = b[k] && b[k][f], y = snap[k] && snap[k][f];
        if ((x || "") === (y || "")) continue;
        n++; let i = 0; while (i < Math.min((x || "").length, (y || "").length) && x[i] === y[i]) i++;
        console.log(`DIFF ${k} [${f}] @${i}\n  base: …${(x || "∅").slice(Math.max(0, i - 60), i + 120)}\n  new:  …${(y || "∅").slice(Math.max(0, i - 60), i + 120)}`);
      }
    }
    console.log(n ? n + " differences" : "snapshot identical");
    process.exit(n || errs.length ? 1 : 0);
  }
  console.log("ERRORS:", errs.length ? errs.slice(0, 5).join("\n") : "none");
  process.exit(errs.length ? 1 : 0);
})();
