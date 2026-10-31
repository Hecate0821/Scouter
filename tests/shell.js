// The console shell in jsdom: the source switch (ideal / modeled / measured), the CLI and JSON views, and the URL state.
// - Every view under every source shows either some blocks or the empty state, never both and never neither; measured blocks
//   show only under "measured", computed-only blocks never do; no NaN / undefined in what is visible.
// - The page and the data plane agree: the call behind the CLI / JSON buttons (planeCall) gives the same step time as the
//   page's own engine call, and the printed command, run through tools/scouter.js, prints that same number again.
// - ?model=&hw=&topo=&src= restores the context.
// Usage: node shell.js
const path = require("path");
const { execSync } = require("child_process");
const load = require("./load");

const ROOT = path.resolve(__dirname, "..");
const problems = [];
(async () => {
  const { w, d, errs } = await load(path.join(ROOT, "index.html"), "#infer/step");
  const go = h => { w.location.hash = h; w.dispatchEvent(new w.HashChangeEvent("hashchange")); };
  const setSel = (sel, v) => { const el = d.querySelector(sel); el.value = v; el.dispatchEvent(new w.Event("change", { bubbles: true })); };
  const src = s => d.querySelector(`#g-src button[data-src="${s}"]`).click();
  const shown = el => { for (let x = el; x && x !== d.body; x = x.parentElement) if (x.hidden || x.classList.contains("src-off")) return false; return true; };
  const VIEWS = Array.from(d.querySelectorAll("section[data-pane]")).map(s => s.dataset.pane);

  // --- every view under every source ---
  for (const s of ["ideal", "measured", "modeled"]) {
    src(s);
    if (w.eval("S.src") !== s || d.body.dataset.src !== s) problems.push(`source ${s} not applied`);
    if (d.querySelector(`#g-src button[data-src="${s}"]`).getAttribute("aria-checked") !== "true") problems.push(`source ${s}: button not checked`);
    for (const v of VIEWS) {
      go(v);
      const pane = d.querySelector(`section[data-pane="${v}"]`), blocks = Array.from(pane.querySelectorAll(".block")).filter(shown);
      const empty = shown(d.getElementById("g-empty"));
      if (!blocks.length === !empty) problems.push(`${s} ${v}: ${blocks.length} blocks and ${empty ? "an" : "no"} empty state`);
      for (const b of blocks) {
        const tag = b.dataset.src || (b.querySelector("[data-src]") ? "mixed" : "");
        if (s === "measured" && !tag) problems.push(`${s} ${v}: a computed block is shown`);
        if (s !== "measured" && tag === "measured") problems.push(`${s} ${v}: a measured block is shown`);
        const bad = (b.textContent.match(/(?<!得到 )NaN|undefined/g) || []).length;
        if (bad) problems.push(`${s} ${v}: ${bad} NaN / undefined`);
      }
      if (s !== "measured") for (const el of pane.querySelectorAll('[data-src="measured"]')) if (shown(el)) problems.push(`${s} ${v}: measured content shown`);
    }
  }
  const adv = d.querySelector("#g-adv input");
  src("ideal"); if (!adv.disabled || w.eval("S.glob.etaC") !== 1) problems.push("ideal: efficiency inputs should be locked at the hardware roofline");
  src("modeled"); if (adv.disabled || w.eval("S.glob.etaC") !== w.eval("Common.G0.etaC")) problems.push("modeled: efficiencies should be back to the defaults");

  // --- page, data plane and command line agree ---
  const CASES = [
    ["infer/step", "Infer.step(S.model,HW[S.hw],TOPO[S.topo],S.inf,S.glob).stepT", () => setSel("[data-bind='inf.engine']", "vllm")],
    ["train/step", "Train.run(S.model,HW[S.hw],TOPO[S.topo],Object.assign({N:TOPO[S.topo].gpus},S.trn),S.glob).stepT", () => setSel("[data-bind='trn.fw']", "torchtitan")],
    ["arch/module", "Arch.measure(S.model,HW[S.hw],S.glob,S.arch).mods.full.t", () => setSel("[data-bind='arch.phase']", "prefill")],
  ];
  const PICK = { infer: "stepT", train: "stepT", module: "measure.mods.full.t" };
  setSel("#g-model", "kimik3"); setSel("#g-hw", "B200");
  for (const [view, expr, tweak] of CASES) for (const s of ["modeled", "ideal"]) {
    go(view); tweak(); src(s);
    const page = w.eval(expr), [cmd, q] = w.eval("planeCall()"), get = (o, p) => p.split(".").reduce((x, k) => x[k], o);
    const api = get(w.eval("Scouter")[cmd](q), PICK[cmd]);
    if (Math.abs(api - page) > 1e-12 * page) problems.push(`${view} ${s}: data plane ${api} != page ${page}`);
    d.getElementById("g-cli").click();
    const line = d.getElementById("g-dlg-pre").textContent.split("\n")[0];
    if (!line.startsWith("node tools/scouter.js " + cmd)) problems.push(`${view} ${s}: CLI line is "${line}"`);
    else {
      const out = JSON.parse(execSync(line + " --pick " + PICK[cmd], { cwd: ROOT, encoding: "utf8", shell: "/bin/sh" }));
      if (Math.abs(out[PICK[cmd]] - page) > 1e-12 * page) problems.push(`${view} ${s}: CLI prints ${out[PICK[cmd]]}, page has ${page}\n    ${line}`);
    }
    d.getElementById("g-dlg-x").click();
    d.getElementById("g-json").click();
    try { JSON.parse(d.getElementById("g-dlg-pre").textContent); } catch (e) { problems.push(`${view} ${s}: JSON view does not parse`); }
    if (d.getElementById("g-dlg").hidden) problems.push(`${view} ${s}: dialog did not open`);
    d.getElementById("g-dlg-x").click();
  }
  if (errs.length) problems.push(...errs.slice(0, 5));

  // --- URL state ---
  const u = await load(path.join(ROOT, "index.html"), "#arch/module", null, "?model=glm53f&hw=H100&topo=hgx8&src=ideal");
  const st = u.w.eval("[S.modelKey, S.hw, S.topo, S.src, S.glob.etaM].join(' ')");
  if (st !== "glm53f H100 hgx8 ideal 1") problems.push(`URL state: got "${st}"`);
  if (u.errs.length) problems.push(...u.errs.slice(0, 5));

  console.log(`shell: ${VIEWS.length} views × 3 sources, ${CASES.length * 2} page / plane / CLI comparisons`);
  if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
  console.log("shell ok");
  process.exit(0);
})();
