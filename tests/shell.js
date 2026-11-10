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

  // --- every view under every source; "modeled" twice: on a real model it is HuggingArch, on a custom one Scouter's own model ---
  const views = (s, who) => {
    src(s); const hga = w.eval("hgaOn()");
    if (w.eval("S.src") !== s || d.body.dataset.src !== s) problems.push(`${who}: source ${s} not applied`);
    if (d.querySelector(`#g-src button[data-src="${s}"]`).getAttribute("aria-checked") !== "true") problems.push(`${who}: button not checked`);
    for (const v of VIEWS) {
      go(v);
      const pane = d.querySelector(`section[data-pane="${v}"]`), blocks = Array.from(pane.querySelectorAll(".block")).filter(shown);
      const empty = shown(d.getElementById("g-empty"));
      if (!blocks.length === !empty) problems.push(`${who} ${v}: ${blocks.length} blocks and ${empty ? "an" : "no"} empty state`);
      for (const b of blocks) {
        const tag = b.dataset.src || (b.querySelector("[data-src]") ? "mixed" : "");
        if (s === "measured" && !tag) problems.push(`${who} ${v}: a computed block is shown`);
        if (s !== "measured" && tag === "measured") problems.push(`${who} ${v}: a measured block is shown`);
        if (tag === "hga" && !hga) problems.push(`${who} ${v}: a HuggingArch block is shown off the HuggingArch basis`);
        const bad = (b.textContent.match(/(?<!得到 )NaN|undefined/g) || []).length;
        if (bad) problems.push(`${who} ${v}: ${bad} NaN / undefined`);
      }
      if (s !== "measured") for (const el of pane.querySelectorAll('[data-src="measured"]')) if (shown(el)) problems.push(`${who} ${v}: measured content shown`);
      if (hga && (v === "infer/step" || v === "train/step") && !shown(pane.querySelector('[data-src="hga"]'))) problems.push(`${who} ${v}: the HuggingArch block is hidden`);
    }
  };
  const adv = d.querySelector("#g-adv input"), srcBtn = () => d.querySelector('#g-src button[data-src="modeled"]').textContent;
  if (!w.eval("realModel()")) problems.push(`default model ${w.eval("S.modelKey")} should be a real model`);
  for (const s of ["ideal", "measured", "modeled"]) views(s, `${w.eval("S.modelKey")} ${s}`);
  src("ideal"); if (!adv.disabled || w.eval("S.glob.etaC") !== 1) problems.push("ideal: efficiency inputs should be locked at the hardware roofline");
  // A real model's middle button is HuggingArch: the page computes on the ideal basis beside it, never on Scouter's modeled one
  src("modeled"); if (!adv.disabled || w.eval("S.glob.etaC") !== 1 || !w.eval("S.glob.lb")) problems.push("real model under HuggingArch: efficiencies should stay locked at the ideal basis");
  if (srcBtn() !== w.eval('t("shell.src.hga")')) problems.push(`real model: middle button reads "${srcBtn()}"`);
  // An edited model is no longer the one HuggingArch simulated
  w.eval("S.model.L += 1; syncBasis(); render(); applySource()");
  if (w.eval("hgaOn()") || adv.disabled || w.eval("S.glob.etaC") !== w.eval("Common.G0.etaC")) problems.push("edited real model: should be back on the modeled basis");
  if (srcBtn() !== w.eval('t("shell.src.modeled")')) problems.push(`edited model: middle button reads "${srcBtn()}"`);
  setSel("#g-model", "musespark");
  if (adv.disabled || w.eval("S.glob.etaC") !== w.eval("Common.G0.etaC")) problems.push("custom model: modeled efficiencies should be back to the defaults");
  if (srcBtn() !== w.eval('t("shell.src.modeled")')) problems.push(`custom model: middle button reads "${srcBtn()}"`);
  views("modeled", "musespark modeled");

  // --- page, data plane and command line agree ---
  const CASES = [
    ["infer/step", "Infer.step(S.model,HW[S.hw],TOPO[S.topo],S.inf,S.glob).stepT", () => setSel("[data-bind='inf.engine']", "vllm")],
    ["train/step", "Train.run(S.model,HW[S.hw],TOPO[S.topo],Object.assign({N:TOPO[S.topo].gpus},S.trn),S.glob).stepT", () => setSel("[data-bind='trn.fw']", "torchtitan")],
    ["arch/module", "Arch.measure(S.model,HW[S.hw],S.glob,S.arch).mods.full.t", () => setSel("[data-bind='arch.phase']", "prefill")],
    ["infer/kv", "KV.plan(S.model,HW[S.hw],TOPO[S.topo],Object.assign({},S.inf,S.kv),S.glob).ttft", () => setSel("[data-bind='kv.routing']", "random")],
  ];
  const PICK = { infer: "stepT", train: "stepT", module: "measure.mods.full.t", kv: "ttft" };
  setSel("#g-model", "musespark"); setSel("#g-hw", "B200");   // the modeled basis lives on custom models now
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
  // --- a real model's step pages: the HuggingArch block, the data plane and the CLI read the same cell ---
  setSel("#g-model", "qwen35"); setSel("#g-hw", "H200"); src("modeled");
  const HGA_CASES = [
    ["infer/step", "hga", "Scouter.hga(hgaInferQuery()).decode.hga.t", "decode.hga.t", "ih", "vllm"],
    ["train/step", "hgaTrain", "Scouter.hgaTrain(hgaTrainQuery()).hga.t", "hga.t", "th", "torchtitan"],
  ];
  for (const [view, want, expr, at, pre, eng] of HGA_CASES) {
    go(view);
    for (let i = 0; i < 100 && w.eval("hgaReady(S.modelKey)") === "loading"; i++) await new Promise(r => setTimeout(r, 50));
    d.querySelector(`#${pre}-eng button[data-k="${eng}"]`).click();
    if (d.getElementById(`${pre}-body`).hidden) { problems.push(`${view} HuggingArch: block is empty — ${d.getElementById(`${pre}-lede`).textContent}`); continue; }
    const page = w.eval(expr), [cmd, q] = w.eval("planeCall()"), get = (o, p) => p.split(".").reduce((x, k) => x[k], o);
    if (cmd !== want || q.engine !== eng) { problems.push(`${view} HuggingArch: planeCall is ${cmd} ${JSON.stringify(q)}`); continue; }
    const api = get(w.eval("Scouter")[cmd](q), at);
    if (!(page > 0) || Math.abs(api - page) > 1e-12 * page) problems.push(`${view} HuggingArch: data plane ${api} != page ${page}`);
    d.getElementById("g-cli").click();
    const line = d.getElementById("g-dlg-pre").textContent.split("\n")[0];
    if (!line.startsWith("node tools/scouter.js " + cmd + " ")) problems.push(`${view} HuggingArch: CLI line is "${line}"`);
    else {
      const out = JSON.parse(execSync(line + " --pick " + at, { cwd: ROOT, encoding: "utf8", shell: "/bin/sh" }));
      if (Math.abs(out[at] - page) > 1e-12 * page) problems.push(`${view} HuggingArch: CLI prints ${out[at]}, page has ${page}\n    ${line}`);
    }
    d.getElementById("g-dlg-x").click();
  }
  if (errs.length) problems.push(...errs.slice(0, 5));

  // --- the step page hands its workload to the module page ---
  const sp = await load(path.join(ROOT, "index.html"), "#infer/step", null, "?model=qwen35&hw=H200&topo=hgx8&src=ideal");
  const spSet = (k, v) => { const el = sp.d.querySelector(`[data-bind="${k}"]`); el.value = v; el.dispatchEvent(new sp.w.Event(el.tagName === "SELECT" ? "change" : "input", { bubbles: true })); };
  const toMods = () => { sp.w.eval('showTab("infer/step")'); sp.d.querySelector("#i-to-mods a").click(); sp.w.eval('showTab("arch/module/full")'); };
  const ACORE = 'fT((o=>o.t/o.layers)(infRun().ops.find(x=>x.key==="acore")))';
  spSet("inf.B", 32); spSet("inf.n", 4096); toMods();
  const paged = () => Array.from(sp.d.querySelectorAll("#am-ops tbody tr")).map(r => Array.from(r.children).map(c => c.textContent.trim())).find(r => r[0].includes("paged attention"));
  if (sp.w.eval("[S.arch.B, S.arch.n].join()") !== "32,4096") problems.push("step page: B and context did not reach the module page");
  const pg = paged(); if (!pg || pg[4] !== sp.w.eval(ACORE)) problems.push(`step page: attention core is ${sp.w.eval(ACORE)}, module page shows ${pg && pg[4]}`);
  // Every precision the rack offers has to be selectable on the module page, or its dropdown goes blank
  for (const [from, to] of [["wDt", "eDt"], ["dDt", "wDt"], ["kvDt", "kvDt"], ["stateDt", "stateDt"]])
    for (const v of Array.from(sp.d.querySelector(`[data-bind="inf.${from}"]`).options).map(o => o.value)) {
      const base = sp.w.eval(`S.inf.${from}`); spSet(`inf.${from}`, v); toMods();
      if (sp.d.querySelector(`[data-bind="arch.${to}"]`).value !== v) problems.push(`step page: ${from}=${v} leaves arch.${to} blank`);
      if (/NaN|undefined/.test(sp.d.getElementById("am-ops").textContent)) problems.push(`step page: ${from}=${v} breaks the module table`);
      spSet(`inf.${from}`, base); }
  spSet("inf.phase", "prefill"); spSet("inf.P", 16384); toMods();
  if (sp.w.eval("[S.arch.phase, S.arch.n].join()") !== "prefill,16384") problems.push("step page: prefill should carry the prompt length");
  if (sp.errs.length) problems.push(...sp.errs.slice(0, 5));

  // --- appendix/data is a data page: all of it shows under every source ---
  { const prev = w.eval("S.src");
    for (const s of ["ideal", "modeled", "measured"]) { src(s); go("appendix/data");
      const bs = Array.from(d.querySelectorAll('section[data-pane="appendix/data"] > .block'));
      if (bs.filter(shown).length !== bs.length) problems.push(`appendix/data under ${s}: ${bs.filter(shown).length} of ${bs.length} sections shown`); }
    src(prev); }

  // --- train/step keeps what only it shows; the mesh and the per-pass roofline moved to train/par ---
  const ts = await load(path.join(ROOT, "index.html"), "#train/step", null, "?model=qwen35&hw=H200&topo=hgx8&src=ideal");
  const tsShow = h => ts.w.eval(`showTab("${h}")`);
  const tsSet = (k, v) => { const el = ts.d.querySelector(`[data-bind="${k}"]`); el.value = v; el.dispatchEvent(new ts.w.Event(el.tagName === "SELECT" ? "change" : "input", { bubbles: true })); };
  const tsPane = ts.d.querySelector('section[data-pane="train/step"]');
  if (tsPane.querySelectorAll(":scope > .block").length !== 8) problems.push(`train/step: ${tsPane.querySelectorAll(":scope > .block").length} sections, expected 8 (HuggingArch first)`);
  for (const id of ["t-comm", "t-gantt", "t-analysis", "t-roof", "t-mesh", "t-groups"]) if (ts.d.getElementById(id)) problems.push(`train/step: ${id} should be gone`);
  // The a2a note follows the config: it shows only while a2a outlasts the expert GEMM
  const epSlow = () => ts.w.eval("(()=>{ const r=trnRun(), n=ctx().d.nMoe; return r.ep>1 && r.a2aPerLayer>r.moeT/n; })()");
  const barNote = ts.d.getElementById("t-bar-note");
  if (barNote.hidden === epSlow()) problems.push("train/step: a2a note out of step with the run");
  if (!barNote.hidden && !barNote.querySelector('a[href="#train/par/ep"]')) problems.push("train/step: a2a note lost its link");
  const ep0 = ts.w.eval("S.trn.ep"); tsSet("trn.ep", 1); ts.w.eval("render()");
  if (!barNote.hidden) problems.push("train/step: a2a note shown with EP = 1");
  tsSet("trn.ep", ep0); ts.w.eval("render()");
  // The operator table hands its sequence length and precision to the module page, and the two agree per layer
  const tsToMods = () => { tsShow("train/step"); ts.d.querySelector('#t-to-mods a[href="#arch/module"]').click(); tsShow("arch/module/full"); };
  tsToMods();
  if (ts.w.eval("[S.arch.phase, S.arch.S===S.trn.S, S.arch.prec===S.trn.prec].join()") !== "train,true,true") problems.push("train/step: workload did not reach the module page");
  const qkvMod = Array.from(ts.d.querySelectorAll("#am-ops tbody tr")).map(r => Array.from(r.children).map(c => c.textContent.trim())).find(r => r[0].includes("QKV"));
  const qkvStep = ts.w.eval('fT((o=>(o.fwd.t+o.dgrad.t+o.wgrad.t)/o.layers)(trnRun().ops.find(x=>x.key==="aqkv")))');
  if (!qkvMod || qkvMod[4] !== qkvStep) problems.push(`train/step: QKV per layer is ${qkvStep}, module page shows ${qkvMod && qkvMod[4]}`);
  for (const v of Array.from(ts.d.querySelector('[data-bind="trn.prec"]').options).map(o => o.value)) {
    const base = ts.w.eval("S.trn.prec"); tsSet("trn.prec", v); tsToMods();
    if (ts.d.querySelector('[data-bind="arch.prec"]').value !== v) problems.push(`train/step: prec=${v} leaves arch.prec blank`);
    if (/NaN|undefined/.test(ts.d.getElementById("am-ops").textContent)) problems.push(`train/step: prec=${v} breaks the module table`);
    tsSet("trn.prec", base); }
  // Mesh on train/par: it renders and answers clicks, for either framework
  tsShow("train/par");
  const meshOK = tag => { const dims = ts.d.querySelectorAll("#p-mesh-dim button"), gpus = ts.d.querySelectorAll("#p-mesh button[data-k]");
    if (!dims.length || !gpus.length) { problems.push(`train/par ${tag}: mesh is empty`); return; }
    gpus[1].click(); if (ts.w.eval("meshSel") !== 1) problems.push(`train/par ${tag}: picking a GPU does nothing`);
    const dv = ts.d.querySelectorAll("#p-mesh-dim button")[dims.length - 1].dataset.v; ts.d.querySelectorAll("#p-mesh-dim button")[dims.length - 1].click();
    if (ts.w.eval("meshDim") !== dv) problems.push(`train/par ${tag}: changing the mesh color does nothing`);
    if (ts.d.querySelectorAll("#p-groups thead th").length !== 6 || ts.d.querySelectorAll("#p-groups tbody tr:first-child td").length !== 6) problems.push(`train/par ${tag}: groups table should have 6 columns`);
    if (/NaN|undefined/.test(ts.d.getElementById("par-mesh").textContent)) problems.push(`train/par ${tag}: mesh shows NaN / undefined`); };
  meshOK("torchtitan");
  // Roofline: two views of one chart; each draws its own axes and legend, and neither leaks into the other
  const roof = ts.d.getElementById("p-roof"), roofLede = ts.d.getElementById("p-roof-lede"), roofLeg = ts.d.getElementById("p-roof-legend");
  const pick = v => ts.d.querySelector(`#p-roof-view button[data-v="${v}"]`).click();
  const aggLede = roofLede.textContent, aggLeg = roofLeg.textContent;
  if (roof.getAttribute("viewBox") !== "0 0 680 360" || !roof.querySelector("circle")) problems.push("train/par: aggregate roofline not drawn");
  pick("pass");
  if (roof.getAttribute("viewBox") !== "0 0 680 380") problems.push("train/par: per-pass roofline has the wrong canvas");
  if (roofLede.textContent === aggLede || roofLeg.textContent === aggLeg) problems.push("train/par: lede or legend did not follow the view");
  if (/NaN|undefined/.test(roof.innerHTML + roofLede.textContent + roofLeg.textContent)) problems.push("train/par: per-pass roofline shows NaN / undefined");
  ts.w.eval("S.trn.prec='mxfp8'; render()");
  if (/NaN|undefined/.test(roof.innerHTML)) problems.push("train/par: per-pass roofline breaks under low precision");
  ts.w.eval("S.trn.prec='bf16'; render()");
  pick("agg");
  if (roof.getAttribute("viewBox") !== "0 0 680 360" || roofLede.textContent !== aggLede) problems.push("train/par: switching back to the aggregate view failed");
  tsShow("train/step"); ts.d.querySelector('#t-to-mods a[href="#train/par/roof"]').click();
  if (ts.w.eval("roofView") !== "pass") problems.push("train/step: the per-pass link should open that view");
  ts.w.eval('roofView="agg"');
  tsSet("trn.fw", "megatron"); tsShow("train/par"); meshOK("megatron");
  if (ts.errs.length) problems.push(...ts.errs.slice(0, 5));

  // --- train/par: FSDP / CP / EP are a switch, and every way of reaching one opens that one ---
  const dimOn = () => ["fsdp", "cp", "ep"].filter(k => !ts.d.getElementById("par-" + k).hidden).join();
  const dimBtn = () => ts.d.querySelector('#p-dim button[aria-pressed="true"]').dataset.v;
  const tsGo = h => { ts.w.location.hash = h; ts.w.dispatchEvent(new ts.w.HashChangeEvent("hashchange")); };
  const dimIs = (want, how) => { if (dimOn() !== want || dimBtn() !== want) problems.push(`train/par ${how}: shows ${dimOn()} (button ${dimBtn()}), expected ${want}`); };
  tsShow("train/step"); tsShow("train/par");
  if (ts.d.querySelectorAll("#p-dim button").length !== 3) problems.push("train/par: switch should have 3 buttons");
  dimIs("fsdp", "default");
  // All three stay drawn while hidden, so a switch never shows a blank section
  for (const id of ["p-f-tbl", "p-c-gantt", "p-e-gantt"]) if (!ts.d.getElementById(id).children.length) problems.push(`train/par: ${id} is empty while hidden`);
  // jsdom refuses replaceState on file:// URLs, so watch the call here; the real address bar is checked in a browser
  let replaced = ""; ts.w.history.replaceState = (x, y, u) => { replaced = u; };
  ts.d.querySelector('#p-dim button[data-v="cp"]').click(); dimIs("cp", "button");
  delete ts.w.history.replaceState;
  if (replaced !== "#train/par/cp") problems.push(`train/par: address set to "${replaced}" after the switch`);
  ts.w.eval("render()"); dimIs("cp", "after a render");
  tsGo("#overview"); tsGo("#train/par/ep"); dimIs("ep", "link from another page");
  const scrolled = []; ts.w.Element.prototype.scrollIntoView = function () { scrolled.push(this.id); };
  tsGo("#train/par/fsdp"); dimIs("fsdp", "link on the same page");
  if (scrolled[scrolled.length - 1] !== "par-fsdp") problems.push(`train/par: same-page link scrolled to ${scrolled[scrolled.length - 1]}`);
  tsGo("#train/par/roof"); dimIs("fsdp", "unrelated anchor");
  // The address already says #train/par/cp and the browser sends no hashchange: the link must still go there
  tsGo("#train/par/cp"); scrolled.length = 0;
  const sameLink = Array.from(ts.d.querySelectorAll('section[data-pane="train/par"] a[href="#train/par/cp"]'))[0];
  if (!sameLink) problems.push("train/par: no link to the CP section"); else { sameLink.click(); if (scrolled[0] !== "par-cp") problems.push("train/par: clicking the current link does not go back to it"); }
  ts.w.Element.prototype.scrollIntoView = function () {};
  ts.d.querySelector('#g-src button[data-src="measured"]').click();
  if (!ts.d.getElementById("par-dims").classList.contains("src-off")) problems.push("train/par: switch stays on under measured");
  ts.d.querySelector('#g-src button[data-src="ideal"]').click();
  tsGo("#train/par/fsdp");
  // --- arch/overview is the current model, arch/compare is several ---
  const pane = k => ts.d.querySelector(`section[data-pane="${k}"]`);
  for (const id of ["m-compare", "m-cache", "m-cache-legend"]) {
    if (!pane("arch/compare").querySelector("#" + id)) problems.push(`arch/compare: missing ${id}`);
    if (pane("arch/overview").querySelector("#" + id)) problems.push(`arch/overview: ${id} should have moved`); }
  tsShow("arch/overview");
  if (!ts.d.getElementById("m-stack").children.length || !ts.d.querySelectorAll("#m-params tbody tr").length) problems.push("arch/overview: current model is empty");
  tsShow("arch/compare");
  const nModels = ts.w.eval("Object.keys(MODELS).length");
  if (ts.d.querySelectorAll("#m-compare tbody tr").length !== nModels) problems.push("arch/compare: model table does not list every model");
  if (ts.d.querySelectorAll("#m-compare tbody tr.hl").length !== 1) problems.push("arch/compare: current model not marked");
  if (!ts.d.querySelectorAll("#m-cache path").length || !ts.d.getElementById("m-cache-legend").textContent.trim()) problems.push("arch/compare: cache chart is empty");
  if (/NaN|undefined/.test(pane("arch/compare").textContent)) problems.push("arch/compare: NaN or undefined");
  // --- overview cards, glossary header, determinism citations ---
  const cards = Array.from(pane("overview").querySelectorAll(".cards a.card")).map(a => a.getAttribute("href"));
  if (cards.join() !== "#arch/overview,#train/step,#infer/step,#rl/mismatch,#other/network,#appendix/data") problems.push(`overview: cards are ${cards.join()}`);
  if (/模型结构|model structure/.test(pane("overview").querySelector('a[href="#appendix/data"]').textContent)) problems.push("overview: appendix card still claims the model structure");
  if (/本页|on this page/.test(pane("appendix/quant").querySelector("table thead th:nth-child(3)") ? Array.from(pane("appendix/quant").querySelectorAll("thead th")).map(x => x.textContent).join() : "")) problems.push("appendix/quant: glossary header says this page");
  tsShow("rl/determinism");
  const det = pane("rl/determinism"), detText = det.textContent;
  if (/\{[a-zA-Z0-9]+\}/.test(detText) || /NaN|undefined/.test(detText)) problems.push("rl/determinism: unfilled placeholder or bad number");
  if (!ts.d.getElementById("determinism-oss")) problems.push("rl/determinism: release list has no anchor");
  if (det.querySelectorAll('a[href="#rl/determinism/oss"]').length !== 2) problems.push("rl/determinism: pointers to the release list went missing");
  // What the pointers promise has to be in the list they point at
  const ossLinks = Array.from(ts.d.querySelectorAll("#r-det-oss a")).map(a => a.getAttribute("href")).join(" ");
  for (const v of ["v0.12.0", "v0.22.0", "v0.29.0", "v0.20.0", "v0.28.0", "v0.5.21"]) if (!ossLinks.includes(v)) problems.push(`rl/determinism: ${v} is not in the release list`);
  const ossCell = ts.d.getElementById("r-det-oss").textContent;
  for (const n of ["18.1%", "28.9%", "33.6%"]) if (!ossCell.includes(n)) problems.push(`rl/determinism: release list lost ${n}`);
  // One source run in the data table links once; each row keeps its own data
  const pubRows = Array.from(ts.d.querySelectorAll("#r-det-pub tbody tr")), srcUrls = pubRows.map(r => r.children[1].querySelector("a")).filter(Boolean).map(a => a.getAttribute("href"));
  if (new Set(srcUrls).size !== srcUrls.length) problems.push("rl/determinism: data table repeats a source link");
  if (pubRows.length !== ts.w.eval('DB.set("determinism_public").length')) problems.push("rl/determinism: data table lost a row");
  if (pubRows.some(r => !r.children[5].textContent.trim())) problems.push("rl/determinism: data row without a value");
  // --- URL state ---
  const u = await load(path.join(ROOT, "index.html"), "#arch/module", null, "?model=glm53f&hw=H100&topo=hgx8&src=ideal");
  const st = u.w.eval("[S.modelKey, S.hw, S.topo, S.src, S.glob.etaM].join(' ')");
  if (st !== "glm53f H100 hgx8 ideal 1") problems.push(`URL state: got "${st}"`);
  if (u.errs.length) problems.push(...u.errs.slice(0, 5));

  console.log(`shell: ${VIEWS.length} views × 3 sources, ${CASES.length * 2 + HGA_CASES.length} page / plane / CLI comparisons`);
  if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
  console.log("shell ok");
  process.exit(0);
})();
