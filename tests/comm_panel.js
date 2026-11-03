// The parallelism page reads like a Perfetto trace: bars open a scaling panel on click, several at once, with no hover layer.
// Also pins the claim the panel makes about linear attention: state-passing CP bytes do not grow with the sequence.
// And how overlap reads: overlapping bars share time with compute, blocking bars never do, and the SM track shows the wait.
const path = require("path");
const load = require("./load");
const problems = [];
(async () => {
  const { w, d, errs } = await load(path.resolve(__dirname, "..", "index.html"), "#train/par", null, "?model=qwen35&hw=H200&topo=hgx8");
  const set = (bind, v) => { const el = d.querySelector(`[data-bind="${bind}"]`); el.value = v; el.dispatchEvent(new w.Event(el.tagName === "SELECT" ? "change" : "input", { bubbles: true })); };
  set("trn.cp", 2); set("trn.linCP", "state");
  const grid = () => d.getElementById("x-t-grid"), cells = () => Array.from(grid().querySelectorAll("[data-xk]"));
  const panels = () => grid().querySelectorAll(".xp");
  if (grid().querySelector("[data-tip]")) problems.push("the timeline still has hover tips");
  if (d.getElementById("x-t-vol").querySelector("[data-tip]")) problems.push("the volume bars still have hover tips");
  if (!/\d/.test(d.getElementById("x-t-scale").textContent)) problems.push("no token scaling summary");
  const key = pred => { const c = cells().find(pred); if (!c) problems.push("bar not found: " + pred); return c && c.dataset.xk; };
  const click = k => cells().find(c => c.dataset.xk === k).click();
  const lin = key(c => /^x\|cp\|AG\|fwd/.test(c.dataset.xk) && c.getAttribute("aria-label").includes("FP32"));
  const comp = key(c => c.dataset.xk[0] === "c");
  if (panels().length) problems.push("panels open before any click");
  click(lin); if (panels().length !== 1) problems.push(`one click: ${panels().length} panels`);
  click(comp); if (panels().length !== 2) problems.push(`two clicks: ${panels().length} panels`);
  if (grid().querySelectorAll(".xtb.on").length !== 2 || grid().querySelectorAll(".xtg .gd").length !== 4) problems.push("open bars should be marked, with guides at their ends");
  const figs = Array.from(grid().querySelectorAll(".xp .xm"));
  if (figs.length < 4) problems.push(`expected charts for both open bars, got ${figs.length}`);
  const linP = Array.from(panels()).find(p => p.textContent.includes("FP32"));
  const seq = linP && Array.from(linP.querySelectorAll(".xm")).find(f => f.querySelector("figcaption b").textContent === w.eval('t("comm.knob.seq")'));
  if (!seq) problems.push("no sequence-length chart on the linear CP panel");
  else { const vals = Array.from(seq.querySelectorAll("text.xv")).map(x => x.textContent);
    if (vals.length < 3 || new Set(vals).size !== 1) problems.push(`linear CP state bytes should stay flat with the sequence: ${vals}`); }
  click(lin); if (panels().length !== 1) problems.push("clicking again should close only that panel");
  click(comp);
  // State-passing linear CP draws both exchanges, forward and backward
  for (const pass of ["fwd", "bwd"]) for (const op of ["P2P", "AG"])
    if (!cells().some(c => c.dataset.xk.startsWith(`x|cp|${op}|${pass}`))) problems.push(`linear CP is missing ${op} on ${pass}`);
  // The claim opens the block and the legend explains the marks
  if (!d.querySelector("#par-comm > h2 + p.note").textContent.includes("SM")) problems.push("the overlap claim is missing");
  for (const m of ["wait", "block", "ov", "sm"]) if (!d.querySelector(`#par-comm .xleg .xk.${m}`)) problems.push(`the legend does not explain ${m}`);
  // Overlap is position on one time axis
  const span = b => { const m = /left:([\d.]+)%;width:([\d.]+)%/.exec(b.getAttribute("style")); return [+m[1], +m[1] + +m[2]]; };
  // Styles round left and width separately to 0.001%, so a bar ending where the next begins can read as a hair of overlap
  const meets = (a, b) => a[0] < b[1] - 0.01 && b[0] < a[1] - 0.01;
  const strips = () => Array.from(grid().querySelectorAll(".xst")), compsOf = st => Array.from(st.querySelectorAll(".xtr.sm button.xtb")).map(span);
  for (const st of strips()) for (const b of st.querySelectorAll("button.xtb.block"))
    if (compsOf(st).some(c => meets(span(b), c))) problems.push(`a blocking bar runs alongside compute: ${b.dataset.xk}`);
  if (!grid().querySelector(".xtr.sm .xtw")) problems.push("the SM track should show the wait for blocking calls");
  set("trn.cpType", "ulysses");
  for (const pass of ["fwd", "bwd"]) {
    const a2a = Array.from(grid().querySelectorAll(`button.xtb[data-xk^="x|cp|A2A|${pass}"]`));
    if (a2a.length !== 2) { problems.push(`Ulysses ${pass}: expected 2 all-to-alls, got ${a2a.length}`); continue; }
    for (const b of a2a) { if (!b.classList.contains("over")) problems.push(`Ulysses ${pass}: all-to-all should be drawn as overlap`);
      if (!compsOf(b.closest(".xst")).some(c => meets(span(b), c))) problems.push(`Ulysses ${pass}: all-to-all should share time with compute`); }
  }
  // Inference shares the chart
  const inf = await load(path.resolve(__dirname, "..", "index.html"), "#infer/deploy");
  const ig = inf.d.getElementById("x-i-grid"), ib = ig.querySelector("button.xtb[data-xk]");
  if (!ig.querySelector(".xtr.sm") || !ib) problems.push("the inference timeline did not render");
  else { ib.click(); if (ig.querySelectorAll(".xp").length !== 1) problems.push("the inference timeline does not open a panel"); }
  // Defaults, with no URL state
  const u = await load(path.resolve(__dirname, "..", "index.html"), "#train/par");
  const st = u.w.eval('[S.trn.fw, S.hw, S.topo].join(" ")');
  if (st !== "torchtitan GB300 nvl64") problems.push(`defaults: ${st}`);
  for (const e of errs.concat(u.errs, inf.errs)) problems.push("page error: " + e);
  if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
  console.log("comm panel ok");
  process.exit(0);
})();
