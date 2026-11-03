// The parallelism page: cells open a scaling panel on click, several at once, with no hover layer left in the grid.
// Also pins the claim the panel makes about linear attention: state-passing CP bytes do not grow with the sequence.
// And how overlap reads: a band joins compute to what overlaps it, and only those rows get one.
const path = require("path");
const load = require("./load");
const problems = [];
(async () => {
  const { w, d, errs } = await load(path.resolve(__dirname, "..", "index.html"), "#train/par", null, "?model=qwen35&hw=H200&topo=hgx8");
  const set = (bind, v) => { const el = d.querySelector(`[data-bind="${bind}"]`); el.value = v; el.dispatchEvent(new w.Event(el.tagName === "SELECT" ? "change" : "input", { bubbles: true })); };
  set("trn.cp", 2); set("trn.linCP", "state");
  const grid = () => d.getElementById("x-t-grid"), cells = () => Array.from(grid().querySelectorAll("[data-xk]"));
  const panels = () => grid().querySelectorAll(".xp");
  if (grid().querySelector("[data-tip]")) problems.push("the grid still has hover tips");
  if (d.getElementById("x-t-vol").querySelector("[data-tip]")) problems.push("the volume bars still have hover tips");
  if (!/\d/.test(d.getElementById("x-t-scale").textContent)) problems.push("no token scaling summary");
  const key = pred => { const c = cells().find(pred); if (!c) problems.push("cell not found: " + pred); return c && c.dataset.xk; };
  const click = k => cells().find(c => c.dataset.xk === k).click();
  const lin = key(c => /^x\|cp\|AG\|fwd/.test(c.dataset.xk) && grid().textContent.length && c.textContent.includes("FP32"));
  const comp = key(c => c.dataset.xk[0] === "c");
  if (panels().length) problems.push("panels open before any click");
  click(lin); if (panels().length !== 1) problems.push(`one click: ${panels().length} panels`);
  click(comp); if (panels().length !== 2) problems.push(`two clicks: ${panels().length} panels`);
  const figs = Array.from(grid().querySelectorAll(".xp .xm"));
  if (figs.length < 4) problems.push(`expected charts for both open cells, got ${figs.length}`);
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
  // Overlap: the claim opens the block, blocking calls get no band, and Ulysses puts both all-to-alls on the attention row with a band
  if (!d.querySelector('#par-comm > h2 + p.note').textContent.includes("SM")) problems.push("the overlap claim is missing");
  if (!d.querySelector("#par-comm .xleg .xk.ov")) problems.push("the legend does not explain overlap");
  const rows = () => Array.from(grid().querySelectorAll(".xr:not(.xh)"));
  const chipsOf = (r, re) => Array.from(r.querySelectorAll(".xchip")).filter(c => re.test(c.dataset.xk));
  for (const r of rows()) if (r.querySelector(".xb") && !r.querySelector(".xchip.bg,.xchip.part,.xchip.fused")) problems.push("a band on a row with nothing overlapping");
  const halo = rows().find(r => chipsOf(r, /^x\|cp\|P2P\|fwd/).length);
  if (!halo || halo.querySelector(".xb")) problems.push("the blocking linear halo should not have a band");
  set("trn.cpType", "ulysses");
  for (const pass of ["fwd", "bwd"]) {
    const row = rows().find(r => chipsOf(r, new RegExp(`^x\\|cp\\|A2A\\|${pass}`)).length);
    if (!row) { problems.push(`no Ulysses row on ${pass}`); continue; }
    if (chipsOf(row, new RegExp(`^x\\|cp\\|A2A\\|${pass}`)).length !== 2) problems.push(`Ulysses ${pass}: both all-to-alls should sit on the attention row`);
    if (!row.querySelector(".xc0.xb .xw")) problems.push(`Ulysses ${pass}: the attention cell should name the overlap`);
    if (!row.querySelector(".xl.xbe .xchip.part")) problems.push(`Ulysses ${pass}: the band should end on the CP lane`);
  }
  // Defaults, with no URL state
  const u = await load(path.resolve(__dirname, "..", "index.html"), "#train/par");
  const st = u.w.eval('[S.trn.fw, S.hw, S.topo].join(" ")');
  if (st !== "torchtitan GB300 nvl64") problems.push(`defaults: ${st}`);
  for (const e of errs.concat(u.errs)) problems.push("page error: " + e);
  if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
  console.log("comm panel ok");
  process.exit(0);
})();
