// The engine's training collectives against traces of the same layouts run with real PyTorch parallelism (sim/: FSDP2,
// DTensor TP, all-to-all EP on CPU gloo processes). The traces are checked in, so this test needs neither Python nor torch;
// regenerate them with `cd sim && uv run python -m scouter_sim.run` after changing a case.
// For each case: Train.run with the case's model and layout (torchtitan), then both sides become per-rank wire bytes for
// one micro-batch, keyed by (group, op, pass): AG / RS move (n−1)/n of the full tensor, AR 2(n−1)/n, A2A the rows sent to
// other ranks. Every key must issue the same number of calls and agree within 1% in bytes (FSDP2 pads each parameter's first
// dim to a multiple of the shard count; the engine does not). A key only one side has must stay under 1% of that group's
// bytes and is listed as not modeled.
// Also static: the app (index.html, src/) never refers to the simulator, and the simulator never reads the app; they meet
// only in sim/cases and sim/traces.
// Usage: node sim.js [-v]
const fs = require("fs");
const path = require("path");
const load = require("../src/engine/node.js");

const ROOT = path.resolve(__dirname, "..");
const SIM = path.join(ROOT, "sim");
const verbose = process.argv.includes("-v");
const problems = [];

// --- static: the two sides never import each other ---
const files = (dir, ext) => fs.readdirSync(dir, { withFileTypes: true }).flatMap(f => {
  const p = path.join(dir, f.name);
  return f.isDirectory() ? (f.name === ".venv" || f.name === "node_modules" ? [] : files(p, ext)) : ext.some(e => f.name.endsWith(e)) ? [p] : [];
});
for (const f of [path.join(ROOT, "index.html"), ...files(path.join(ROOT, "src"), [".js"])])
  if (/\bsim\/|scouter_sim/.test(fs.readFileSync(f, "utf8"))) problems.push(`${path.relative(ROOT, f)} refers to the simulator`);
for (const f of files(path.join(SIM, "scouter_sim"), [".py"]))
  if (/\.\.\/src|src\/engine|index\.html|locales\//.test(fs.readFileSync(f, "utf8"))) problems.push(`${path.relative(ROOT, f)} reads the app`);

// --- per case ---
const E = load();
const { Train, HW, TOPO, G0 } = E;
const pct = x => (100 * x).toFixed(2) + "%";
const fB = b => b >= 1e6 ? (b / 1e6).toFixed(3) + " MB" : b >= 1e3 ? (b / 1e3).toFixed(2) + " KB" : b.toFixed(0) + " B";
const wire = r => r.op === "AG" || r.op === "RS" ? r.numel * r.elem * (r.n - 1) / r.n
  : r.op === "AR" ? 2 * r.numel * r.elem * (r.n - 1) / r.n
  : r.send.reduce((s, x, i) => s + (i === r.me ? 0 : x), 0) * r.row * r.elem;
const keyOf = (dim, op, pass) => `${dim} ${op} ${pass}`;

const cases = fs.readdirSync(path.join(SIM, "cases")).filter(f => f.endsWith(".json")).sort();
if (!cases.length) problems.push("sim/cases has no case");
for (const file of cases) {
  const c = JSON.parse(fs.readFileSync(path.join(SIM, "cases", file), "utf8"));
  const tf = path.join(SIM, "traces", file);
  if (!fs.existsSync(tf)) { problems.push(`${c.name}: no trace (cd sim && uv run python -m scouter_sim.run cases/${file})`); continue; }
  const trace = JSON.parse(fs.readFileSync(tf, "utf8"));
  if (JSON.stringify(trace.layout) !== JSON.stringify(c.layout) || trace.tok !== c.tok) problems.push(`${c.name}: trace is stale (layout or tokens differ from the case)`);

  // engine: one micro-batch of exactly this layout
  const L = c.layout, tp = L.tp || 1, ep = L.ep || 1, dp = L.N / tp;
  const cfg = Object.assign({}, Train.BASE, { fw: "torchtitan", N: L.N, tp, cp: 1, pp: 1, vpp: 1, ep, etp: 1, dispatcher: "standard", cpType: "allgather",
    linCP: "state", recompute: "none", rcModules: [], tok: c.tok, S: c.tok, Gtok: c.tok * dp, imb: 1, offload: false, reshard: true, gradDt: "fp32", hsdp: 1, prec: "bf16" });
  const r = Train.run(c.model, HW.H200, TOPO.hgx8, cfg, Object.assign({}, G0));
  const bad = r.warn.filter(w => w.bad);
  if (bad.length) { problems.push(`${c.name}: the engine rejects the layout: ${bad.map(String).join("; ")}`); continue; }
  const eng = {};
  const add = (k, bytes, calls) => { const o = eng[k] || (eng[k] = { bytes: 0, calls: 0 }); o.bytes += bytes; o.calls += calls; };
  for (const x of r.par.fsdp ? r.par.fsdp.xs : []) add(keyOf(x.dim, x.op, x.pass), x.bytes, x.calls);
  for (const cm of r.comm) if ((cm.cls === "tp" || cm.cls === "ep") && cm.xs)
    for (const x of cm.xs) add(keyOf(cm.cls, x.op === "sync" ? "A2A·count" : x.op, x.pass), x.bytes * cm.layers, cm.layers);

  // trace: every rank should see the same bytes (balanced routing); compare rank 0, check the others match it
  const per = trace.ranks.map(recs => {
    const o = {};
    for (const x of recs) { const op = x.op === "A2A" && /int/.test(x.dtype) ? "A2A·count" : x.op, k = keyOf(x.dim, op, x.pass);
      const v = o[k] || (o[k] = { bytes: 0, calls: 0 }); v.bytes += wire(x); v.calls += 1; }
    return o;
  });
  const sim = per[0];
  per.forEach((o, i) => { if (JSON.stringify(Object.entries(o).map(([k, v]) => [k, v.bytes]).sort()) !== JSON.stringify(Object.entries(sim).map(([k, v]) => [k, v.bytes]).sort()))
    problems.push(`${c.name}: rank ${i} moved different bytes from rank 0`); });

  const dimTotal = {};
  for (const [k, v] of Object.entries(sim)) dimTotal[k.split(" ")[0]] = (dimTotal[k.split(" ")[0]] || 0) + v.bytes;
  const rows = [];
  for (const k of [...new Set([...Object.keys(eng), ...Object.keys(sim)])].sort()) {
    const e = eng[k] ? eng[k].bytes : 0, s = sim[k] ? sim[k].bytes : 0, dim = k.split(" ")[0];
    let verdict;
    if (e > 0 && s > 0) verdict = Math.abs(e / s - 1) > 0.01 ? "MISMATCH" : Math.abs(eng[k].calls - sim[k].calls) > 1e-9 ? "CALLS DIFFER" : "ok";
    else verdict = Math.max(e, s) <= 0.01 * (dimTotal[dim] || Infinity) ? (s > 0 ? "not modeled (small)" : "not issued (small)") : s > 0 ? "NOT MODELED" : "NOT ISSUED";
    if (/^[A-Z]/.test(verdict)) problems.push(`${c.name}: ${k}: engine ${fB(e)}, PyTorch ${fB(s)} → ${verdict.toLowerCase()}`);
    rows.push(`  ${k.padEnd(20)} engine ${fB(e).padStart(11)} ×${String(eng[k] ? +eng[k].calls.toFixed(2) : 0).padStart(3)}   PyTorch ${fB(s).padStart(11)} ×${String(sim[k] ? sim[k].calls : 0).padStart(3)}   ${s > 0 && e > 0 ? pct(e / s - 1).padStart(7) : "".padStart(7)}  ${verdict}`);
  }
  console.log(`${c.name} (${c.about}; torch ${trace.torch}, ${trace.backend}):`);
  if (verbose || rows.some(x => !/ ok$/.test(x))) console.log(rows.join("\n"));
  else console.log(`  ${rows.length} (group, op, pass) totals agree within 1%`);
}

if (problems.length) { console.log(problems.join("\n")); console.log(problems.length + " sim problems"); process.exit(1); }
console.log("sim ok: the engine's collectives match what PyTorch issues for every case");
