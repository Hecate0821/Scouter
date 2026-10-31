// The data plane (src/plane/scouter.js over src/engine and src/data) on its own in Node, the way an agent uses it.
// Static: the plane never touches the page (DOM, t(), UI state) and src/ui never reaches into the plane's internals; the
// plane script sits between the data store and the UI in index.html.
// Runs: every model × GPU × basis through roofline / infer / train / module, a sweep, every data set through points and every
// calibration set through calib. Each result must be JSON-serializable without NaN, and the ideal basis (hardware roofline)
// must never be slower than the modeled one. Last, tools/scouter.js must print the same numbers as the API.
// Usage: node plane.js
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const load = require("../src/plane/node.js");

const ROOT = path.resolve(__dirname, "..");
const problems = [];

// --- static ---
const plane = fs.readFileSync(path.join(ROOT, "src/plane/scouter.js"), "utf8").replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, "");
for (const [re, what] of [[/\b(window|document|localStorage|location|navigator)\b/, "the browser"], [/\bt\(|I18N\b/, "the language layer"], [/\bS\.[a-z]|\$\(|\bqa\(/, "UI state or DOM helpers"]])
  if (re.test(plane)) problems.push(`src/plane/scouter.js touches ${what}`);
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const at = f => html.indexOf(`<script src="${f}"></script>`);
if (!(at("src/data/query.js") < at("src/plane/scouter.js") && at("src/plane/scouter.js") < at("src/ui/base.js"))) problems.push("index.html must load src/plane/scouter.js after the data store and before src/ui");

// --- runs ---
const P = load(), S = P.Scouter;
const nan = (o, seen = new Set()) => typeof o === "number" ? Number.isNaN(o) : o && typeof o === "object" && !seen.has(o) && (seen.add(o), Object.values(o).some(v => nan(v, seen)));
const bad = (what, r) => { if (nan(r)) problems.push(`${what}: NaN in the result`); JSON.stringify(r); return r; };
const cat = S.catalog();
for (const k of ["models", "hw", "topo", "sources", "runs", "defaults", "sets"]) if (!cat[k]) problems.push(`catalog has no ${k}`);
if (JSON.stringify(cat.sources) !== JSON.stringify(["ideal", "modeled", "measured"])) problems.push("sources must be ideal, modeled, measured");
if (Object.keys(P.Common.IDEAL).join() !== Object.keys(P.Common.G0).join()) problems.push("Common.IDEAL must have the same keys as Common.G0");
let n = 0;
for (const model of Object.keys(cat.models)) for (const hw of Object.keys(cat.hw)) {
  const topo = hw.startsWith("GB") ? "nvl64" : "hgx8", q = { model, hw, topo };
  const r = {};
  for (const basis of ["ideal", "modeled"]) {
    r[basis] = {
      roof: bad(`roofline ${hw} ${basis}`, S.roofline({ hw, basis, flops: 1e12, bytes: 1e9 })),
      inf: bad(`infer ${model} ${hw} ${basis}`, S.infer(Object.assign({ basis }, q))),
      pre: bad(`prefill ${model} ${hw} ${basis}`, S.infer(Object.assign({ basis, phase: "prefill" }, q))),
      trn: bad(`train ${model} ${hw} ${basis}`, S.train(Object.assign({ basis }, q))),
      mod: bad(`module ${model} ${hw} ${basis}`, S.module(Object.assign({ basis }, q))),
    };
    n += 5;
  }
  const slower = (a, b, what) => { if (!(a <= b * (1 + 1e-9))) problems.push(`${model} ${hw}: ideal ${what} ${a} > modeled ${b}`); };
  slower(r.ideal.inf.stepT, r.modeled.inf.stepT, "decode step");
  slower(r.ideal.pre.stepT, r.modeled.pre.stepT, "prefill step");
  slower(r.ideal.trn.stepT, r.modeled.trn.stepT, "train step");
  slower(r.ideal.roof.kernel.t, r.modeled.roof.kernel.t, "kernel");
  for (const k of Object.keys(r.ideal.mod.measure.mods)) slower(r.ideal.mod.measure.mods[k].t, r.modeled.mod.measure.mods[k].t, "module " + k);
}
const sw = S.sweep({ run: "infer", model: "qwen35", over: { B: [1, 8, 64], phase: ["decode", "prefill"] }, pick: ["stepT", "tComp"] });
if (sw.rows.length !== 6 || sw.rows.some(x => !(x.stepT > 0))) problems.push("sweep: expected 6 rows with stepT");
for (const id of Object.keys(cat.sets)) { const p = S.points({ set: id }); if (p.points.length !== cat.sets[id].n) problems.push(`points ${id}: ${p.points.length} != ${cat.sets[id].n}`); n++; }
for (const id of Object.keys(P.Calib.SETS)) for (const basis of ["ideal", "modeled"]) { const c = S.calib({ set: id, basis }); if (!c.n || !(c.median > 0)) problems.push(`calib ${id} ${basis}: no points`); n++; }
for (const [f, q] of [["infer", { model: "nope" }], ["train", { hw: "X1" }], ["points", {}], ["calib", { set: "nope" }], ["sweep", { run: "nope" }], ["infer", { basis: "measured" }]]) {
  let threw = false; try { S[f](q); } catch (e) { threw = true; } if (!threw) problems.push(`${f}(${JSON.stringify(q)}) should throw`);
}

// --- the command line prints what the API returns ---
const cli = args => JSON.parse(execFileSync(process.execPath, [path.join(ROOT, "tools/scouter.js"), ...args], { encoding: "utf8" }));
const a = cli(["infer", "--model", "kimik3", "--hw", "B200", "--basis", "ideal", "--B", "128", "--pick", "stepT,tComp"]);
const b = S.infer({ model: "kimik3", hw: "B200", basis: "ideal", B: 128, pick: ["stepT", "tComp"] });
if (JSON.stringify(a) !== JSON.stringify(b)) problems.push(`cli infer differs: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
const c = cli(["sweep", "--run", "train", "--model", "qwen35", "--over.ep", "4,8", "--pick", "stepT"]);
if (c.rows.length !== 2) problems.push("cli sweep: expected 2 rows");
if (!cli(["catalog"]).models.qwen35) problems.push("cli catalog has no qwen35");

console.log(`plane runs: ${n}`);
if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
console.log("plane ok: headless, ideal never slower than modeled, CLI matches the API");
