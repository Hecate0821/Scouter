// The engine runs on its own. Load src/engine/ in a bare context (no window, document, t(), I18N or data store) and drive
// every model × GPU × topology through inference (every engine, decode / prefill, speculative decoding), training (both
// frameworks, several layouts, every precision recipe and EP backend), the architecture explorer (including Arch.explain for
// every module), the quantization helpers, and every calibration set against its measured points in data/db.json.
// Checks:
//   - nothing throws, every result is JSON-serializable, no number is NaN; every calibration point gets a positive prediction;
//   - the output carries no language: text comes back as message keys (engine_*); a plain string has no CJK, and a plain string with
//     a space, or any plain-string name, must be one of the names in the engine's tables (models, GPUs, engines, kernels, formats…);
//   - static: engine files never call t() and never mention I18N, document, window, localStorage, navigator, location or DB;
//   - index.html loads the engine files in the same order as src/engine/node.js.
// Usage: node engine.js
const fs = require("fs");
const path = require("path");
const acorn = require("acorn");
const load = require("../src/engine/node.js");

const ROOT = path.resolve(__dirname, "..");
const CJK = /[　-〿㐀-鿿＀-￯]/;
const problems = [];
const bad = (where, what) => { if (problems.length < 200) problems.push(where + ": " + what); else problems.length++; };

// --- static ---
const FORBIDDEN = new Set(["I18N", "document", "window", "localStorage", "navigator", "location", "DB"]);
const walk = (n, f, parent) => { if (!n || typeof n.type !== "string") return; f(n, parent);
  for (const k in n) { const v = n[k]; if (Array.isArray(v)) v.forEach(c => walk(c, f, n)); else if (v && typeof v.type === "string") walk(v, f, n); } };
for (const f of load.FILES) {
  const src = fs.readFileSync(path.join(ROOT, "src/engine", f), "utf8");
  walk(acorn.parse(src, { ecmaVersion: "latest", locations: true }), (n, p) => {
    if (n.type === "CallExpression" && n.callee.type === "Identifier" && n.callee.name === "t") bad(`src/engine/${f}:${n.loc.start.line}`, "calls t()");
    if (n.type !== "Identifier" || !FORBIDDEN.has(n.name)) return;
    if (p && p.type === "MemberExpression" && p.property === n && !p.computed) return;      // x.window
    if (p && p.type === "Property" && p.key === n && !p.computed && !p.shorthand) return;   // { window: … }
    bad(`src/engine/${f}:${n.loc.start.line}`, "refers to " + n.name);
  });
}
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const order = [...html.matchAll(/<script src="src\/engine\/([^"]+)"><\/script>/g)].map(m => m[1]);
if (order.join() !== load.FILES.join()) bad("index.html", `engine scripts ${order.join(", ")} ≠ node.js ${load.FILES.join(", ")}`);

// --- runtime ---
const E = load();
const { Msg, MODELS, HW, TOPO, G0, Infer, Train, Arch, Quant, Par } = E;
let runs = 0;
const TABLE = new Set();   // names that are the same in every language
const names = v => { if (typeof v === "string") TABLE.add(v); else if (v && typeof v === "object" && !(v instanceof Msg)) Object.values(v).forEach(names); };
names([MODELS, HW, TOPO, Infer.ENGINES, Infer.SPEC, Train.PREC, Par.EP, Par.CP, Quant.SCHEMES, Quant.RECIPES]);
function check(where, v, seen = new Set()) {
  if (typeof v === "number") { if (Number.isNaN(v)) bad(where, "NaN"); return; }
  if (typeof v === "string") {
    if (CJK.test(v)) bad(where, "language text: " + v.slice(0, 60));
    else if ((/\s/.test(v) || /\.name$/.test(where)) && !TABLE.has(v)) bad(where, "text outside a message: " + v.slice(0, 60));
    return; }
  if (!v || typeof v !== "object" || seen.has(v)) return;
  seen.add(v);
  if (v instanceof Msg) { if (!/^engine_[a-z]+\.[A-Za-z0-9_.]+$/.test(v.key)) bad(where, "message key " + v.key); if (v.p) check(where, v.p, seen); return; }
  for (const k of Object.keys(v)) check(where + "." + k, v[k], seen);
}
function run(where, f) {
  runs++;
  let r;
  try { r = f(); } catch (e) { bad(where, "threw " + (e && e.stack ? e.stack.split("\n").slice(0, 3).join(" | ") : e)); return; }
  try { JSON.stringify(r); } catch (e) { bad(where, "not serializable: " + e.message); }
  check(where, r);
}
check("Common.MODELS", MODELS); check("Common.HW", HW); check("Common.TOPO", TOPO);
check("Infer.ENGINES", Infer.ENGINES); check("Train.PREC", Train.PREC); check("Par.EP", Par.EP); check("Quant.SCHEMES", Quant.SCHEMES);

const g = Object.assign({}, G0);
for (const mk of Object.keys(MODELS)) {
  const m = MODELS[mk];
  for (const hk of Object.keys(HW)) {
    const hw = HW[hk];
    for (const tk of Object.keys(TOPO)) {
      const topo = TOPO[tk], G = topo.gpus;
      for (const engine of Object.keys(Infer.ENGINES)) {
        const P = Infer.ENGINES[engine];
        for (const phase of ["decode", "prefill"]) for (const [N, dpA] of G > 8 ? [[8, 1], [64, 8]] : [[1, 1], [8, 1], [8, 8]]) {
          const base = Object.assign({}, Infer.BASE, { engine, phase, N, dpA, moePar: P.defaultMoe });
          run(`infer ${mk} ${hk} ${tk} ${engine} ${phase} N${N} dp${dpA}`, () => Infer.step(m, hw, topo, base, g));
          if (phase === "decode") for (const method of Object.keys(Infer.SPEC))
            run(`infer ${mk} ${hk} ${tk} ${engine} spec ${method}`, () => Infer.step(m, hw, topo, Object.assign({}, base, { specK: 3, specMethod: method, accept: 2 }), g));
        }
      }
      for (const fw of ["megatron", "torchtitan"]) {
        const layouts = G > 8 ? [{ tp: 1, cp: 1, ep: 8 }, { tp: 2, cp: 4, ep: 16 }, { tp: 1, cp: 1, ep: 64 }] : [{ tp: 1, cp: 1, ep: 8 }, { tp: 2, cp: 2, ep: 4 }, { tp: 1, cp: 8, ep: 8 }];
        for (const lay of layouts) {
          const c = Object.assign({}, Train.BASE, Train.FW_DEF[fw], { fw, N: G }, lay);
          run(`train ${mk} ${hk} ${tk} ${fw} ${JSON.stringify(lay)}`, () => Train.run(m, hw, topo, c, g));
        }
        const c0 = Object.assign({}, Train.BASE, Train.FW_DEF[fw], { fw, N: G });
        for (const prec of Object.keys(Train.PREC)) run(`train ${mk} ${hk} ${tk} ${fw} prec ${prec}`, () => Train.run(m, hw, topo, Object.assign({}, c0, { prec }), g));
        if (fw === "torchtitan") for (const d of Object.keys(Par.EP)) run(`train ${mk} ${hk} ${tk} tt ep ${d}`, () => Train.run(m, hw, topo, Object.assign({}, c0, { dispatcher: d }), g));
      }
    }
    for (const phase of ["decode", "prefill", "train"]) {
      const wl = Object.assign(Arch.defaults(m), { phase, maxChunks: 4 });
      run(`arch ${mk} ${hk} ${phase}`, () => ({ m: Arch.measure(m, hw, g, wl), f: Arch.facts(m, wl), p: Arch.peaks(hw, wl), present: Arch.present(m),
        x: Arch.present(m).map(p => Arch.explain(m, hw, g, wl, p.id)) }));
    }
    if (m.moe.E > 0) for (const k of Quant.ORDER) run(`quant moe ${mk} ${hk} ${k}`, () => Quant.moeLayer(hw, g, m, k, 512, 8, { fuse: "fused", align: "auto", BM: 128, deqRate: 40 }));
  }
}
for (const hk of Object.keys(HW)) {
  const hw = HW[hk], o = { fuse: "fused", align: 1, floor: false, BM: 128, deqRate: 40 };
  for (const k of Quant.ORDER) for (const M of [1, 64, 4096]) run(`quant gemm ${hk} ${k} M${M}`, () => Quant.gemm(hw, g, k, M, 1024, 2048, o));
  for (const k of Quant.ORDER) run(`quant crossover ${hk} ${k}`, () => Quant.crossover(hw, g, k, 1024, 2048, o));
  for (const k of Quant.RORDER) run(`quant train ${hk} ${k}`, () => Quant.trainGemm(hw, g, k, 8192, 4096, 4096, { nMicro: 4 }));
}

// --- calibration: each set's prediction for every measured point it claims to reproduce ---
const DBJ = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "db.json"), "utf8"));
const where = (p, f) => Object.keys(f).every(k => Array.isArray(f[k]) ? f[k].includes(p[k]) : p[k] === f[k]);
const calib = [];
for (const [id, K] of Object.entries(E.Calib.SETS)) {
  const pts = DBJ.points.filter(p => p.set === (K.set || id) && (!K.where || where(p, K.where)));
  if (!pts.length) bad(`Calib.SETS.${id}`, "matches no data point");
  const ratios = [];
  for (const p of pts) run(`calib ${p.id}`, () => {
    const pred = K.predict(p, g), meas = K.meas(p);
    if (!(pred > 0 && Number.isFinite(pred))) bad(`calib ${p.id}`, "prediction " + pred);
    if (!(meas > 0 && Number.isFinite(meas))) bad(`calib ${p.id}`, "measurement " + meas);
    ratios.push(pred / meas); return { pred, meas };
  });
  ratios.sort((a, b) => a - b);
  const mid = (ratios.length - 1) / 2, md = ratios.length ? (ratios[Math.floor(mid)] + ratios[Math.ceil(mid)]) / 2 : 0;
  calib.push(`${id} ${pts.length} pts, median pred/meas ${md.toFixed(2)}`);   // the same median as the calibration page
}
for (const p of DBJ.points.filter(p => p.replay && p.replay.kind === "mok_ratio"))
  run(`calib ${p.id}`, () => { const r = E.Calib.mokRatio(p.replay, g); if (!(r > 0 && Number.isFinite(r))) bad(`calib ${p.id}`, "ratio " + r); return r; });

console.log("calibration: " + calib.join("; "));
console.log(`engine runs: ${runs}`);
if (problems.length) { console.log(problems.slice(0, 200).join("\n")); console.log(problems.length + " engine problems"); process.exit(1); }
console.log("engine ok: runs headless, no language in its output");
