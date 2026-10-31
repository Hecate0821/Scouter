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
      srv: bad(`serve ${model} ${hw} ${basis}`, S.serve(Object.assign({ basis, N: 8, conc: 32, isl: 2048, osl: 512 }, q))),
    };
    n += 6;
  }
  const slower = (a, b, what) => { if (!(a <= b * (1 + 1e-9))) problems.push(`${model} ${hw}: ideal ${what} ${a} > modeled ${b}`); };
  slower(r.ideal.inf.stepT, r.modeled.inf.stepT, "decode step");
  slower(r.ideal.pre.stepT, r.modeled.pre.stepT, "prefill step");
  slower(r.ideal.trn.stepT, r.modeled.trn.stepT, "train step");
  slower(r.ideal.roof.kernel.t, r.modeled.roof.kernel.t, "kernel");
  slower(r.ideal.srv.tpot, r.modeled.srv.tpot, "serve TPOT");
  for (const k of Object.keys(r.ideal.mod.measure.mods)) slower(r.ideal.mod.measure.mods[k].t, r.modeled.mod.measure.mods[k].t, "module " + k);
}
// the per-kernel efficiency table: every cell points at a data point whose value / peak is that η, and every public row marked
// keff is used by exactly one cell
const KEFF = P.Common.KEFF, ALL = Object.keys(P.DB.sets).flatMap(id => P.DB.set(id)), pts = Object.fromEntries(ALL.map(p => [p.id, p]));
for (const r of KEFF) {
  const p = pts[r.ref];
  if (!p) { problems.push(`KEFF ${r.cls} ${r.dt} sm${r.sm}: no data point ${r.ref}`); continue; }
  const eta = p.set === "kernel_public" ? p.value / p.peak : p.tflops / P.Common.HW.H200.bf16;
  if (Math.abs(eta - r.eta) > 0.006) problems.push(`KEFF ${r.ref}: table says ${r.eta}, data gives ${eta.toFixed(3)}`);
  if (p.set === "kernel_public" && (!p.keff || p.cls !== r.cls || p.kind !== r.kind || p.dtype !== r.dt || p.sm !== r.sm)) problems.push(`KEFF ${r.ref}: class / kind / dtype / sm / keff disagree with the data point`);
}
for (const p of ALL) if (p.set === "kernel_public" && p.keff && KEFF.filter(r => r.ref === p.id).length !== 1) problems.push(`kernel_public ${p.id}: keff but used by ${KEFF.filter(r => r.ref === p.id).length} table cells`);
{ // the table changes modeled numbers only: ideal ignores it, and kEff 0 gives back one η_c
  const q = { model: "kimik3", hw: "B300", N: 8, phase: "prefill" }, on = S.infer(q), off = S.infer(Object.assign({ g: { kEff: 0 } }, q));
  const a = on.ops.find(o => o.key === "acore"), b = off.ops.find(o => o.key === "acore");
  if (!(Math.abs(a.etaC - 0.74) < 1e-9 && b.etaC === P.Common.G0.etaC)) problems.push(`kEff: attention η on B300 should be 0.74 with the table and ${P.Common.G0.etaC} without`);
  if (S.infer(Object.assign({ basis: "ideal" }, q)).ops.some(o => o.etaC !== 1)) problems.push("kEff: the ideal basis must keep η = 1");
}
// ncu import → SOL efficiencies: a synthetic Nsight Compute raw CSV (tests/fixtures) through tools/import_ncu.py, then Scouter.kernels
{
  const fs = require("fs"), os = require("os"), out = path.join(os.tmpdir(), "scouter-ncu-test.json"), fl = path.join(os.tmpdir(), "scouter-ncu-flops.json");
  fs.writeFileSync(fl, JSON.stringify({ "0": 1.4e11 }));
  execFileSync("uv", ["run", "--no-project", path.join(ROOT, "tools/import_ncu.py"), path.join(__dirname, "fixtures/ncu_raw_synthetic.csv"), "--name", "synth", "--hw", "H200",
    "--conditions", "synthetic", "--flops", fl, "--out", out], { encoding: "utf8" });
  const set = JSON.parse(fs.readFileSync(out, "utf8")), k = S.kernels({ points: set.points, hw: "H200" });
  const [g, c] = k.points;
  if (set.profile.tool !== "Nsight Compute" || set.points.length !== 2) problems.push("import_ncu: expected 2 kernels and a profile");
  if (Math.abs(g.etaC - 1.4e11 / 200e-6 / 989e12) > 1e-9 || Math.abs(c.etaM - 2e9 / 500e-6 / 4800e9) > 1e-9 || c.etaC !== null) problems.push(`import_ncu: SOL efficiencies ${g.etaC} / ${c.etaM}`);
  const h = S.kernels();
  if (!(h.measured.etaC > 0.7 && h.measured.etaC < 0.8 && h.measured.etaM > 0.85 && h.measured.etaM < 0.9)) problems.push("kernels: H200 best GEMM / copy efficiency out of range");
  fs.unlinkSync(out); fs.unlinkSync(fl); n += 2;
}
// serve: one request alone has no prefill interference; more concurrency never lowers TPOT
const one = S.serve({ model: "kimik3", hw: "B300", N: 8, conc: 1 }), many = S.serve({ model: "kimik3", hw: "B300", N: 8, conc: 64 });
if (one.interf !== 0 || !(many.tpot > one.tpot) || !(many.tokPerGpu > one.tokPerGpu)) problems.push("serve: concurrency should raise TPOT and throughput, and conc 1 has no interference");
const cli = args => JSON.parse(execFileSync(process.execPath, [path.join(ROOT, "tools/scouter.js"), ...args], { encoding: "utf8" }));
// trace: common public kernel names land in the right category, and a trace built from the engine's own step compares at 1.0
const NAMES = { comm: ["ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "cross_device_reduce_1stage", "deep_ep::internode_ll::dispatch"],
  moe: ["fused_moe_kernel", "cutlass_moe_fp4_grouped_gemm", "moe_align_block_size_kernel"], lin: ["fused_recurrent_gated_delta_rule_fwd_kernel", "causal_conv1d_update_kernel", "chunk_gated_delta_rule_fwd_kernel_h"],
  attn: ["flash_fwd_splitkv_kernel", "fmha_v2_flash_attention_fp16", "flashmla::sparse_decode", "BatchPrefillWithPagedKVCacheKernel"],
  gemm: ["nvjet_tst_128x256_64x4_2x1_v_bz_coopB_TNN", "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n", "deep_gemm::fp8_gemm_kernel", "cutlass_80_tensorop_bf16_s16816gemm"],
  samp: ["sampling_from_probs_kernel", "softmax_warp_forward"], elem: ["rms_norm_kernel", "rotary_embedding_kernel", "act_and_mul_kernel", "vectorized_elementwise_kernel"] };
for (const [c, ns] of Object.entries(NAMES)) for (const nm of ns) { const got = P.Trace.catOf(nm); if (got !== c) problems.push(`trace: "${nm}" is ${got}, expected ${c}`); }
{
  const q = { model: "glm53f", hw: "B200", N: 8, B: 32 }, r = S.infer(q), steps = 10, NAME = Object.fromEntries(Object.entries(NAMES).map(([c, ns]) => [c, ns[0]]));
  const evs = r.ops.filter(o => o.overlap !== "moe").map(o => ({ ph: "X", cat: "kernel", name: NAME[P.Trace.opCat(o.key)], dur: o.t * 1e6 * steps }))
    .concat(r.comm.filter(x => !x.overlap).map(x => ({ ph: "X", cat: "kernel", name: NAME.comm, dur: x.t * 1e6 * steps })), [{ ph: "X", cat: "cpu_op", name: "aten::mm", dur: 1e6 }]);
  const tr = bad("trace", S.trace(Object.assign({ trace: { traceEvents: evs }, steps }, q)));
  for (const row of tr.rows) if (Math.abs(row.ratio - 1) > 1e-9) problems.push(`trace round trip ${row.cat}: ratio ${row.ratio}`);
  const list = bad("trace list", S.trace(Object.assign({ trace: [{ name: "fused_moe_kernel", us: 500, calls: 4 }, { name: "mystery_kernel", us: 100 }], steps: 2 }, q)));
  if (list.unmatched[0].name !== "mystery_kernel" || Math.abs(list.rows.find(x => x.cat === "moe").meas - 250e-6) > 1e-12) problems.push("trace: list input not summed per step");
  const fs = require("fs"), os = require("os"), f = require("path").join(os.tmpdir(), "scouter-trace-test.json");
  fs.writeFileSync(f, JSON.stringify({ traceEvents: evs }));
  const viaCli = cli(["trace", "--model", "glm53f", "--hw", "B200", "--N", "8", "--B", "32", "--steps", "10", "--file.trace", f, "--pick", "total.ratio"]);
  if (Math.abs(viaCli["total.ratio"] - 1) > 1e-9) problems.push("cli trace: --file.trace did not round-trip");
  fs.unlinkSync(f); n += 3;
}
const sw = S.sweep({ run: "infer", model: "qwen35", over: { B: [1, 8, 64], phase: ["decode", "prefill"] }, pick: ["stepT", "tComp"] });
if (sw.rows.length !== 6 || sw.rows.some(x => !(x.stepT > 0))) problems.push("sweep: expected 6 rows with stepT");
for (const id of Object.keys(cat.sets)) { const p = S.points({ set: id }); if (p.points.length !== cat.sets[id].n) problems.push(`points ${id}: ${p.points.length} != ${cat.sets[id].n}`); n++; }
for (const id of Object.keys(P.Calib.SETS)) for (const basis of ["ideal", "modeled"]) { const c = S.calib({ set: id, basis }); if (!c.n || !(c.median > 0)) problems.push(`calib ${id} ${basis}: no points`); n++; }
for (const [f, q] of [["infer", { model: "nope" }], ["train", { hw: "X1" }], ["points", {}], ["calib", { set: "nope" }], ["sweep", { run: "nope" }], ["infer", { basis: "measured" }]]) {
  let threw = false; try { S[f](q); } catch (e) { threw = true; } if (!threw) problems.push(`${f}(${JSON.stringify(q)}) should throw`);
}

// --- the command line prints what the API returns ---
const a = cli(["infer", "--model", "kimik3", "--hw", "B200", "--basis", "ideal", "--B", "128", "--pick", "stepT,tComp"]);
const b = S.infer({ model: "kimik3", hw: "B200", basis: "ideal", B: 128, pick: ["stepT", "tComp"] });
if (JSON.stringify(a) !== JSON.stringify(b)) problems.push(`cli infer differs: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
const c = cli(["sweep", "--run", "train", "--model", "qwen35", "--over.ep", "4,8", "--pick", "stepT"]);
if (c.rows.length !== 2) problems.push("cli sweep: expected 2 rows");
if (!cli(["catalog"]).models.qwen35) problems.push("cli catalog has no qwen35");

console.log(`plane runs: ${n}`);
if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
console.log("plane ok: headless, ideal never slower than modeled, CLI matches the API");
