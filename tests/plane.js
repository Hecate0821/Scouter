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
const vmHas = model => require("vm").runInContext("HGA_DATA.has(" + JSON.stringify(model) + ")", P.context);
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
      kv: bad(`kv ${model} ${hw} ${basis}`, S.kv(Object.assign({ basis, N: 8, sessions: 256, turns: 20 }, q))),
    };
    n += 7;
  }
  const slower = (a, b, what) => { if (!(a <= b * (1 + 1e-9))) problems.push(`${model} ${hw}: ideal ${what} ${a} > modeled ${b}`); };
  slower(r.ideal.inf.stepT, r.modeled.inf.stepT, "decode step");
  slower(r.ideal.pre.stepT, r.modeled.pre.stepT, "prefill step");
  slower(r.ideal.trn.stepT, r.modeled.trn.stepT, "train step");
  slower(r.ideal.roof.kernel.t, r.modeled.roof.kernel.t, "kernel");
  slower(r.ideal.srv.tpot, r.modeled.srv.tpot, "serve TPOT");
  slower(r.ideal.kv.ttft, r.modeled.kv.ttft, "kv TTFT per turn");
  slower(r.ideal.kv.ttftHit, r.modeled.kv.ttftHit, "kv TTFT on a hit");
  if (!(r.ideal.kv.hitRate >= r.modeled.kv.hitRate - 1e-9)) problems.push(`${model} ${hw}: ideal kv hit rate below modeled`);
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
// kv: hits never cost more than recomputing; hit rate falls with sessions and rises with a host tier; random routing over
// replicas cuts local hits; vLLM puts linear-attention states on blocks at least one state page long; sliding windows cap at W
{
  const base = { model: "qwen35", hw: "H200", N: 8 }, r = S.kv(base);
  for (const x of r.turns) if (!(x.hbm <= x.miss * (1 + 1e-9) && x.miss <= x.none * (1 + 1e-9))) problems.push(`kv: turn ${x.t} hit ${x.hbm} / miss ${x.miss} / none ${x.none} out of order`);
  for (let i = 1; i < r.sweep.length; i++) if (r.sweep[i].hitRate > r.sweep[i - 1].hitRate + 1e-12) problems.push("kv: hit rate rose with more sessions");
  const busy = Object.assign({ sessions: 4096 }, base), hbm = S.kv(Object.assign({ hostGB: 0 }, busy)), host = S.kv(Object.assign({ hostGB: 400 }, busy));
  if (!(host.hitRate > hbm.hitRate)) problems.push("kv: a host tier should raise the hit rate once HBM is full");
  const rnd = S.kv(Object.assign({ replicas: 4, routing: "random" }, base));
  if (Math.abs(rnd.frac.hbm + rnd.frac.host - (r.frac.hbm + r.frac.host) / 4) > 1e-9) problems.push("kv: random routing over 4 replicas should keep 1/4 of local hits");
  const v = S.kv({ model: "qwen35", hw: "H200", N: 1, engine: "vllm" }), per = v.perTok;
  if (v.page % 16 || v.page * per.kvTokRank < per.snap / v.layers.lin || v.gran !== v.page) problems.push(`kv: vLLM hybrid block ${v.page} should hold one state page`);
  if (S.kv({ model: "qwen35", hw: "H200", N: 1, engine: "tokenspeed" }).gran !== 1) problems.push("kv: TokenSpeed tails should hit at the exact turn end");
  const sw = S.kv({ model: "musespark", hw: "B300", N: 8 }), at = L => sw.space.find(x => x.L === L);
  if (sw.perTok.W <= 0 || at(1048576).swa !== at(65536).swa) problems.push("kv: sliding-window bytes should stop growing past the window");
  n += 9;
}
const cli = args => JSON.parse(execFileSync(process.execPath, [path.join(ROOT, "tools/scouter.js"), ...args], { encoding: "utf8" }));
// trace: common public kernel names land in the right category, and a trace built from the engine's own step compares at 1.0
const NAMES = { comm: ["ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "cross_device_reduce_1stage", "deep_ep::internode_ll::dispatch"],
  moe: ["fused_moe_kernel", "cutlass_moe_fp4_grouped_gemm", "moe_align_block_size_kernel"], lin: ["fused_recurrent_gated_delta_rule_fwd_kernel", "causal_conv1d_update_kernel", "chunk_gated_delta_rule_fwd_kernel_h"],
  attn: ["flash_fwd_splitkv_kernel", "fmha_v2_flash_attention_fp16", "flashmla::sparse_decode", "BatchPrefillWithPagedKVCacheKernel"],
  gemm: ["nvjet_tst_128x256_64x4_2x1_v_bz_coopB_TNN", "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n", "deep_gemm::fp8_gemm_kernel", "cutlass_80_tensorop_bf16_s16816gemm"],
  samp: ["sampling_from_probs_kernel", "softmax_warp_forward"], elem: ["rms_norm_kernel", "rotary_embedding_kernel", "act_and_mul_kernel", "vectorized_elementwise_kernel"] };
// names that used to land in the wrong bucket: "combine" / "dispatch" / "gemm" inside attention, mHC and MoE kernel names, AMD and trtllm-gen names
const NAMES2 = { attn: ["flash_fwd_splitkv_combine_kernel", "_fwd_grouped_kernel_stage1", "_fwd_kernel_stage2", "_sparse_decode_gathered_kernel"],
  elem: ["mhc_post_combine_norm_prefill_kernel", "mhc_pre_gemm_sqrsum_kernel", "void alloc_decode_kernel<int>"], comm: ["rcclGenericKernel", "all_reduce_mhc_kernel"],
  moe: ["bmm_E4m3_E2m1E2m1_Fp32_t128x8x512", "finalizeKernel", "activationDeepSeek"], lin: ["recompute_w_u_fwd_kernel"], gemm: ["Cijk_Ailk_Bljk_BBS_BH_MT128x128x64"] };
for (const [c, ns] of Object.entries(NAMES2)) for (const nm of ns) { const got = P.Trace.catOf(nm); if (got !== c) problems.push(`trace: "${nm}" is ${got}, expected ${c}`); }
for (const [c, ns] of Object.entries(NAMES)) for (const nm of ns) { const got = P.Trace.catOf(nm); if (got !== c) problems.push(`trace: "${nm}" is ${got}, expected ${c}`); }
{
  const q = { model: "glm53f", hw: "B200", N: 8, B: 32 }, r = S.infer(q), steps = 10, NAME = Object.fromEntries(Object.entries(NAMES).map(([c, ns]) => [c, ns[0]]));
  const evs = r.ops.map(o => ({ ph: "X", cat: "kernel", name: NAME[P.Trace.opCat(o.key)], dur: o.t * 1e6 * steps }))
    .concat(r.comm.map(x => ({ ph: "X", cat: "kernel", name: NAME.comm, dur: x.t * 1e6 * steps })), [{ ph: "X", cat: "cpu_op", name: "aten::mm", dur: 1e6 }]);
  const tr = bad("trace", S.trace(Object.assign({ trace: { traceEvents: evs }, steps }, q)));
  for (const row of tr.rows) if (Math.abs(row.ratio - 1) > 1e-9) problems.push(`trace round trip ${row.cat}: ratio ${row.ratio}`);
  const list = bad("trace list", S.trace(Object.assign({ trace: [{ name: "fused_moe_kernel", us: 500, calls: 4 }, { name: "mystery_kernel", us: 100 }], steps: 2 }, q)));
  if (list.unmatched[0].name !== "mystery_kernel" || Math.abs(list.rows.find(x => x.cat === "moe").meas - 250e-6) > 1e-12) problems.push("trace: list input not summed per step");
  const fs = require("fs"), os = require("os"), f = require("path").join(os.tmpdir(), "scouter-trace-test.json");
  fs.writeFileSync(f, JSON.stringify({ traceEvents: evs }));
  const viaCli = cli(["trace", "--model", "glm53f", "--hw", "B200", "--N", "8", "--B", "32", "--steps", "10", "--file.trace", f, "--pick", "total.ratio"]);
  if (Math.abs(viaCli["total.ratio"] - 1) > 1e-9) problems.push("cli trace: --file.trace did not round-trip");
  fs.unlinkSync(f); n += 3;
  // two merged ranks (combine_traces' args.source_idx), PDL overlap on one stream, and a slow rank's all-reduce: averaged per rank,
  // the overlap goes to the earlier kernel, and the comm floor is the fastest calls
  const ev = (src, tid, ts, dur, name) => ({ ph: "X", cat: "kernel", pid: 10000 * src, tid, ts, dur, name, args: { source_idx: src } });
  const mr = S.trace(Object.assign({ trace: { traceEvents: [ev(0, 7, 0, 10, "nvjet_a"), ev(0, 7, 8, 10, "nvjet_b"), ev(0, 8, 30, 5, "ncclDevKernel_AllReduce"),
    ev(1, 7, 0, 10, "nvjet_a"), ev(1, 7, 10, 10, "nvjet_b"), ev(1, 8, 30, 50, "ncclDevKernel_AllReduce")] }, steps: 1 }, q));
  const near = (a, b) => Math.abs(a - b) < 1e-12, gm = mr.rows.find(x => x.cat === "gemm").meas;
  if (mr.ranks !== 2 || !near(gm, 19e-6) || !near(mr.overlap, 1e-6) || !near(mr.comm.floor, 5e-6) || !near(mr.comm.wait, 22.5e-6) || !near(mr.gpu.busy, 46.5e-6) || !near(mr.gpu.wall, 57.5e-6))
    problems.push(`trace: ranks / PDL overlap / comm floor wrong (${JSON.stringify({ ranks: mr.ranks, gm, overlap: mr.overlap, comm: mr.comm, gpu: mr.gpu })})`);
  n++;
}
const sw = S.sweep({ run: "infer", model: "qwen35", over: { B: [1, 8, 64], phase: ["decode", "prefill"] }, pick: ["stepT", "tComp"] });
if (sw.rows.length !== 6 || sw.rows.some(x => !(x.stepT > 0))) problems.push("sweep: expected 6 rows with stepT");
for (const id of Object.keys(cat.sets)) { const p = S.points({ set: id }); if (p.points.length !== cat.sets[id].n) problems.push(`points ${id}: ${p.points.length} != ${cat.sets[id].n}`); n++; }
for (const id of Object.keys(P.Calib.SETS)) for (const basis of ["ideal", "modeled"]) { const c = S.calib({ set: id, basis }); if (!c.n || !(c.median > 0)) problems.push(`calib ${id} ${basis}: no points`); n++;
  // a lower bound never exceeds a measurement
  if (basis === "ideal") for (const r of c.rows) if (r.ratio > 1) problems.push(`calib ${id}: ideal lower bound above the measurement at ${r.p.id} (${r.ratio.toFixed(3)})`); }
// ideal is a strict lower bound: TP MoE reads only the top-k expert slices, no CUDA graph padding, and the step is at least
// the critical path and each resource's total (compute, HBM, comm); an all-reduce moves its message once
{ const r = S.infer({ model: "kimik3", hw: "B300", N: 8, B: 60, basis: "ideal" }), moe = r.ops.find(o => o.key === "moe"), k = P.Common.MODELS.kimik3.moe.k;
  const lk = P.Common.link(P.Common.HW.B300, P.Common.TOPO.hgx8, 8, 1), ar = P.Common.arT(1e6, 8, lk, P.Common.IDEAL);
  if (Math.abs(moe.touched - k) > 1e-9 || r.rowsG !== r.rowsR || !r.bound || r.stepT < Math.max(r.bound.path, r.bound.compute, r.bound.hbm, r.bound.comm) * (1 - 1e-12) || Math.abs(ar - 1e6 / lk.bw) > 1e-15)
    problems.push(`ideal lower bound: touched ${moe.touched} (k ${k}), rows ${r.rowsG}/${r.rowsR}, bound ${JSON.stringify(r.bound)}, AR ${ar}`); n++; }
for (const [f, q] of [["infer", { model: "nope" }], ["train", { hw: "X1" }], ["points", {}], ["calib", { set: "nope" }], ["sweep", { run: "nope" }], ["infer", { basis: "measured" }]]) {
  let threw = false; try { S[f](q); } catch (e) { threw = true; } if (!threw) problems.push(`${f}(${JSON.stringify(q)}) should throw`);
}

// --- HuggingArch (real models): every exported cell, on and off the grid; the waterfalls add up exactly ---
if (!(at("src/data/hga/index.js") > at("src/data/query.js") && at("src/data/hga/index.js") < at("src/plane/scouter.js"))) problems.push("index.html must load src/data/hga/index.js before src/plane/scouter.js");
const closes = (wf, what) => { const end = wf[wf.length - 1].t, sum = wf[0].t + wf.slice(1, -1).reduce((x, s) => x + s.d, 0);
  if (!(Math.abs(sum - end) <= 1e-9 * Math.max(1, end))) problems.push(`${what}: waterfall does not close (${sum} vs ${end})`); };
let nh = 0;
// HuggingArch under the ideal: allowed where HuggingArch counts less for some operator class (docs/hga.md, "When HuggingArch comes
// in under the ideal"), but far under means the two sides describe different deployments or one side has a bug: fail below FLOOR
const FLOOR = 0.85, under = [];
const ratio = (what, h, i, roles) => { const r = h / i; if (!(r < 1)) return;
  const by = (roles || []).map(x => [x.role, (x.ideal.t ?? x.ideal) - (x.hga.t ?? x.hga)]).sort((a, b) => b[1] - a[1])[0];
  under.push({ what, r, by: by ? by[0] : "" }); };
for (const model of Object.keys(P.Hga.REAL)) {
  if (!vmHas(model)) { problems.push(`no HuggingArch export for ${model} (tools/hga/export.py --model ${model})`); continue; }
  const cells = S.hgaCells({ model });
  for (const [key, cards] of Object.entries(cells.infer)) for (const [hw, ok] of Object.entries(cards)) {
    if (ok !== true) continue;
    const [engine, sig] = key.split(".");
    for (const [isl, b, exact] of [[8192, 64, true], [6000, 48, false], [1024, 1, true]]) {
      const r = bad(`hga ${model} ${key} ${hw} ${isl}/${b}`, S.hga({ model, engine, sig, hw, isl, b })); nh++;
      if (!r.ok) { if (r.reason !== "capacity") problems.push(`hga ${model} ${key} ${hw} ${isl}/${b}: ${r.reason}`); continue; }
      if (r.exact !== exact) problems.push(`hga ${model} ${key} ${hw} ${isl}/${b}: exact should be ${exact}`);
      closes(r.decode.waterfall, `hga ${model} ${key} ${hw} decode`); closes(r.prefill.waterfall, `hga ${model} ${key} ${hw} prefill`);
      if (!(r.decode.hga.t > 0 && r.decode.ideal.t > 0 && r.prefill.hga.t > 0)) problems.push(`hga ${model} ${key} ${hw}: a phase has no time`);
      for (const ph of ["decode", "prefill"]) ratio(`hga ${model} ${key} ${hw} ${isl}/${b} ${ph}`, r[ph].hga.t, r[ph].ideal.t, r[ph].roles);
    }
  }
  for (const [key, cards] of Object.entries(cells.train)) for (const [hw, ok] of Object.entries(cards)) {
    if (ok !== true) continue;
    const [engine, sig] = key.split(".");
    for (const [seq, mbs] of [[8192, 1], [12000, 2]]) {
      const r = bad(`hgaTrain ${model} ${key} ${hw} ${seq}/${mbs}`, S.hgaTrain({ model, engine, sig, hw, seq, mbs })); nh++;
      if (!r.ok) { problems.push(`hgaTrain ${model} ${key} ${hw}: ${r.reason}`); continue; }
      closes(r.waterfall, `hgaTrain ${model} ${key} ${hw} ${seq}/${mbs}`);
      if (!(r.hga.t > 0 && r.ideal.t > 0 && r.hga.mfu > 0 && r.hga.mfu < 1)) problems.push(`hgaTrain ${model} ${key} ${hw}: step ${r.hga.t}, MFU ${r.hga.mfu}`);
      // the operator classes add up to each side's micro-batch (HuggingArch's to its three-digit export)
      for (const [side, tol] of [["ideal", 1e-9], ["hga", 0.02]]) { const sum = r.roles.reduce((a, x) => a + x[side].t, 0), want = r[side].micro;
        if (Math.abs(sum - want) > tol * want) problems.push(`hgaTrain ${model} ${key} ${hw} ${seq}/${mbs}: ${side} classes add up to ${sum}, micro-batch ${want}`); }
      ratio(`hgaTrain ${model} ${key} ${hw} ${seq}/${mbs}`, r.hga.t, r.ideal.t, r.roles);
    }
  }
  for (const k of ["sglang", "vllm", "megatron", "torchtitan"]) if (P.Hga.sigsFor(k, model).length < 3) problems.push(`${model}: fewer than 3 ${k} signatures`);
}
for (const k of ["sglang", "vllm", "megatron", "torchtitan"]) if (P.Hga.SIG[k].length !== 5) problems.push(`${k} must have 5 signatures`);
for (const [f, q] of [["hga", { model: "musespark" }], ["hga", { model: "qwen35", hw: "RTXP6000" }], ["hgaTrain", { model: "qwen35", sig: "nope" }]]) {
  let threw = false; try { S[f](q); } catch (e) { threw = true; } if (!threw) problems.push(`${f}(${JSON.stringify(q)}) should throw`);
}
const ha = cli(["hga", "--model", "qwen35", "--engine", "sglang", "--sig", "tp8", "--hw", "H200", "--isl", "6000", "--b", "48", "--pick", "decode.waterfall,throughput"]);
const hb = S.hga({ model: "qwen35", engine: "sglang", sig: "tp8", hw: "H200", isl: 6000, b: 48, pick: ["decode.waterfall", "throughput"] });
if (JSON.stringify(ha) !== JSON.stringify(hb)) problems.push("cli hga differs from the API");
console.log(`hga points: ${nh}`);
under.sort((a, b) => a.r - b.r);
const byClass = under.reduce((o, x) => (o[x.by] = (o[x.by] || 0) + 1, o), {});
console.log(`hga under the ideal: ${under.length} phases, by the class HuggingArch counts less for: ${JSON.stringify(byClass)}` +
  (under.length ? `; lowest ${under[0].r.toFixed(3)} at ${under[0].what}` : ""));
if (process.env.HGA_UNDER) for (const x of under) console.log(`  ${x.r.toFixed(3)} ${x.by} ${x.what}`);
for (const x of under) if (x.r < FLOOR) problems.push(`${x.what}: HuggingArch is ${x.r.toFixed(3)} of the ideal (most in ${x.by}); check that the signature describes one deployment on both sides (docs/hga.md)`);

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
