// Every config key must reach the simulator: a key the engine calls inert must change nothing,
// and every other key must change the result somewhere. Also checks the sweep-wide invariants of the parallel layouts.
const { Scouter } = require("../src/plane/node.js")();
const E = require("../src/engine/node.js")();
const { MODELS, Train, Infer, Common, HW, TOPO } = E;
const fails = [], fail = m => { if (fails.length < 40) fails.push(m); else fails.more = (fails.more || 0) + 1; };
const pow2 = N => { const a = []; for (let i = 1; i <= N; i *= 2) a.push(i); return a; };
const sigT = r => JSON.stringify([r.stepT, r.tokPerGpu, r.mem.used, r.comm.map(x => [x.key, x.bytes, x.t])]);
const sigI = r => JSON.stringify([r.stepT, r.tokPerGpu, r.ttft, r.mem.Bmax, r.mem.weights, r.comm.map(x => [x.key, x.bytes, x.t])]);
const flip = v => !v, quiet = {};

const TRN = { imb: [1.4], dispatcher: { megatron: ["deepep", "hybridep", "ncclep"], torchtitan: ["ag", "deepep", "distmoe", "mok"] }, cpType: { megatron: ["allgather", "a2a"], torchtitan: ["ring", "ulysses"] },
  linCP: { megatron: ["headwise"], torchtitan: ["ulysses"] }, cpLB: flip, epOverlap: flip, sharedOverlap: flip, tpOverlap: flip, hostSyncUs: [500], mokF: [0.5], cpuAdamGBs: [1], overlapOffload: flip,
  etaA: [0.1], vpp: [2], overlapGrad: flip, overlapParam: flip, gradDt: ["bf16"], reshard: flip, etp: [2], offload: flip, workspaceGB: [1] };
const INF = { imb: [1.5], redundant: [64], dispFp8: flip, tbo: flip, hostSyncUs: [500], dcp: [2], prefixCache: flip, specMethod: ["eagle3", "dflash"], accept: [1.3], biFactor: [5], bi: flip, implEff: [0.5] };

// ---- training: inert keys do nothing, the rest respond somewhere ----
const seenT = {}, g = { modeled: Common.G0, ideal: Common.IDEAL };
const layouts = [[1, 1, 1, 1], [1, 1, 1, 8], [2, 2, 1, 2], [1, 4, 1, 2], [1, 1, 2, 2], [2, 1, 2, 4], [1, 2, 2, 1]];
for (const mk of Object.keys(MODELS)) for (const fw of ["megatron", "torchtitan"]) for (const b of ["modeled", "ideal"]) for (const [hw, topo] of [["H200", "hgx8"]]) for (const [tp, cp, pp, ep] of layouts) for (const offload of [false, true]) {
  const m = MODELS[mk], c0 = Object.assign({}, Train.BASE, Train.FW_DEF[fw], { fw, N: TOPO[topo].gpus, tp, cp, pp, ep, offload, recompute: "full", S: 8192, tok: 8192, Gtok: 8192 * 32 });
  if (fw === "torchtitan" && (ep < tp || ep % tp)) continue;
  const inert = Train.inert(m, c0, g[b]), base = sigT(Train.run(m, HW[hw], TOPO[topo], c0, g[b]));
  for (const [k, alt] of Object.entries(TRN)) {
    if ((k === "etp" || k.startsWith("overlap") && k !== "overlapOffload" || k === "epOverlap" || k === "sharedOverlap") && fw !== "megatron") continue;
    if (["gradDt", "reshard", "cpLB", "mokF"].includes(k) && fw !== "torchtitan") continue;
    if (k === "vpp" && pp < 2) continue;
    const vals = typeof alt === "function" ? [alt(c0[k])] : Array.isArray(alt) ? alt : alt[fw];
    for (const v of vals) {
      const c = Object.assign({}, c0, { [k]: v }), r = Train.run(m, HW[hw], TOPO[topo], c, g[b]);
      if (r.warn.some(w => w.bad) || !Number.isFinite(r.stepT)) continue;
      const same = sigT(r) === base;
      if (inert[k] && !same) fail(`train ${mk}/${fw}/${b} tp${tp} cp${cp} pp${pp} ep${ep} offload=${offload}: ${k} is marked inert (${inert[k]}) but ${k}=${v} changes the result`);
      if (!inert[k]) { seenT[k] = seenT[k] || 0; if (!same) seenT[k]++; else if (process.env.VERBOSE) (quiet["T " + k] = quiet["T " + k] || []).push(`${mk}/${fw}/${b} tp${tp} cp${cp} pp${pp} ep${ep} off=${offload} ${v}`); }
    }
  }
}
{ const c = Object.assign({}, Train.BASE, Train.FW_DEF.torchtitan, { fw: "torchtitan", N: 8, ep: 8, dispatcher: "mok" }), m = MODELS.qwen35, a = Train.run(m, HW.H200, TOPO.hgx8, c, g.modeled), b = Train.run(m, HW.H200, TOPO.hgx8, Object.assign({}, c, { mokF: 0.5 }), g.modeled);
  seenT.mokF = sigT(a) !== sigT(b) ? 1 : 0; if (Train.inert(m, c, g.modeled).mokF) fail("mokF must not be inert with the MoK dispatcher"); }
for (const k of Object.keys(TRN)) if (!seenT[k]) fail(`train: ${k} never changes any result`);

// ---- inference ----
const seenI = {};
const moeOf = { sglang: ["tp", "replicated", "deepep", "standard"], vllm: ["tp", "ag", "deepep"], tokenspeed: ["replicated", "tp"] };
for (const mk of Object.keys(MODELS)) for (const b of ["modeled", "ideal"]) for (const engine of Object.keys(moeOf)) for (const moePar of moeOf[engine]) for (const phase of ["decode", "prefill"]) for (const [N, dpA] of [[1, 1], [8, 1], [8, 8]]) for (const specK of [0, 3]) {
  if (phase === "prefill" && specK) continue;
  const m = MODELS[mk], c0 = Object.assign({}, Infer.BASE, { engine, moePar, phase, N, dpA, specK, accept: specK ? 1 : 0, B: 16, P: 16384, C: 4096, bi: false });
  const run = c => (phase === "decode" ? Infer.decode : Infer.prefill)(m, HW.H200, TOPO.hgx8, c, g[b]);
  const inert = Infer.inert(m, c0, g[b]), base = sigI(run(c0));
  for (const [k, alt] of Object.entries(INF)) {
    const vals = typeof alt === "function" ? [alt(c0[k])] : alt;
    for (const v of vals) {
      const c = Object.assign({}, c0, { [k]: v }), same = sigI(run(c)) === base;
      if (inert[k] && !same) fail(`infer ${mk}/${b}/${engine}/${moePar}/${phase} N${N} dpA${dpA} k${specK}: ${k} is marked inert (${inert[k]}) but ${k}=${v} changes the result`);
      if (!inert[k]) { seenI[k] = seenI[k] || 0; if (!same) seenI[k]++; else if (process.env.VERBOSE) (quiet["I " + k] = quiet["I " + k] || []).push(`${mk}/${b}/${engine}/${moePar}/${phase} N${N} dpA${dpA} k${specK} ${v}`); }
    }
  }
}
{ const c = Object.assign({}, Infer.BASE, { bi: true }), m = MODELS.qwen35, a = Infer.decode(m, HW.H200, TOPO.hgx8, c, g.modeled), b = Infer.decode(m, HW.H200, TOPO.hgx8, Object.assign({}, c, { biFactor: 5 }), g.modeled);
  seenI.biFactor = sigI(a) !== sigI(b) ? 1 : 0; if (Infer.inert(m, c, g.modeled).biFactor) fail("biFactor must not be inert with bi on"); }
for (const k of Object.keys(INF)) if (!seenI[k]) fail(`infer: ${k} never changes any result`);

// ---- every parallel layout: invariants ----
let nLay = 0;
const flat = {};
for (const model of Object.keys(MODELS)) for (const fw of ["megatron", "torchtitan"]) for (const [hw, topo, N] of [["H200", "hgx8", 8], ["GB300", "nvl64", 64]])
  for (const tp of pow2(N)) for (const cp of pow2(N / tp)) for (const pp of pow2(N / tp / cp)) for (const ep of pow2(N)) {
    const q = { model, hw, topo, fw, N, tp, cp, pp, ep, S: 8192, tok: 8192, Gtok: 8192 * N * 4, recompute: "none", offload: false };
    if (fw === "megatron") Object.assign(q, { cpType: "allgather", dispatcher: "alltoall" });
    const mo = Scouter.train(Object.assign({ basis: "modeled" }, q)), id = Scouter.train(Object.assign({ basis: "ideal" }, q));
    if (mo.warn.some(w => w.bad)) continue;
    nLay++;
    const tag = `${model}/${fw}/${hw} tp${tp} cp${cp} pp${pp} ep${ep}`, by = k => mo.comm.filter(x => x.key === k).reduce((s, x) => s + x.bytes, 0);
    if (!(id.stepT <= mo.stepT * (1 + 1e-9))) fail(`${tag}: ideal ${id.stepT} slower than modeled ${mo.stepT}`);
    if (!(mo.mfu <= 1 && id.mfu <= 1)) fail(`${tag}: MFU above 1`);
    if (tp === 1 && by("tp") > 0) fail(`${tag}: TP traffic at TP 1`);
    if (cp === 1 && by("cpkv") + by("cplin") + by("cpswa") > 0) fail(`${tag}: CP traffic at CP 1`);
    if (ep === 1 && by("a2a") > 0) fail(`${tag}: EP traffic at EP 1`);
    const fl = (id.flF + id.flB) / (id.T / (tp * cp)), k = `${model}/${fw}/${hw}`;
    // Work per token is layout-independent except the replicated projections of MLA / DSA under TP and the transition matrix of the state-passing linear CP
    if (tp === 1 && cp === 1) { if (flat[k] == null) flat[k] = fl; else if (Math.abs(fl / flat[k] - 1) > 1e-9) fail(`${tag}: FLOPs per token change with PP / EP: ${fl / flat[k]}`); }
  }

// ---- GDN CP: linCP reaches the simulator, and state passing is the cheap wire option ----
{ const q = { model: "qwen35", hw: "H200", topo: "hgx8", fw: "torchtitan", cp: 2, ep: 4, tok: 8192, S: 8192, pick: null };
  const st = Scouter.train(Object.assign({ linCP: "state" }, q)), ul = Scouter.train(Object.assign({ linCP: "ulysses" }, q)), b = r => r.comm.find(x => x.key === "cplin").bytes;
  const l = MODELS.qwen35.lin, conv = (l.conv - 1) * Common.derive(MODELS.qwen35).linConvDim * 2, exp = l.n * (2 * (2 * l.Hv * l.dk * (l.dv + l.dk) * 4) / 2 + 2 * conv);
  if (Math.abs(b(st) / exp - 1) > 1e-9) fail(`GDN state CP bytes ${b(st)} != ${exp}`);
  if (!(b(st) < b(ul) / 10)) fail(`GDN state CP should be far below head-wise a2a: ${b(st)} vs ${b(ul)}`);
  if (!(st.flF > ul.flF)) fail("GDN state CP must charge the transition matrix"); }

if (process.env.VERBOSE) for (const [k, v] of Object.entries(quiet)) console.log(k, v.length, v.slice(0, 4).join(" | "));
if (fails.length) { console.error(fails.join("\n")); if (fails.more) console.error(`... and ${fails.more} more`); process.exit(1); }
console.log(`config reach: ${Object.keys(seenT).length} train keys, ${Object.keys(seenI).length} infer keys; ${nLay} valid layouts`);
console.log("config reach ok");
