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

// --- two-batch overlap: two micro-batches cost weight reads and kernels, hide a2a behind each other; the ideal does not split ---
{
  const { MODELS, HW, TOPO, Infer, Common } = E, m = MODELS.glm53f, hw = HW.GB200, topo = TOPO.nvl64;
  const cfg = (B, tbo, extra) => Object.assign({}, Infer.BASE, { N: 64, dpA: 64, moePar: "deepep", B, wDt: "nvfp4", tbo }, extra || {});
  const dec = (c, gg) => Infer.decode(m, hw, topo, c, gg);
  const on = dec(cfg(64, true), Common.G0), off = dec(cfg(64, false), Common.G0);
  if (!on.mbs || on.mbs.rows[0] + on.mbs.rows[1] !== on.rowsG) bad("tbo", "the two micro-batches do not add up to the padded batch");
  const moeB = r => r.ops.filter(o => o.key === "moe").reduce((s, o) => s + o.bytes, 0);
  if (!(moeB(on) > 1.5 * moeB(off))) bad("tbo", "each micro-batch should read the expert weights again");
  if (!(on.nKern > 1.9 * off.nKern)) bad("tbo", "each micro-batch should launch its own kernels");
  if (!(dec(cfg(4, true), Common.G0).stepT > dec(cfg(4, false), Common.G0).stepT)) bad("tbo", "at 4 requests per GPU TBO should cost time");
  for (const B of [4, 64, 512]) {
    if (!(dec(cfg(B, true), Common.IDEAL).stepT <= dec(cfg(B, true), Common.G0).stepT)) bad("tbo", `ideal slower than modeled at B ${B}`);
    if (!(dec(cfg(B, true), Common.IDEAL).stepT <= dec(cfg(B, false), Common.IDEAL).stepT * (1 + 1e-9))) bad("tbo", `ideal with TBO slower than without at B ${B}`);
  }
  const one = dec(cfg(1, true), Common.G0), oneOff = dec(cfg(1, false), Common.G0);
  if (one.mbs || Math.abs(one.stepT - oneOff.stepT) > 1e-12) bad("tbo", "one request cannot be split, so TBO should change nothing");
  const sw = Infer.tboSweep(m, hw, topo, cfg(64, false), Common.G0);
  if (!sw || sw.rows.length !== Infer.TBO_B.length || sw.swapped) bad("tbo", "tboSweep rows");
  if (Infer.tboSweep(MODELS.qwen38_27b, hw, topo, cfg(64, false), Common.G0) !== null) bad("tbo", "a dense model has no TBO sweep");
  const pf = Infer.prefill(m, hw, topo, cfg(1, true, { phase: "prefill", C: 8192, P: 8192 }), Common.G0);
  if (!(pf.ttft > 0 && Number.isFinite(pf.ttft))) bad("tbo", "prefill with TBO");
}

// --- DeepSeek-V4.1-Flash (CSA): parameters, KV bytes, bounded replay, Engram placement and the SGLang-unsupported combinations ---
{
  const { MODELS, HW, TOPO, Infer, Common } = E, m = MODELS.dsv41f, d = Common.derive(m), G = Common.G0;
  const near = (x, want, tol) => Math.abs(x - want) <= tol * want;
  // vLLM recipe: 552B backbone, 16B active per output token; the Engram tables (about 197B) come on top of the backbone
  if (!near(d.total - d.pEngTab, 551.8e9, 0.01)) bad("dsv41f", `backbone ${(d.total - d.pEngTab) / 1e9}B, expected about 552B`);
  if (!near(d.pEngTab, 196.6e9, 0.01)) bad("dsv41f", `Engram tables ${d.pEngTab / 1e9}B, expected about 196.6B`);
  if (!near(d.active, 16.1e9, 0.02)) bad("dsv41f", `active ${d.active / 1e9}B, expected about 16B`);
  // bytes per token over the whole model: compressed latent at the 4 kv sources (+ ratio), index K, window latent at all 40 layers
  const kvTok = dt => { const k = Common.csaKv(m, dt); return k.main + k.index; };
  if (Math.round(kvTok("nvfp4")) !== 890) bad("dsv41f", `NVFP4 compressed + index KV ${kvTok("nvfp4")} B/token, expected 890`);
  if (Math.round(kvTok("fp8")) !== 1630) bad("dsv41f", `FP8 (V4 layout) compressed + index KV ${kvTok("fp8")} B/token, expected 1630`);
  const base = Object.assign({}, Infer.BASE, { engine: "sglang", N: 4, dpA: 1, moePar: "replicated", wDt: "mxfp4", dDt: "fp8", kvDt: "nvfp4" });
  const pf = (c, gg) => Infer.prefill(m, HW.B200, TOPO.hgx8, Object.assign({}, base, { phase: "prefill", C: 8192, P: 8192 }, c), gg || G);
  const rp = pf({}), full = pf({ replay: false });
  if (!rp.replay || rp.replay.cut !== 20) bad("dsv41f", "SGLang's replay should cut at the last kv source (layer 20)");
  if (!(rp.ttft < 0.8 * full.ttft)) bad("dsv41f", `bounded replay should cut prefill time (replay ${rp.ttft}, full ${full.ttft})`);
  if (!(pf({}, Common.IDEAL).ttft <= rp.ttft)) bad("dsv41f", "ideal replay prefill slower than modeled");
  if (pf({ engine: "vllm", moePar: "tp" }).replay.cut !== 21) bad("dsv41f", "vLLM's replay should cut after the last kv source");
  if (pf({ engine: "tokenspeed" }).replay) bad("dsv41f", "an engine without replay should run the full prefill");
  const wk = r => r.warn.map(w => w.key);
  if (!wk(pf({ N: 8, dpA: 8, moePar: "deepep" })).includes("engine_infer.warn.replayOff")) bad("dsv41f", "SGLang DP attention + DeepEP should turn replay off with a warning");
  const dec = (c, hw) => Infer.decode(m, hw || HW.B200, TOPO.hgx8, Object.assign({}, base, { phase: "decode", B: 64 }, c), G);
  if (!wk(dec({ N: 8, dpA: 8, moePar: "deepep", tbo: true })).includes("engine_infer.warn.csaTbo")) bad("dsv41f", "SGLang TBO should be flagged");
  if (!wk(dec({ specK: 3, specMethod: "mtp", accept: 1 })).includes("engine_infer.warn.csaSpec")) bad("dsv41f", "SGLang speculation other than DSpark should be flagged");
  if (wk(dec({ specK: 5, specMethod: "dspark", accept: 2 })).includes("engine_infer.warn.csaSpec")) bad("dsv41f", "DSpark is supported");
  if (!wk(dec({ N: 8 }, HW.H200)).includes("engine_infer.warn.csaKvHopper")) bad("dsv41f", "NVFP4 KV on Hopper should warn");
  if (wk(dec({ N: 8, kvDt: "fp8" }, HW.H200)).includes("engine_infer.warn.csaKvHopper")) bad("dsv41f", "FP8 KV on Hopper is the supported layout");
  // Engram in host memory: no HBM for the tables, and a host read on the critical path instead
  const gpu = dec({}), host = dec({ engramAt: "host" });
  const tab = d.pEngTab * Common.dtB("mxfp8") / 4;
  if (!near(gpu.mem.weights - host.mem.weights, tab, 0.01)) bad("dsv41f", `host Engram should free ${tab / 1e9} GB per GPU, freed ${(gpu.mem.weights - host.mem.weights) / 1e9}`);
  // SGLang --engram-config (docs/advanced_features/engram.mdx): host + replication none keeps a shard per rank, so the
  // rows still meet in a collective; per_node maps a whole table per node and needs none; GPU tables inside one NVLink
  // domain default to peer_read
  const ck = r => r.comm.map(x => x.key);
  if (!ck(host).includes("enghost") || !ck(host).includes("engar")) bad("dsv41f", "host Engram (replication none) should read this rank's shard from host and all-reduce");
  const pn = dec({ engramAt: "host", engramRepl: "per_node" });
  if (!ck(pn).includes("enghost") || ck(pn).some(k => ["engar", "engrs", "enga2a", "engpeer"].includes(k))) bad("dsv41f", "per_node host Engram needs no collective");
  if (!ck(gpu).includes("engpeer") || ck(gpu).includes("engar")) bad("dsv41f", "GPU Engram on one NVLink domain should default to peer_read");
  if (!ck(dec({ engramLookup: "all_reduce" })).includes("engar")) bad("dsv41f", "an explicit all_reduce lookup should be kept");
  const dpH = dec({ N: 8, dpA: 8, moePar: "deepep", engramAt: "host" });
  if (!ck(dpH).includes("enga2a")) bad("dsv41f", "host Engram under DP 8 should look up by all_to_all (24 columns divide over 8)");
  if (!wk(dec({ engramLookup: "all_to_all" })).includes("engine_infer.warn.engDp")) bad("dsv41f", "all_to_all without DP attention should be flagged");
  if (!wk(dec({ N: 16, engramAt: "host", engramRepl: "per_node" })).includes("engine_infer.warn.engNode")) bad("dsv41f", "per_node across nodes should be flagged");
  // staging hides the host read of a prefill batch behind the previous forward (non-Grace only)
  const pfs = c => Infer.step(m, HW.B200, TOPO.hgx8, Object.assign({}, base, { phase: "prefill", C: 8192, p: 0, engramAt: "host", engramRepl: "per_node" }, c), G);
  const st0 = pfs({}), st1 = pfs({ engramStage: true });
  if (!(st1.stepT < st0.stepT) || !st1.comm.some(x => x.overlap === "stage")) bad("dsv41f", "Engram staging should hide the prefill host read");
  if (pfs({ engramStage: true }).exposedStage > 0) bad("dsv41f", "an 8K prefill's Engram rows should fit behind one forward");
  // MXFP4 tables halve the HBM they take
  const m4 = dec({ engramDt: "mxfp4" });
  if (!(m4.mem.weights < gpu.mem.weights)) bad("dsv41f", "MXFP4 Engram tables should take less HBM");
  // CSA KV waterfall: the model card puts 890 B/token at roughly 1/4 of DeepSeek-V4-Flash
  const steps = E.KV.csaSteps(m, 1), r4 = steps[3].tok / steps[0].tok;
  if (Math.round(steps[3].tok) !== 890 || !(r4 > 0.2 && r4 < 0.3)) bad("dsv41f", `CSA waterfall ${steps.map(x => Math.round(x.tok)).join(" → ")}, 890 should be about 1/4 of V4-Flash`);
  // HSI is introduced in post-training (tech report 2.3.2): at 64K a pretraining step scores the full context in the reindex layers
  const { Train } = E;
  const tr = hsi => Train.run(m, HW.B200, TOPO.nvl64, Object.assign({}, Train.BASE, { N: 64, ep: 64, cp: 8, S: 65536, tok: 65536, hsi }), G);
  const idxFl = r => r.ops.filter(o => o.key === "aidx").reduce((s, o) => s + o.fwd.fl, 0);
  const tPre = tr(false), tPost = tr(true);
  if (!(idxFl(tPre) > 1.5 * idxFl(tPost))) bad("dsv41f", `pretraining (no HSI) at 64K should score far more positions: ${idxFl(tPre)} vs ${idxFl(tPost)}`);
  // DSpark is the model's own drafter: its weights count once, on top of the target's
  const sp = dec({ specK: 5, specMethod: "dspark", accept: 2 });
  if (!(sp.mem.weights > gpu.mem.weights)) bad("dsv41f", "the DSpark drafter's weights should be counted");
  if (!(sp.tpot < gpu.tpot)) bad("dsv41f", "DSpark with 2 accepted tokens per step should lower TPOT at B 64");
}

// --- Profile: the categories add up to the step (plus the DP wait), DP wait only with DP attention, EPLB only with an EP group ---
{
  const { Profile } = E, G = Object.assign({}, G0), d = MODELS.dsv41f;
  const c = o => Object.assign({}, Infer.BASE, { engine: "sglang", N: 8, dpA: 8, moePar: "deepep", B: 64, n: 8192, C: 8192, P: 8192 }, o);
  for (const phase of ["decode", "prefill"]) {
    const sp = Profile.split(d, HW.GB300, TOPO.nvl64, c({ phase, imb: 1.3 }), G, { dpSkew: 1.2 });
    const sum = sp.cats.reduce((s, x) => s + x.t, 0), ks = sp.kern.compute.t + sp.kern.comm.t + sp.kern.idle.t;
    if (Math.abs(sum - sp.total) > 1e-9 * sp.total || Math.abs(ks - sp.total) > 1e-9 * sp.total) bad("profile", `${phase}: categories ${sum} / kern ${ks} should add up to ${sp.total}`);
    if (!(sp.dpLoss > 0 && sp.eplbLoss > 0)) bad("profile", `${phase}: DP 8 with dpSkew 1.2 and imb 1.3 should lose time to both DPLB and EPLB`);
    if (!(Profile.observer(sp, 0.6).t < Profile.observer(sp, 0.8).t)) bad("profile", "a cheaper observer should cost less");
  }
  const one = Profile.split(d, HW.GB300, TOPO.nvl64, c({ phase: "decode", N: 1, dpA: 1, moePar: "tp" }), G, { dpSkew: 1.5 });
  if (one.dpLoss !== 0 || one.eplbLoss !== 0 || one.total !== one.stepT) bad("profile", "one GPU has no DP wait and no EP imbalance");
  if (!Profile.inert(d, c({ dpA: 1 })).dpSkew || Profile.inert(d, c({})).dpSkew) bad("profile", "dpSkew is inert exactly when DP = 1");
}

// --- RTX PRO 6000 (SM120, PCIe only): its home is the PCIe 8-GPU box, TP > 2 falls back to an unfused NCCL ring,
//     and the SM100-only kernels (V4.1 FP4 KV layout, Dist-MoE) are off ---
{
  const { Common } = E, hw = HW.RTXP6000, topo = TOPO.pcie8, G = Object.assign({}, G0);
  if (Common.homeTopo("RTXP6000") !== "pcie8" || Common.homeTopo("GB300") !== "nvl64" || Common.homeTopo("H200") !== "hgx8") bad("rtxp6000", "homeTopo should map GB → nvl64, PCIe cards → pcie8, the rest → hgx8");
  if (Common.sm10x(hw) || !Common.sm10x(HW.B300) || Common.sm10x(HW.H200)) bad("rtxp6000", "sm10x should be true only for SM100 / SM103");
  const q = MODELS.qwen38_27b, base = Object.assign({}, Infer.BASE, { engine: "vllm", phase: "decode", N: 8, dpA: 1, moePar: "tp", B: 32, n: 8192 });
  const onPcie = Infer.decode(q, hw, topo, base, G), onHgx = Infer.decode(q, hw, TOPO.hgx8, base, G);
  const arOf = r => r.comm.filter(x => x.key === "ar").reduce((s, x) => s + x.t, 0);
  if (!(arOf(onPcie) > arOf(onHgx))) bad("rtxp6000", "TP8 on a PCIe-only box should use the slower NCCL ring for the layer all-reduce");
  if (onPcie.comm.some(x => x.key === "ar" && x.name.key === "engine_infer.comm.arFused")) bad("rtxp6000", "AR + norm fusion should be off on a PCIe-only box with TP > 2");
  const d = MODELS.dsv41f, dc = Object.assign({}, Infer.BASE, { engine: "sglang", phase: "decode", N: 8, dpA: 1, moePar: "replicated", B: 16, n: 8192, kvDt: "nvfp4" });
  const wk = r => r.warn.map(w => w.key || "");
  if (!wk(Infer.step(d, hw, topo, dc, G)).includes("engine_infer.warn.csaKvHopper")) bad("rtxp6000", "SM120 has no V4.1 NVFP4 KV layout kernel and should warn");
  if (wk(Infer.step(d, HW.B200, TOPO.hgx8, dc, G)).includes("engine_infer.warn.csaKvHopper")) bad("rtxp6000", "B200 has the V4.1 NVFP4 KV layout kernel");
}

// --- deployment page: DCP sweep, mixed chunked-prefill step, chunked prefill, layouts on G GPUs ---
{
  const { Common } = E, hw = HW.B300, topo = TOPO.hgx8, bads = r => r.warn.filter(w => w.bad).map(w => w.key).join(", ");
  const base = Object.assign({}, Infer.BASE, { N: 8, dpA: 1, moePar: "tp", B: 32, n: 32768, kvDt: "fp8" });
  // DCP: per-token KV and the concurrency cap move the right way, the collectives appear only above dcp 1, the ideal stays a lower bound
  for (const mk of ["glm53f", "kimik3", "dsv41f"]) run(`dcp ${mk}`, () => {
    const m = MODELS[mk], c = Object.assign({}, base, { wDt: m.wDefault }), sw = Infer.dcpSweep(m, hw, topo, c, G0, 256);
    if (sw.why || sw.rows.length < 2) { bad(`dcp ${mk}`, "no sweep: " + sw.why); return null; }
    for (let i = 1; i < sw.rows.length; i++) { const a = sw.rows[i - 1], b = sw.rows[i];
      if (!(b.kvTok < a.kvTok && b.Bmax >= a.Bmax)) bad(`dcp ${mk}`, `dcp ${b.dcp}: KV / token ${b.kvTok} vs ${a.kvTok}, cap ${b.Bmax} vs ${a.Bmax}`);
      if (!(b.dcpT > 0)) bad(`dcp ${mk}`, `dcp ${b.dcp} has no DCP communication`); }
    if (sw.rows[0].dcpT !== 0) bad(`dcp ${mk}`, "DCP communication at dcp 1");
    for (const ph of ["decode", "prefill"]) { const cc = Object.assign({}, c, { dcp: 8, phase: ph, p: ph === "prefill" ? 16384 : 0, C: 4096, P: 32768 }), f = ph === "decode" ? Infer.decode : Infer.prefill;
      const mo = f(m, hw, topo, cc, Common.G0), id = f(m, hw, topo, cc, Common.IDEAL);
      if (bads(mo)) bad(`dcp ${mk} ${ph}`, "dcp 8 at TP 8 is invalid: " + bads(mo));
      if (!(id.stepT <= mo.stepT * (1 + 1e-9))) bad(`dcp ${mk} ${ph}`, `ideal ${id.stepT} slower than modeled ${mo.stepT}`); }
    return sw.rows.map(r => [r.dcp, r.Bmax]);
  });
  run("dcp checks", () => {
    const q = MODELS.qwen38_27b, c = Object.assign({}, base, { wDt: "bf16" });
    if (Infer.decode(q, hw, topo, c, G0).stepT !== Infer.decode(q, hw, topo, Object.assign({}, c, { dcp: 2 }), G0).stepT) bad("dcp gqa", "SGLang must ignore dcp for GQA (inert dcpGqa)");
    if (Infer.dcpSweep(q, hw, topo, c, G0, 256).why !== "dcpGqa") bad("dcp gqa", "sweep must say dcpGqa under SGLang");
    if (Infer.dcpSweep(MODELS.glm53f, hw, topo, Object.assign({}, base, { dpA: 8, moePar: "deepep" }), G0, 256).why !== "dcpNoTp") bad("dcp", "DP attention has no DCP group");
    if (!bads(Infer.decode(MODELS.glm53f, hw, topo, Object.assign({}, base, { wDt: "fp8", dpA: 2, moePar: "deepep", dcp: 8 }), G0))) bad("dcp", "dcp 8 with attention TP 4 must be invalid");
    return true;
  });
  // the mixed step is not faster than its larger part, and not slower than the sum of its decode-only and prefill-only parts unless the
  // combined rows lose a row-count fusion the parts had (SGLang fuses AR + RMSNorm only up to 2048 rows); the ideal is not slower
  for (const mk of Object.keys(MODELS)) for (const [N, dpA, moePar] of [[8, 1, "tp"], [8, 8, "deepep"]]) run(`mixStep ${mk} N${N} dp${dpA}`, () => {
    const m = MODELS[mk], c = Object.assign({}, base, { N, dpA, moePar, n: 8192, wDt: m.wDefault }), out = {};
    for (const [b, gg] of [["modeled", Common.G0], ["ideal", Common.IDEAL]]) {
      const r = Infer.mixStep(m, hw, topo, c, gg, 2048, 4096), pre = Infer.step(m, hw, topo, Object.assign({}, c, { phase: "prefill", C: 2048, p: 4096, reqPerChunk: 0, graph: false }), gg);
      if (!(r.stepT >= Math.max(r.mix.dec, r.mix.pre) * (1 - 1e-9))) bad(`mixStep ${mk} N${N} dp${dpA} ${b}`, `${r.stepT} below max(${r.mix.dec}, ${r.mix.pre})`);
      if (r.fused === pre.fused && !(r.stepT <= (r.mix.dec + r.mix.pre) * (1 + 1e-9))) bad(`mixStep ${mk} N${N} dp${dpA} ${b}`, `${r.stepT} above ${r.mix.dec} + ${r.mix.pre}`);
      out[b] = r.stepT; }
    if (!(out.ideal <= out.modeled * (1 + 1e-9))) bad(`mixStep ${mk} N${N} dp${dpA}`, `ideal ${out.ideal} slower than modeled ${out.modeled}`);
    return out;
  });
  // chunked prefill: the engine picks the schedule; SGLang's switches work and only SGLang has them; the ideal is not slower
  for (const mk of ["glm53f", "kimik3", "qwen35"]) for (const engine of ["sglang", "vllm"]) run(`chunked ${mk} ${engine}`, () => {
    const m = MODELS[mk], c = Object.assign({}, base, { engine, n: 8192, P: 16384, C: 8192, osl: 1024, wDt: m.wDefault });
    const r = Infer.chunked(m, hw, topo, c, Common.G0), ri = Infer.chunked(m, hw, topo, c, Common.IDEAL), w = `chunked ${mk} ${engine}`;
    if (r.mixed !== (engine === "vllm")) bad(w, "mixed = " + r.mixed);
    r.rows.forEach((x, i) => { const y = ri.rows[i];
      if (!(x.sep.stall > 0 && x.sep.ttft >= x.ttftAlone * (1 - 1e-9) && x.sep.tpot >= r.tDec * (1 - 1e-9))) bad(w, `separate C ${x.C}`);
      if (x.mix && !(x.mix.stall >= r.tDec * (1 - 1e-9) && x.mix.tpot >= r.tDec * (1 - 1e-9) && x.mix.ttft > 0)) bad(w, `mixed C ${x.C}`);
      if (!(y.sep.ttft <= x.sep.ttft * (1 + 1e-9) && (!x.mix || y.mix.stall <= x.mix.stall * (1 + 1e-9)))) bad(w, `ideal slower than modeled at C ${x.C}`); });
    const k = Infer.chunked(m, hw, topo, Object.assign({}, c, { pdInterval: 4 }), Common.G0), mx = Infer.chunked(m, hw, topo, Object.assign({}, c, { mixChunk: true }), Common.G0);
    if (engine === "sglang") {
      if (k.k !== 4 || !mx.mixed) bad(w, "SGLang switches not applied");
      k.rows.forEach((x, i) => { if (x.nChunks > 1 && !(x.sep.stall < r.rows[i].sep.stall && x.sep.ttft > r.rows[i].sep.ttft)) bad(w, `interval at C ${x.C}`); });
    } else if (k.k !== 0) bad(w, "pdInterval must be inert outside SGLang");
    return r.rows.length;
  });
  // layouts on 64 GPUs: GPUs divide, dense models have no EP layouts, the ideal is not slower. A layout may be invalid for a reason
  // outside the layout itself (DeepSeek-V4.1's Engram columns under vLLM's all-to-all lookup); the page marks those, so only the
  // DCP / TP group checks are asserted here
  for (const mk of Object.keys(MODELS)) for (const engine of ["sglang", "vllm"]) run(`layouts ${mk} ${engine}`, () => {
    const m = MODELS[mk], c = Object.assign({}, Infer.BASE, { engine, wDt: m.wDefault, kvDt: "fp8" }), L = Infer.layouts(m, c, TOPO.nvl64, 64), moe = Common.derive(m).nMoe > 0;
    if (!moe && L.some(l => l.fam !== "tp" && l.fam !== "dcp")) bad(`layouts ${mk}`, "EP layouts for a dense model");
    for (const l of L) { const w = `layouts ${mk} ${engine} ${l.id}`;
      if (64 % l.N || l.N % l.a || l.a % l.dcp) bad(w, "group sizes do not divide");
      const cc = Object.assign({}, c, l.cfg), mo = Infer.layoutPoint(m, HW.GB300, TOPO.nvl64, cc, Common.G0, 32), id = Infer.layoutPoint(m, HW.GB300, TOPO.nvl64, cc, Common.IDEAL, 32);
      const inv = mo.warn.filter(x => x.bad && /^engine_infer\.warn\.(dcp|dpDivides)/.test(x.key)).map(x => x.key);
      if (inv.length) bad(w, "invalid: " + inv.join(", "));
      if (!(id.stepT <= mo.stepT * (1 + 1e-9))) bad(w, `ideal ${id.stepT} slower than modeled ${mo.stepT}`); }
    return L.length;
  });
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
