/* 数据平面的入口：计算核心（src/engine）加数据点库（src/data），给界面、命令行（tools/scouter.js）和 agent 用同一套接口。
   输入是普通对象：模型 / GPU / 拓扑写 key，其余字段覆盖默认配置；输出是能 JSON 序列化的普通对象（文字是 Msg）。
   不碰 DOM、不调 t()、不读界面状态；浏览器里和 Node 里（src/plane/node.js）跑的是同一份代码。
   口径 basis：ideal = 硬件屋顶（Common.IDEAL），modeled = 默认效率与固定开销（Common.G0）。实测不是算出来的，走 points / calib。 */
const Scouter = (function(){
const { MODELS, HW, TOPO, BASIS } = Common;
const VERSION = 1;
const SOURCES = ["ideal", "modeled", "measured"];
const KEYS = ["model", "hw", "topo", "basis", "g", "pick"];

const pick = (table, key, what) => {
  if(!(key in table)) throw new Error(what + ' "' + key + '" not found; one of: ' + Object.keys(table).join(", "));
  return table[key];
};
/* 效率参数：口径的那一套，再用 q.g 覆盖个别项 */
const gOf = q => Object.assign({}, pick(BASIS, (q && q.basis) || "modeled", "basis"), (q && q.g) || {});
const ctxOf = q => ({
  m: q.model && typeof q.model === "object" ? q.model : pick(MODELS, q.model || "qwen35", "model"),
  hw: pick(HW, q.hw || "H200", "hw"), topo: pick(TOPO, q.topo || "hgx8", "topo"), g: gOf(q) });
const rest = q => { const o = Object.assign({}, q); for(const k of KEYS) delete o[k]; return o; };
const get = (o, path) => path.split(".").reduce((x, k) => x == null ? x : x[k], o);
/* pick: ["stepT", "mem.total"] → 只留这些字段（点号是嵌套） */
const select = (r, keys) => keys && keys.length ? Object.fromEntries(keys.map(k => [k, get(r, k)])) : r;
const echo = (q, ctx) => ({ model: q.model && typeof q.model === "object" ? "custom" : q.model || "qwen35", hw: ctx.hw.name, topo: q.topo || "hgx8",
  basis: (q && q.basis) || "modeled" });

/* 一个 GPU 的屋顶：各精度的峰值、口径下的可达峰值和拐点；给了 flops / bytes 就再算这一个 kernel */
function roofline(q){
  q = q || {};
  const hw = pick(HW, q.hw || "H200", "hw"), g = gOf(q), bw = hw.hbm * 1e9 * g.etaM;
  const peaks = {};
  for(const d of ["bf16", "fp8", "fp4"]){ const p = Common.peakOf(hw, d);
    peaks[d] = { peakTFLOPs: p / 1e12, attainableTFLOPs: p * g.etaC / 1e12, ridge: Common.ridge(p, hw, g) }; }
  const out = { hw: hw.name, basis: q.basis || "modeled", hbmGBs: hw.hbm, attainableGBs: bw / 1e9, memGB: hw.mem, nvlinkGBs: hw.nvl, peaks };
  if(q.flops != null || q.bytes != null){
    const dt = q.dtype || "bf16", r = Common.roof(+q.flops || 0, +q.bytes || 0, Common.peakOf(hw, dt), hw, g, q.kernels || 0);
    out.kernel = Object.assign({ dtype: dt, flops: +q.flops || 0, bytes: +q.bytes || 0, TFLOPs: r.t > 0 ? (+q.flops || 0) / r.t / 1e12 : 0 }, r);
  }
  return out;
}
/* 推理一步（Infer.step）：q 里除 model / hw / topo / basis / g 以外的字段覆盖 Infer.BASE */
function infer(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Infer.BASE, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, Infer.step(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
/* 训练一步（Train.run）：覆盖 Train.BASE；N 默认是拓扑的卡数 */
function train(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Train.BASE, { N: c.topo.gpus }, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, Train.run(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
/* 架构页：一层里每个 attention / MLP 模块的 kernel、时间和瓶颈（引擎中立的参考画像），覆盖 Arch.defaults */
function module(q){
  q = q || {}; const c = ctxOf(q), wl = Object.assign(Arch.defaults(c.m), rest(q));
  return select({ query: Object.assign(echo(q, c), { workload: wl }), modules: Arch.present(c.m), measure: Arch.measure(c.m, c.hw, c.g, wl), facts: Arch.facts(c.m, wl) }, q.pick);
}
const RUN = { roofline, infer, train, module };
/* 扫参：run 是上面的一个函数名，over 的每个字段给一组值（多个字段取笛卡儿积），pick 选出每一行要的字段 */
function sweep(q){
  q = q || {}; const f = pick(RUN, q.run || "infer", "run"), over = q.over || {}, keys = Object.keys(over);
  let rows = [{}];
  for(const k of keys){ if(!Array.isArray(over[k])) throw new Error("over." + k + " must be an array"); rows = rows.flatMap(r => over[k].map(v => Object.assign({}, r, { [k]: v }))); }
  if(rows.length > 4096) throw new Error("sweep too large: " + rows.length + " points (max 4096)");
  const base = Object.assign({}, q); delete base.run; delete base.over;
  return { run: q.run || "infer", over, rows: rows.map(r => { const res = f(Object.assign({}, base, r, { pick: null }));
    return Object.assign({}, r, select(res, q.pick && q.pick.length ? q.pick : ["stepT"])); }) };
}

/* ---- 实测：数据点库 ---- */
function sets(){
  return Object.fromEntries(Object.keys(DB.sets).map(id => { const s = DB.sets[id];
    return [id, { origin: s.origin, stage: s.stage, hw: s.hw || null, gpus: s.gpus || null, model: s.model || null, n: DB.count(id), dims: s.dims || null,
      metrics: s.metrics ? Object.keys(s.metrics) : null, calib: id in Calib.SETS }]; }));
}
/* 一个数据集里的点；where 是等值筛选（数组 = 其中之一） */
function points(q){
  q = q || {}; if(!q.set) throw new Error("points needs a set; one of: " + Object.keys(DB.sets).join(", "));
  pick(DB.sets, q.set, "set");
  return select({ set: q.set, meta: DB.meta(q.set), points: DB.where(q.set, q.where || {}) }, q.pick);
}
/* 和实测对照：用计算核心复现一个匿名数据集，每个点给实测、预测和预测 / 实测 */
function calib(q){
  q = q || {}; const id = q.set || "cp_attention", K = pick(Calib.SETS, id, "calib set"), g = gOf(q);
  const rows = (K.where ? DB.where(id, K.where) : DB.set(id)).map(p => { let pred = null; try { pred = K.predict(p, g); } catch(e) {}
    const meas = K.meas(p); return { p, meas, pred, ratio: isFinite(pred) && pred > 0 && meas > 0 ? pred / meas : null }; }).filter(r => r.ratio != null);
  const rs = rows.map(r => r.ratio).sort((a, b) => a - b), mid = (rs.length - 1) / 2;
  return select({ set: id, basis: q.basis || "modeled", n: rows.length, median: rs.length ? (rs[Math.floor(mid)] + rs[Math.ceil(mid)]) / 2 : null, rows }, q.pick);
}

/* 实测 kernel 放到屋顶上：单卡 GEMM 和 HBM 拷贝的实测点（数据点库里 stage = "kernel"、带 op 的数据集），
   和同一块 GPU 的理想屋顶、建模屋顶放在一起；measured 是实测里最好的 GEMM 算力和拷贝带宽 */
function kernels(q){
  q = q || {}; const id = q.set || "h200_kernels", meta = pick(DB.sets, id, "set"), hw = pick(HW, meta.hw, "hw");
  const pts = DB.set(id).filter(p => p.op === "bf16_gemm" || p.op === "hbm_copy").map(p => {
    const k = p.op === "bf16_gemm" ? { shape: [p.m, p.n, p.k], flops: 2 * p.m * p.n * p.k, bytes: 2 * (p.m * p.k + p.k * p.n + p.m * p.n) } : { flops: 0, bytes: p.rw_bytes };
    const s = p.ms / 1e3;
    return Object.assign({ id: p.id, op: p.op, s }, k, { I: k.bytes > 0 ? k.flops / k.bytes : null, TFLOPs: k.flops / s / 1e12, GBs: k.bytes / s / 1e9 }); });
  if(!pts.length) throw new Error("set " + id + " has no kernel points");
  const gemm = pts.filter(p => p.flops > 0), copy = pts.filter(p => !p.flops);
  const measured = { TFLOPs: gemm.length ? Math.max(...gemm.map(p => p.TFLOPs)) : null, GBs: copy.length ? Math.max(...copy.map(p => p.GBs)) : null };
  return select({ set: id, hw: hw.name, ideal: roofline({ hw: meta.hw, basis: "ideal" }), modeled: roofline({ hw: meta.hw, basis: "modeled" }), measured, points: pts }, q.pick);
}

/* 目录：有哪些模型、GPU、拓扑、口径、引擎、配方和数据集；默认配置原样给出，agent 照着改 */
function catalog(){
  return { version: VERSION, sources: SOURCES, basis: BASIS,
    models: Object.fromEntries(Object.keys(MODELS).map(k => [k, { name: MODELS[k].name, family: MODELS[k].family }])),
    hw: HW, topo: TOPO, engines: Object.keys(Infer.ENGINES), spec: Object.keys(Infer.SPEC), prec: Object.keys(Train.PREC), epBackends: Object.keys(Par.EP),
    quant: Quant.ORDER, defaults: { infer: Infer.BASE, train: Train.BASE, module: Arch.defaults(MODELS.qwen35) },
    runs: Object.keys(RUN).concat(["sweep", "points", "kernels", "calib", "sets", "catalog"]), sets: sets() };
}

return Object.freeze({ VERSION, SOURCES, gOf, catalog, roofline, infer, train, module, sweep, sets, points, kernels, calib });
})();
