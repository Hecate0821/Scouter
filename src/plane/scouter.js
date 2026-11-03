/* 数据平面的入口：计算核心（src/engine）加数据点库（src/data），给界面、命令行（tools/scouter.js）和 agent 用同一套接口。
   输入是普通对象：模型 / GPU / 拓扑写 key，其余字段覆盖默认配置；输出是能 JSON 序列化的普通对象（文字是 Msg）。
   不碰 DOM、不调 t()、不读界面状态；浏览器里和 Node 里（src/plane/node.js）跑的是同一份代码。
   口径 basis：ideal = 理想下限（Common.IDEAL：效率 1、没有开销、只算必要的流量、资源守恒），modeled = 尽量贴近现实的建模（Common.G0）。实测不是算出来的，走 points / calib。 */
const Scouter = (function(){
const { MODELS, HW, TOPO, BASIS, G0, KEFF } = Common;
const VERSION = 1;
const SOURCES = ["ideal", "modeled", "measured"];
const KEYS = ["model", "hw", "topo", "basis", "g", "pick"];

const pick = (table, key, what) => {
  if(!(key in table)) throw new Error(what + ' "' + key + '" not found; one of: ' + Object.keys(table).join(", "));
  return table[key];
};
/* 效率参数：口径的那一套，再用 q.g 覆盖个别项 */
/* 不指定时：GB300、64 卡。只指定 GPU 时拓扑跟着它走，和页面的规则一样：GB 系列在 NVL72 机柜，其余在单台 HGX */
const DEF_HW = "GB300", topoOf = q => q.topo || (/^GB/.test(q.hw || DEF_HW) ? "nvl64" : "hgx8");
const gOf = q => Object.assign({}, pick(BASIS, (q && q.basis) || "modeled", "basis"), (q && q.g) || {});
const ctxOf = q => ({
  m: q.model && typeof q.model === "object" ? q.model : pick(MODELS, q.model || "qwen35", "model"),
  hw: pick(HW, q.hw || DEF_HW, "hw"), topo: pick(TOPO, topoOf(q), "topo"), g: gOf(q) });
const rest = q => { const o = Object.assign({}, q); for(const k of KEYS) delete o[k]; return o; };
const get = (o, path) => path.split(".").reduce((x, k) => x == null ? x : x[k], o);
/* pick: ["stepT", "mem.total"] → 只留这些字段（点号是嵌套） */
const select = (r, keys) => keys && keys.length ? Object.fromEntries(keys.map(k => [k, get(r, k)])) : r;
const echo = (q, ctx) => ({ model: q.model && typeof q.model === "object" ? "custom" : q.model || "qwen35", hw: ctx.hw.name, topo: topoOf(q),
  basis: (q && q.basis) || "modeled" });

/* 一个 GPU 的屋顶：各精度的峰值、口径下的可达峰值和拐点；给了 flops / bytes 就再算这一个 kernel */
function roofline(q){
  q = q || {};
  const hw = pick(HW, q.hw || DEF_HW, "hw"), g = gOf(q), bw = hw.hbm * 1e9 * g.etaM;
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
/* 推理一步（decode 走 Infer.decode，含投机的 draft / commit；prefill 走 Infer.prefill，含整段 prompt 的 TTFT，和页面同一条路）：
   q 里除 model / hw / topo / basis / g 以外的字段覆盖 Infer.BASE */
function infer(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Infer.BASE, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, (cfg.phase==="prefill"?Infer.prefill:Infer.decode)(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
/* 训练一步（Train.run）：覆盖 Train.BASE 和所选框架的默认值 Train.FW_DEF；N 默认是拓扑的卡数 */
function train(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Train.BASE, Train.FW_DEF[q.fw]||{}, { N: c.topo.gpus }, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, Train.run(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
/* 架构页：一层里每个 attention / MLP 模块的 kernel、时间和瓶颈（引擎中立的参考画像），覆盖 Arch.defaults */
function module(q){
  q = q || {}; const c = ctxOf(q), wl = Object.assign(Arch.defaults(c.m), rest(q));
  return select({ query: Object.assign(echo(q, c), { workload: wl }), modules: Arch.present(c.m), measure: Arch.measure(c.m, c.hw, c.g, wl), facts: Arch.facts(c.m, wl) }, q.pick);
}
/* 闭环压测（Infer.serve）：conc 条请求并发，每条 isl 输入、osl 输出；给平均 TPOT、空载 TTFT 和 cookbook 口径的 tok/s/GPU */
function serve(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Infer.BASE, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, Infer.serve(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
/* KV cache（KV.plan）：每种层每个 token / 每个请求存多少、agent 多轮会话在 HBM / 主机内存 / 远端池里的驻留比例、命中率和每轮 TTFT；
   覆盖 Infer.BASE 和 KV.BASE（会话形状、缓存层级、路由） */
function kv(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Infer.BASE, KV.BASE, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, KV.plan(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
const RUN = { roofline, infer, train, module, serve, kv };
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
    return [id, { origin: s.origin, stage: s.stage, confidence: s.confidence, asOf: s.as_of || null, hw: s.hw || null, gpus: s.gpus || null, model: s.model || null, n: DB.count(id), dims: s.dims || null,
      metrics: s.metrics ? Object.keys(s.metrics) : null, calib: calibOf(id) }]; }));
}
/* 哪些校准视图用这个数据集（Calib.SETS 的键；set 字段指向别的数据集时算那个数据集的） */
const calibOf = id => Object.keys(Calib.SETS).filter(k => (Calib.SETS[k].set || k) === id);
/* 一个数据集里的点；where 是等值筛选（数组 = 其中之一） */
function points(q){
  q = q || {}; if(!q.set) throw new Error("points needs a set; one of: " + Object.keys(DB.sets).join(", "));
  pick(DB.sets, q.set, "set");
  return select({ set: q.set, meta: DB.meta(q.set), points: DB.where(q.set, q.where || {}) }, q.pick);
}
/* 和实测对照：用计算核心复现一个匿名数据集，每个点给实测、预测和预测 / 实测 */
function calib(q){
  q = q || {}; const id = q.set || "cp_attention", K = pick(Calib.SETS, id, "calib set"), g = gOf(q);
  const sid = K.set || id, rows = (K.where ? DB.where(sid, K.where) : DB.set(sid)).map(p => { let pred = null; try { pred = K.predict(p, g); } catch(e) {}
    const meas = K.meas(p); return { p, meas, pred, ratio: isFinite(pred) && pred > 0 && meas > 0 ? pred / meas : null }; }).filter(r => r.ratio != null);
  const rs = rows.map(r => r.ratio).sort((a, b) => a - b), mid = (rs.length - 1) / 2;
  return select({ set: id, data: sid, basis: q.basis || "modeled", n: rows.length, median: rs.length ? (rs[Math.floor(mid)] + rs[Math.ceil(mid)]) / 2 : null, rows }, q.pick);
}

/* 实测 kernel 放到屋顶上，并按 Nsight Compute 的 SOL（speed of light）口径给每个点的实测效率：
   ηc = 解析 FLOPs ÷ 时长 ÷ 规格峰值，ηm = 字节 ÷ 时长 ÷ HBM 峰值。点有两种：
   - 单卡 GEMM / HBM 拷贝（h200_kernels 那种，op = bf16_gemm / hbm_copy）：FLOPs 和字节按形状算；
   - tools/import_ncu.py 导入的 ncu 点（kernel、us、dram_bytes，可选 flops）：字节是 ncu 量到的 DRAM 读 + 写，
     另带 ncu 自己报的 sm_pct / dram_pct（占峰值的百分比）作对照。
   q.points 可以直接传一组这样的点（不进数据点库），这时要给 q.hw 和 q.dtype；measured 是实测里最好的算力和带宽 */
function kernels(q){
  q = q || {}; const id = q.points ? null : q.set || "h200_kernels", meta = id ? pick(DB.sets, id, "set") : { hw: q.hw };
  const hwKey = q.hw || meta.hw, hw = pick(HW, hwKey, "hw"), dt = q.dtype || (meta.profile && meta.profile.dtype) || "bf16", peak = Common.peakOf(hw, dt);
  const sol = k => Object.assign(k, { I: k.bytes > 0 && k.flops > 0 ? k.flops / k.bytes : null, TFLOPs: k.flops > 0 ? k.flops / k.s / 1e12 : null, GBs: k.bytes > 0 ? k.bytes / k.s / 1e9 : null,
    etaC: k.flops > 0 ? k.flops / k.s / peak : null, etaM: k.bytes > 0 ? k.bytes / k.s / (hw.hbm * 1e9) : null });
  const pts = (q.points || DB.set(id)).map(p => {
    if(p.op === "bf16_gemm") return sol({ id: p.id, op: p.op, s: p.ms / 1e3, shape: [p.m, p.n, p.k], flops: 2 * p.m * p.n * p.k, bytes: 2 * (p.m * p.k + p.k * p.n + p.m * p.n) });
    if(p.op === "hbm_copy") return sol({ id: p.id, op: p.op, s: p.ms / 1e3, flops: 0, bytes: p.rw_bytes });
    if(p.us > 0) return sol({ id: p.id || p.kernel, op: "ncu", kernel: p.kernel, s: p.us / 1e6, flops: +p.flops || 0, bytes: +p.dram_bytes || 0, sm_pct: p.sm_pct ?? null, dram_pct: p.dram_pct ?? null });
    return null; }).filter(Boolean);
  if(!pts.length) throw new Error((id ? "set " + id : "points") + " has no kernel points");
  const best = k => { const v = pts.map(p => p[k]).filter(x => x > 0); return v.length ? Math.max(...v) : null; };
  return select({ set: id, hw: hw.name, dtype: dt, profile: meta.profile || null, ideal: roofline({ hw: hwKey, basis: "ideal" }), modeled: roofline({ hw: hwKey, basis: "modeled" }),
    measured: { TFLOPs: best("TFLOPs"), GBs: best("GBs"), etaC: best("etaC"), etaM: best("etaM") }, points: pts }, q.pick);
}
/* 按 kernel 类别的效率表（Common.KEFF）：每一格带出处的数据点和它相对 η_c / η_m 的倍数 */
function eff(){
  return { reference: { etaC: G0.etaC, etaM: G0.etaM }, table: KEFF.map(r => { const sid = r.ref.split(":")[0], p = DB.one(sid, { id: r.ref }) || null;
    return Object.assign({}, r, { ratio: r.eta / (r.kind === "m" ? G0.etaM : G0.etaC), evidence: p }); }) };
}

/* 实测 trace 对照（Trace.compare）：trace 是 Chrome trace 或 [{name, us, calls}]，steps 是覆盖的步数；其余字段和 infer 一样，
   按同样的设置跑一步 Infer.step，逐类别（gemm / attn / lin / moe / comm / samp / elem）比每步时间 */
function trace(q){
  q = q || {}; if(q.trace == null) throw new Error("trace needs a trace: a Chrome trace {traceEvents} or a list of {name, us, calls}");
  const c = ctxOf(q), o = rest(q); delete o.trace; delete o.steps;
  const cfg = Object.assign({}, Infer.BASE, o), r = Infer.step(c.m, c.hw, c.topo, cfg, c.g);
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, Trace.compare(q.trace, q.steps, r)), q.pick);
}

/* 目录：有哪些模型、GPU、拓扑、口径、引擎、配方和数据集；默认配置原样给出，agent 照着改 */
function catalog(){
  return { version: VERSION, sources: SOURCES, basis: BASIS,
    models: Object.fromEntries(Object.keys(MODELS).map(k => [k, { name: MODELS[k].name, family: MODELS[k].family }])),
    hw: HW, topo: TOPO, engines: Object.keys(Infer.ENGINES), spec: Object.keys(Infer.SPEC), prec: Object.keys(Train.PREC), epBackends: Object.keys(Par.EP),
    quant: Quant.ORDER, defaults: { infer: Infer.BASE, kv: KV.BASE, train: Train.BASE, trainFw: Train.FW_DEF, module: Arch.defaults(MODELS.qwen35) },
    runs: Object.keys(RUN).concat(["sweep", "points", "kernels", "eff", "calib", "trace", "sets", "catalog"]), calib: Object.keys(Calib.SETS), sets: sets() };
}

return Object.freeze({ VERSION, SOURCES, gOf, catalog, roofline, infer, train, module, serve, kv, sweep, sets, points, kernels, eff, calib, trace });
})();
