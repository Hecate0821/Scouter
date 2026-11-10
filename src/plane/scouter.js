/* 数据平面的入口：计算核心（src/engine）加数据点库（src/data），给界面、命令行（tools/scouter.js）和 agent 用同一套接口。
   输入是普通对象：模型 / GPU / 拓扑写 key，其余字段覆盖默认配置；输出是能 JSON 序列化的普通对象（文字是 Msg）。
   不碰 DOM、不调 t()、不读界面状态；浏览器里和 Node 里（src/plane/node.js）跑的是同一份代码。
   口径 basis：ideal = 理想下限（Common.IDEAL：效率 1、没有开销、只算必要的流量、资源守恒），modeled = 尽量贴近现实的建模（Common.G0）。实测不是算出来的，走 points / calib。
   真实模型（Hga.REAL）的仿真是 HuggingArch（hga / hgaTrain，和理想下限逐项对照）；modeled 只给自定义模型用，页面上真实模型不再显示它。 */
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
/* 不指定时：GB300、64 卡。只指定 GPU 时拓扑跟着它走，和页面的规则一样（Common.homeTopo）：GB 系列在 NVL72 机柜，PCIe 卡在 PCIe 8 卡服务器，其余在单台 HGX */
const DEF_HW = "GB300", topoOf = q => q.topo || (q.hw && q.hw in HW ? Common.homeTopo(q.hw) : q.hw ? "hgx8" : "nvl64");
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
/* 部署页：同样 G 张卡（默认拓扑的卡数）的各种切法在每卡并发 b（默认 32）下的一步（Infer.layouts + layoutPoint），
   当前配置只改 DCP 的扫描（Infer.dcpSweep，attention 不切 TP 时是 null），和 chunked prefill 两种调度的停顿、TTFT、TPOT（Infer.chunked）；覆盖 Infer.BASE */
function deploy(q){
  q = q || {}; const c = ctxOf(q), o = rest(q), G = o.G || c.topo.gpus, b = o.b || 32; delete o.G; delete o.b;
  const cfg = Object.assign({}, Infer.BASE, o);
  const layouts = Infer.layouts(c.m, cfg, c.topo, G).map(l => Object.assign({}, l, Infer.layoutPoint(c.m, c.hw, c.topo, Object.assign({}, cfg, l.cfg), c.g, b)));
  return select({ query: Object.assign(echo(q, c), { cfg, G, b }), layouts, dcp: Infer.dcpSweep(c.m, c.hw, c.topo, cfg, c.g, 256), chunked: Infer.chunked(c.m, c.hw, c.topo, cfg, c.g) }, q.pick);
}
/* KV cache（KV.plan）：每种层每个 token / 每个请求存多少、agent 多轮会话在 HBM / 主机内存 / 远端池里的驻留比例、命中率和每轮 TTFT；
   覆盖 Infer.BASE 和 KV.BASE（会话形状、缓存层级、路由） */
function kv(q){
  q = q || {}; const c = ctxOf(q), cfg = Object.assign({}, Infer.BASE, KV.BASE, rest(q));
  return select(Object.assign({ query: Object.assign(echo(q, c), { cfg }) }, KV.plan(c.m, c.hw, c.topo, cfg, c.g)), q.pick);
}
/* 集群网络（Fabric）：net 覆盖 Fabric.DEF（规模、NVLink 域、机内互联、网卡、叶交换机、收敛比、路由），trn / inf 覆盖训练和推理的默认配置。
   给每一层的带宽、scale-out 的规模、训练每个并行组走的链路，以及训练一步、prefill、decode 的时间和网卡 / NVLink 各露出多少（反事实拆分） */
function cluster(q){
  q = q || {}; const c = ctxOf(q), net = Object.assign({}, Fabric.DEF, q.net || {}), ti = q.trn || {};
  const trn = Object.assign({}, Train.BASE, Train.FW_DEF[ti.fw]||{}, { N: c.topo.gpus }, ti), inf = Object.assign({}, Infer.BASE, q.inf || {});
  return select({ query: Object.assign(echo(q, c), { net, trn, inf }), net: Fabric.resolve(c.hw, c.topo, net), inert: Fabric.inert(c.hw, c.topo, net, c.g, inf),
    ladder: Fabric.ladder(c.hw, c.topo, net, c.g), scaleOut: Fabric.scaleOut(c.hw, c.topo, net, c.g), groups: Fabric.groups(c.m, c.hw, c.topo, trn, net, c.g),
    train: Fabric.split("train", c.m, c.hw, c.topo, trn, net, c.g), prefill: Fabric.split("prefill", c.m, c.hw, c.topo, inf, net, c.g),
    decode: Fabric.split("decode", c.m, c.hw, c.topo, inf, net, c.g) }, q.pick);
}
/* 线上 profile 的拆法（Profile.split）：配置和 infer 一样，另加 dpSkew（最慢的 DP rank 的 attention 工作量 ÷ 平均）；
   observer 是 profile_public 里每一行实测的每节点开销乘上这一步的 kernel 数（上限） */
function profile(q){
  q = q || {}; const c = ctxOf(q), o = rest(q), dpSkew = o.dpSkew != null ? +o.dpSkew : Profile.DEF.dpSkew; delete o.dpSkew;
  const cfg = Object.assign({}, Infer.BASE, o), sp = Profile.split(c.m, c.hw, c.topo, cfg, c.g, { dpSkew });
  const observer = DB.set("profile_public").filter(p => p.usPerNode > 0).map(p => Object.assign({ id: p.id, tool: p.tool, usPerNode: p.usPerNode }, Profile.observer(sp, p.usPerNode)));
  return select({ query: Object.assign(echo(q, c), { cfg, dpSkew }), total: sp.total, stepT: sp.stepT, nKern: sp.nKern, graph: sp.graph, kern: sp.kern, cats: sp.cats,
    dpLoss: sp.dpLoss, eplbLoss: sp.eplbLoss, observer, warn: sp.r.warn }, q.pick);
}
/* ---- HuggingArch：真实模型的仿真后端（src/engine/hga.js）----
   数据是 tools/hga/export.py 离线导出的网格（src/data/hga/<model>.js，HGA_DATA）；这里只取格子、插值，和同一个签名的理想下限逐项对照。
   浏览器里模型的数据按需加载（页面注入 <script>），还没加载时返回 { ok:false, reason:"not-loaded" }；Node 里 src/plane/node.js 全部预先加载。
   签名：q.engine（sglang / vllm；训练 megatron / torchtitan）+ q.sig（Hga.SIG 里的 id），每个引擎 5 个，所有模型共用 */
const hgaData = mk => (typeof HGA_DATA !== "undefined" && HGA_DATA.models[mk]) || null;
function hgaModel(q){
  const mk = q.model || "qwen35";
  if(typeof mk !== "string" || !Hga.isReal(mk)) throw new Error('model "' + (typeof mk === "string" ? mk : "custom") + '" has no HuggingArch simulation; real models: ' + Object.keys(Hga.REAL).join(", "));
  return mk;
}
const hgaMeta = D => D && D.meta ? { hga: D.meta.hga, upstream: D.meta.upstream, generated: D.meta.generated } : null;
/* 推理：签名 × GPU × prompt 长度 isl × 每个 DP rank 的 decode batch b；axis = "b" / "isl" 时另给沿这个轴的整条曲线 */
function hga(q){
  q = q || {}; const mk = hgaModel(q), D = hgaData(mk);
  const p = { model: mk, engine: q.engine || "sglang", sig: q.sig || "tp8", hw: q.hw || DEF_HW, isl: +q.isl || 8192, b: +q.b || 64 };
  if(!(p.hw in Hga.CARD)) throw new Error('GPU "' + p.hw + '" is not in HuggingArch\'s card table; one of: ' + Object.keys(Hga.CARD).join(", "));
  if(!D) return { query: p, ok: false, reason: "not-loaded" };
  const out = Object.assign({ query: p, source: hgaMeta(D) }, Hga.compare(D, p));
  if(q.axis) out.curve = { axis: q.axis, points: Hga.curve(D, Object.assign({}, p, { axis: q.axis })) };
  return select(out, q.pick);
}
/* 训练：签名 × GPU × 序列长度 seq × micro-batch mbs（64 卡、每步 2M token，Hga.GRID）；axis = "seq" 时另给沿序列长度的曲线 */
function hgaTrain(q){
  q = q || {}; const mk = hgaModel(q), D = hgaData(mk);
  const p = { model: mk, engine: q.engine || "megatron", sig: q.sig || "tp8sp", hw: q.hw || DEF_HW, seq: +q.seq || 8192, mbs: +q.mbs || 1 };
  if(!(p.hw in Hga.CARD)) throw new Error('GPU "' + p.hw + '" is not in HuggingArch\'s card table; one of: ' + Object.keys(Hga.CARD).join(", "));
  if(!D) return { query: p, ok: false, reason: "not-loaded" };
  const out = Object.assign({ query: p, source: hgaMeta(D) }, Hga.compareTrain(D, p));
  if(q.axis) out.curve = { axis: "seq", points: Hga.curveTrain(D, p) };
  return select(out, q.pick);
}
/* 这个模型导出了哪些格子：每个引擎签名在每张卡上是能算（ok）还是 HuggingArch 拒绝了（reason） */
function hgaCells(q){
  q = q || {}; const mk = hgaModel(q), D = hgaData(mk);
  if(!D) return { model: mk, ok: false, reason: "not-loaded" };
  const cells = kind => Object.fromEntries(Object.entries(D[kind] || {}).map(([k, v]) => [k, Object.fromEntries(Object.entries(v).map(([c, x]) => [c, x.ok ? true : x.reason]))]));
  return { model: mk, source: hgaMeta(D), grid: D.meta && D.meta.grid, infer: cells("infer"), train: cells("train") };
}
const RUN = { roofline, infer, train, module, serve, deploy, kv, cluster, profile, hga, hgaTrain };
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
    quant: Quant.ORDER, defaults: { infer: Infer.BASE, kv: KV.BASE, train: Train.BASE, trainFw: Train.FW_DEF, module: Arch.defaults(MODELS.qwen35), net: Fabric.DEF },
    runs: Object.keys(RUN).concat(["hgaCells", "sweep", "points", "kernels", "eff", "calib", "trace", "sets", "catalog"]), calib: Object.keys(Calib.SETS), sets: sets(),
    hga: { models: Hga.REAL, cards: Hga.CARD, grid: Hga.GRID, sig: Hga.SIG, roles: Hga.ROLES, loaded: typeof HGA_DATA !== "undefined" ? Object.keys(HGA_DATA.models) : [] } };
}

return Object.freeze({ VERSION, SOURCES, gOf, catalog, roofline, infer, train, module, serve, deploy, kv, cluster, profile, hga, hgaTrain, hgaCells, sweep, sets, points, kernels, eff, calib, trace });
})();
