/* HuggingArch 后端：真实模型（有公开权重、HuggingArch 有核对过的 spec）的仿真不再用 Scouter 的建模口径，而用 HuggingArch
   （github.com/shenh10/HuggingArch，MIT，作者 Han Shen；本仓的 fork 在 vendor/HuggingArch，加了训练步）的逐算子 roofline。
   HuggingArch 是 Python，静态页面跑不了，所以 tools/hga/export.py 离线把每个（模型 × 引擎签名 × GPU）在上下文 × batch 网格上的结果导出到
   src/data/hga/<model>.js；这里只放纯计算：
   · SIG：每个引擎 5 个签名，所有模型共用（推理 SGLang / vLLM，训练 Megatron / TorchTitan），每个签名同时写 HuggingArch 的 ParallelConfig
     （导出用）和 Scouter 的配置（算理想下限用），两边描述的是同一个部署；
   · at：在导出的网格上插值（上下文、batch 都按对数坐标线性插值），所以页面上的滑块可以实时动；
   · compare / compareTrain：同一点上 Scouter 理想下限（Common.IDEAL）和 HuggingArch 仿真逐项对照，给能直接画图的结构化数据：
     瀑布（理想 → 记账差 → 逐算子串行 → launch 下限 → HuggingArch，逐项相加正好闭合）、资源表、按算子类别的时间、瓶颈分布。
   自定义模型（没有 HuggingArch spec）没有这部分，只有理想和 Scouter 自己的建模。 */
const Hga = (function(){
const { MODELS, HW, TOPO, IDEAL, freeze } = Common;

/* Scouter 模型 → HuggingArch 的 model id（snapshot 在 vendor/HuggingArch/backend/arch/snapshots） */
const REAL = freeze({ qwen35:"Qwen/Qwen3.5-35B-A3B", kimik3:"moonshotai/Kimi-K3", glm53f:"zai-org/GLM-5.3-Flash",
  dsv41f:"deepseek-ai/DeepSeek-V4.1-Flash", qwen38_27b:"Qwen/Qwen3.8-27B" });
/* Scouter GPU → HuggingArch gpu_specs.csv 的 SKU（RTX PRO 6000 不在 HuggingArch 的卡表里） */
const CARD = freeze({ H100:"H100-SXM", H200:"H200", B200:"B200", B300:"B300", GB200:"GB200", GB300:"GB300" });
/* 导出网格：推理是 prompt 长度 × 每个 DP rank 的 decode batch（输出固定 1024，decode 的上下文 = 输入 + 输出 / 2，和 HuggingArch 的约定一样）；
   训练是序列长度 × micro-batch（64 卡、每步 2M token）。prefill 不分块（一次 forward 整条 prompt），两边一样 */
const GRID = freeze({ isl:[1024,2048,4096,8192,16384,32768,65536,131072], osl:1024, b:[1,2,4,8,16,32,64,128,256,512],
  seq:[4096,8192,16384,32768], mbs:[1,2], world:64, gtok:2097152 });

/* 引擎签名。hga：HuggingArch ParallelConfig；cfg：Scouter Infer 配置（N 卡、dpA 是 attention DP、moePar 是 MoE 的切法）；
   moeOnly：dense 模型上和别的签名重复或没有意义，跳过；flags：引擎的启动参数（命令行，不翻译） */
const SIG = freeze({
  sglang:[
    { id:"tp8", gpus:8, hga:{ tp_attention:8, tp_dense_ffn:8, tp_moe_expert:8 }, cfg:{ N:8, dpA:1, moePar:"tp" }, flags:"--tp 8" },
    // TP attention 下每卡已有全部 token：EP 不派发，expert 输出并进 TP all-reduce（SGLang 缺省 --moe-a2a-backend none）。HuggingArch 按 token 布局推集合通信，
    // 这种布局下也只有 all-reduce；DeepEP 由 dpa8ep8 覆盖。Scouter 一侧用 ag（attention DP = 1 时通信和 tp 一样，见 infer.js 的 agLocal）
    { id:"tp8ep8", gpus:8, moeOnly:true, hga:{ tp_attention:8, tp_dense_ffn:8, ep_moe:8 }, cfg:{ N:8, dpA:1, moePar:"ag" }, flags:"--tp 8 --ep 8" },
    { id:"dpa8ep8", gpus:8, hga:{ dp_attention:8, ep_moe:8 }, cfg:{ N:8, dpA:8, moePar:"deepep" }, flags:"--tp 8 --dp 8 --enable-dp-attention --ep 8 --moe-a2a-backend deepep" },
    { id:"dpa16ep16", gpus:16, moeOnly:true, hga:{ dp_attention:16, ep_moe:16 }, cfg:{ N:16, dpA:16, moePar:"deepep" }, flags:"--tp 16 --dp 16 --enable-dp-attention --ep 16 --moe-a2a-backend deepep" },
    { id:"tp8dcp8", gpus:8, hga:{ tp_attention:8, tp_dense_ffn:8, tp_moe_expert:8, dcp:8 }, cfg:{ N:8, dpA:1, moePar:"tp", dcp:8 }, flags:"--tp 8 --dcp-size 8" } ],
  vllm:[
    { id:"tp8", gpus:8, hga:{ tp_attention:8, tp_dense_ffn:8, tp_moe_expert:8 }, cfg:{ N:8, dpA:1, moePar:"tp" }, flags:"-tp 8" },
    { id:"tp8ep8", gpus:8, moeOnly:true, hga:{ tp_attention:8, tp_dense_ffn:8, ep_moe:8 }, cfg:{ N:8, dpA:1, moePar:"ag" }, flags:"-tp 8 --enable-expert-parallel" },
    { id:"dp8ep8", gpus:8, hga:{ dp_attention:8, ep_moe:8 }, cfg:{ N:8, dpA:8, moePar:"deepep" }, flags:"-dp 8 --enable-expert-parallel --all2all-backend deepep_low_latency" },
    { id:"dp16ep16", gpus:16, moeOnly:true, hga:{ dp_attention:16, ep_moe:16 }, cfg:{ N:16, dpA:16, moePar:"deepep" }, flags:"-dp 16 --enable-expert-parallel --all2all-backend deepep_low_latency" },
    { id:"tp8dcp8", gpus:8, hga:{ tp_attention:8, tp_dense_ffn:8, tp_moe_expert:8, dcp:8 }, cfg:{ N:8, dpA:1, moePar:"tp", dcp:8 }, flags:"-tp 8 -dcp 8" } ],
  /* 训练：64 卡，TrainingConfig（vendor/HuggingArch/backend/training/step.py）和 Scouter Train 的配置 */
  megatron:[
    { id:"tp8sp", train:{ tp:8, sequence_parallel:true, recompute:"selective" }, cfg:{ tp:8, ep:1, recompute:"selective" }, flags:"--tensor-model-parallel-size 8 --sequence-parallel --recompute-granularity selective --use-distributed-optimizer",
      moe:{ train:{ ep:8 }, cfg:{ ep:8, etp:1 }, flags:"--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --moe-token-dispatcher-type alltoall" } },
    { id:"tp4pp2", train:{ tp:4, pp:2, sequence_parallel:true, recompute:"selective" }, cfg:{ tp:4, ep:1, pp:2, recompute:"selective" }, flags:"--tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --sequence-parallel --recompute-granularity selective --use-distributed-optimizer",
      moe:{ train:{ ep:8 }, cfg:{ ep:8, etp:1 }, flags:"--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --moe-token-dispatcher-type alltoall" } },
    { id:"ep8", moeOnly:true, train:{ ep:8, recompute:"selective" }, cfg:{ tp:1, ep:8, etp:1, recompute:"selective" }, flags:"--expert-model-parallel-size 8 --moe-token-dispatcher-type alltoall --use-distributed-optimizer" },
    { id:"tp2ep8pp4", moeOnly:true, train:{ tp:2, ep:8, pp:4, sequence_parallel:true, recompute:"full" }, cfg:{ tp:2, ep:8, etp:1, pp:4, recompute:"full" }, flags:"--tensor-model-parallel-size 2 --expert-model-parallel-size 8 --pipeline-model-parallel-size 4 --sequence-parallel --recompute-granularity full" },
    { id:"tp2cp4", train:{ tp:2, cp:4, ep:4, etp:2, recompute:"full" }, cfg:{ tp:2, cp:4, ep:4, etp:2, recompute:"full" }, flags:"--tensor-model-parallel-size 2 --context-parallel-size 4 --expert-model-parallel-size 4 --recompute-granularity full" } ],
  torchtitan:[
    { id:"fsdp", train:{ fsdp:true, recompute:"full" }, cfg:{ tp:1, ep:1, recompute:"full" }, flags:"--parallelism.data_parallel_shard_degree 64 --activation_checkpoint.mode full" },
    { id:"hsdp", train:{ fsdp:true, hsdp_replicate:8, recompute:"full" }, cfg:{ tp:1, ep:1, hsdp:8, recompute:"full" }, flags:"--parallelism.data_parallel_replicate_degree 8 --parallelism.data_parallel_shard_degree 8 --activation_checkpoint.mode full" },
    { id:"fsdp_tp8", train:{ fsdp:true, tp:8, sequence_parallel:true, recompute:"selective" }, cfg:{ tp:8, ep:1, recompute:"default" }, flags:"--parallelism.tensor_parallel_degree 8 --activation_checkpoint.mode selective",
      moe:{ train:{ ep:8 }, cfg:{ ep:8 }, flags:"--parallelism.expert_parallel_degree 8 --parallelism.expert_tensor_parallel_degree 1" } },
    { id:"fsdp_ep8", moeOnly:true, train:{ fsdp:true, ep:8, recompute:"full" }, cfg:{ tp:1, ep:8, recompute:"full" }, flags:"--parallelism.expert_parallel_degree 8 --activation_checkpoint.mode full" },
    { id:"fsdp_cp4", train:{ fsdp:true, cp:4, ep:4, recompute:"full" }, cfg:{ tp:1, cp:4, ep:4, recompute:"full" }, flags:"--parallelism.context_parallel_degree 4 --parallelism.expert_parallel_degree 4 --activation_checkpoint.mode full" } ] });
const TRAIN = ["megatron", "torchtitan"], INFER = ["sglang", "vllm"];
/* 导出的点是平铺数组（tools/hga/export.py 的 aggregate）：前 11 个是阶段总量，接着 4 个是按瓶颈分的时间，再按 ROLES 每类 4 个（t、计算、HBM、通信）。
   类别：attention（含线性注意力）、routed MoE（含 latent 投影）、dense FFN 和 shared expert、router、其余（norm、embedding、LM head、残差混合…）、通信 */
const ROLES = ["attn", "moe", "dense", "router", "other", "comm"];
const POINT = ["t", "tc", "tm", "tn", "s0", "fl", "by", "wb", "kv", "cb", "nl"];
function unpack(a){
  if(!a) return null;
  const o = {}; POINT.forEach((k, i) => { o[k] = a[i]; });
  o.bb = a.slice(11, 15); o.role = {};
  ROLES.forEach((k, j) => { o.role[k] = a.slice(15 + 4*j, 19 + 4*j); });
  return o;
}
/* 训练的一趟（fwd / bwd / 重算）：t 计算 HBM 通信 launch FLOPs 字节 通信字节 launch 数，再按 ROLES 每类的时间 */
const PASS = ["t", "compute", "memory", "comm", "launch", "flops", "bytes", "commBytes", "launches"];
function unpackPass(a){ const o = {}; PASS.forEach((k, i) => { o[k] = a[i]; }); o.byRole = Object.fromEntries(ROLES.map((k, j) => [k, a[9 + j]])); return o; }
const isReal = mk => typeof mk === "string" && mk in REAL;
/* 签名里的 moe 一项只对有 routed expert 的模型生效：expert 放在 EP 上（ETP 1），两边描述同一个部署。
   不放 ETP 是因为 Megatron 的 alltoall 派发在 ETP 组里 all-gather 的是按 top-k 复制过的行，HuggingArch 的 walk 只有 hidden 的 all-reduce，
   torchtitan 在 EP 关掉时按 TP 切 expert，Scouter 的 torchtitan 也没有这一种；dense 模型照原样 */
const isMoe = mk => !!(MODELS[mk] && MODELS[mk].moe && MODELS[mk].moe.E > 0);
const resolve = (s, mk) => !s || !s.moe || !isMoe(mk) ? s
  : freeze(Object.assign({}, s, { train:Object.assign({}, s.train, s.moe.train), cfg:Object.assign({}, s.cfg, s.moe.cfg), flags:s.flags + " " + s.moe.flags }));
const sigsFor = (engine, mk) => (SIG[engine] || []).filter(s => !s.moeOnly || !MODELS[mk] || isMoe(mk)).map(s => resolve(s, mk));
const sigOf = (engine, id, mk) => resolve((SIG[engine] || []).find(s => s.id === id) || null, mk);

/* ---- 网格插值：上下文、batch（或序列、micro-batch）都按对数坐标，夹在网格两端 ---- */
function bracket(xs, x){
  const lx = Math.log(Math.max(x, 1e-12));
  if(x <= xs[0]) return [0, 0, 0];
  if(x >= xs[xs.length-1]) return [xs.length-1, xs.length-1, 0];
  let i = 0; while(xs[i+1] < x) i++;
  const a = Math.log(xs[i]), b = Math.log(xs[i+1]);
  return [i, i+1, (lx - a)/(b - a)];
}
/* 两个点（同样结构的对象 / 数组 / 数）按 w 混合；时间、字节、FLOPs 都在对数坐标里插（它们随 batch、上下文大致是幂律），0 和负数按线性 */
function mix(p, q, w){
  if(w === 0 || p === q) return p;
  if(typeof p === "number" && typeof q === "number"){ if(p > 0 && q > 0) return Math.exp(Math.log(p)*(1-w) + Math.log(q)*w); return p*(1-w) + q*w; }
  if(Array.isArray(p)) return p.map((v, i) => mix(v, q[i], w));
  if(p && typeof p === "object"){ const o = {}; for(const k of Object.keys(p)) o[k] = q && k in q ? mix(p[k], q[k], w) : p[k]; return o; }
  return w < 0.5 ? p : q;
}
/* 推理：一个格子（cell = data.infer[engine.sig][card]）在 prompt 长度 isl、每 rank batch b 上的 prefill 和 decode */
function at(cell, isl, b){
  if(!cell || !cell.ok) return null;
  const [i0, i1, wi] = bracket(cell.isl, isl), [j0, j1, wj] = bracket(cell.b, b);
  const dec = (i, j) => cell.dec[i][j];
  if([dec(i0,j0), dec(i0,j1), dec(i1,j0), dec(i1,j1), cell.pre[i0], cell.pre[i1]].some(x => !x)) return null;   // 这个上下文一条请求也放不下
  const d = unpack(mix(mix(dec(i0, j0), dec(i0, j1), wj), mix(dec(i1, j0), dec(i1, j1), wj), wi));
  const pre = unpack(mix(cell.pre[i0], cell.pre[i1], wi));
  const maxB = mix(cell.maxB[i0], cell.maxB[i1], wi);
  return { isl, b, decode:d, prefill:pre, maxB, exact: cell.isl.includes(isl) && cell.b.includes(b) };
}
/* 训练：cell = data.train[fw.sig][card]，序列长度 seq、micro-batch mbs，tr 是签名的 TrainingConfig（tp / cp / pp / fsdp / overlap_dp）。
   格点上原样返回。格点之间只插每个 micro-batch 的量（fwd / bwd / 重算各项、每个 micro-batch 的 FSDP 通信、每个在途 micro-batch 的激活、
   bubble 占 micro-batch 的比例），再按导出的公式（vendor/HuggingArch/backend/training/step.py）用整数个 micro-batch 重新拼一步：
   m = ceil(全局 token / (DP × seq × mbs))，t_pipe = m · t_mb + bubble，FSDP 的 all-gather / reduce-scatter 每个 micro-batch 一次、
   窗口是 m 个 micro-batch 的计算，其余 DP 通信和优化器每步一次。整步直接插会把 m 也插成小数（两个格点的 m 不同），和理想下限的整数 m 对不上 */
const FSDP_PART = /^fsdp_/;
function atTrain(cell, seq, mbs, tr){
  if(!cell || !cell.ok) return null;
  const si = cell.seq.indexOf(seq), mi = cell.mbs.indexOf(mbs);
  if(si >= 0 && mi >= 0) return Object.assign({}, cell.pts[si][mi], { seq, mbs, exact:true });
  const pp = tr.pp || 1;
  const unit = q => { const m = q.st[0], tmb = q.f[0] + q.b[0] + q.r[0];
    return { f:q.f, b:q.b, r:q.r, opt:q.st[6], bub:tmb > 0 ? q.st[2]/tmb : 0, peakInv:q.st[8] > 0 && q.f[5] > 0 ? q.st[8]*q.st[5]/(3*q.f[5]*m) : 0,
      dp:q.dp.map(x => FSDP_PART.test(x[0]) ? [x[0], x[1], x[2], x[3]/m, x[4]/m] : x), mem:[q.mem[0], q.mem[1], q.mem[2], q.mem[3]/Math.min(pp, m), q.mem[5]] }; };
  const [i0, i1, wi] = bracket(cell.seq, seq), [j0, j1, wj] = bracket(cell.mbs, mbs);
  const p = (i, j) => unit(cell.pts[i][j]);
  const u = mix(mix(p(i0, j0), p(i0, j1), wj), mix(p(i1, j0), p(i1, j1), wj), wi);
  const dp = Math.max(1, Math.floor(GRID.world / ((tr.tp || 1) * (tr.cp || 1) * pp))), m = Math.max(1, Math.ceil(GRID.gtok / (dp * seq * mbs)));
  const tmb = u.f[0] + u.b[0] + u.r[0], bubble = pp > 1 ? u.bub * tmb : 0, tPipe = m * tmb + bubble;
  const parts = u.dp.map(x => FSDP_PART.test(x[0]) ? [x[0], x[1], x[2], x[3]*m, x[4]*m] : x), serial = parts.reduce((s, x) => s + x[4], 0);
  const hide = (tr.fsdp ? m : 1) * (u.f[1] + u.b[1]), exposed = tr.overlap_dp === false ? serial : Math.max(0, serial - hide);   // 能藏住 DP 通信的计算窗口
  const t = tPipe + exposed + u.opt, act = u.mem[3] * Math.min(pp, m);
  return { f:u.f, b:u.b, r:u.r, dp:parts, seq, mbs, exact:false,
    st:[m, tPipe, bubble, serial, exposed, t, u.opt, GRID.gtok / (t * GRID.world), 3 * u.f[5] * m * u.peakInv / t],
    mem:[u.mem[0], u.mem[1], u.mem[2], act, u.mem[0] + u.mem[1] + u.mem[2] + act, u.mem[4]] };
}

/* ---- Scouter 的理想下限：同一个签名、同一个点 ---- */
/* 精度用模型的出厂格式（和 HuggingArch 的 as_shipped 一致：权重、dense、KV 的默认精度和 expert 激活的出厂格式 eActShipped 写在 Common.MODELS） */
const shipped = m => { const f8 = m.wDefault === "fp8"; return { wDt:m.wDefault, dDt:m.dDefault||(f8?"fp8":"bf16"), kvDt:m.kvDefault||(f8?"fp8":"bf16"), stateDt:"fp32", eAct:m.eActShipped||"" }; };
/* Scouter 的 op（推理 Infer 和训练 Train 的 key）归到 HuggingArch 的类别：routed MoE 含 latent 投影和 permute（训练的 act 是 expert 的激活，先判），
   dense FFN 含 shared expert；a* / l* 是 attention 和线性注意力（lm 除外）；其余是 norm、embedding、LM head、CE、采样 */
const MOE_KEYS = new Set(["moe", "moeq", "latd", "latu", "moe1", "moe2", "act", "permute", "unperm"]);
const ROLE_OF = k => MOE_KEYS.has(k) ? "moe" : k === "shared" || k === "dense" ? "dense" : k === "router" ? "router"
  : /^(a|l)[a-z]*$/.test(k) && k !== "lm" ? "attn" : "other";
function idealSide(r){
  const byRole = Object.fromEntries(ROLES.map(k => [k, 0]));
  let flops = 0, bytes = 0, kernels = 0, tc = 0, tm = 0;
  for(const o of r.ops){ byRole[ROLE_OF(o.key)] += o.t; flops += o.flops; bytes += o.bytes; kernels += o.kernels||0; tc += o.tc; tm += o.tm; }
  let tn = 0, cb = 0;
  for(const x of r.comm){ byRole.comm += x.t; tn += x.t; cb += x.bytes||0; }
  const b = r.bound || {};
  return { t:r.stepT, compute:tc, memory:tm, comm:tn, launch:0, path:b.path ?? null, by:b.by || null, flops, bytes, commBytes:cb, kernels:kernels + r.comm.length, byRole };
}
function idealInfer(mk, hwKey, engine, sig, isl, osl, b){
  const m = MODELS[mk], hw = HW[hwKey], topo = TOPO[Common.homeTopo(hwKey)];
  const base = Object.assign({}, Infer.BASE, shipped(m), sig.cfg, { engine, isl, osl, C:isl });   // prefill 不分块，和导出（prefill_chunk_size = None）一致
  const dec = Infer.decode(m, hw, topo, Object.assign({}, base, { phase:"decode", B:b, n:isl + osl/2 }), IDEAL);
  const pre = Infer.prefill(m, hw, topo, Object.assign({}, base, { phase:"prefill", B:1, P:isl, p:0 }), IDEAL);
  return { decode: Object.assign(idealSide(dec), { tokPerGpu:dec.tokPerGpu, Bmax:dec.mem.Bmax, warn:dec.warn }),
           prefill: Object.assign(idealSide(pre), { t:pre.ttft, ttft:pre.ttft }) };
}

/* ---- 对照：一个阶段 ----
   HuggingArch 的步长 H = Σ_算子 max(计算, HBM, 通信, launch)：每个算子各按自己更紧的那一项，算子之间不重叠（docs/inference_cost_model.md）。
   理想下限 I = 关键路径和资源守恒里最大的那个（计算、HBM、通信各自的总量都不能被藏掉）。
   瀑布把 H − I 拆成三段，逐项相加正好闭合：
     acc    记账差：max(ΣC, ΣM, ΣN)_HGA − I。两边数的 FLOPs、字节、通信量不一样（HuggingArch 逐算子数激活读写、融合以后剩下的中间张量、
            通信按 α-β 和具体算法；Scouter 理想只数必要的权重 / KV / 激活和信息量下限）
     ser    逐算子串行：Σ max(c, m, n) − max(ΣC, ΣM, ΣN)。HuggingArch 不让一个算子的计算藏在另一个算子的访存或通信后面；理想按资源守恒全部重叠
     launch launch 下限：H − Σ max(c, m, n)。一次 kernel 至少 1 µs（HuggingArch defaults.py 的 LAUNCH_FLOOR_US_DEFAULT），decode 小 kernel 多时显著 */
function phaseCmp(ideal, h){
  const R = Math.max(h.tc, h.tm, h.tn), s0 = h.s0 ?? R;
  const I = ideal.t, H = h.t;
  const hga = { t:H, compute:h.tc, memory:h.tm, comm:h.tn, launch:Math.max(0, H - s0), s0, flops:h.fl, bytes:h.by, weightBytes:h.wb, kvBytes:h.kv, commBytes:h.cb,
    kernels:h.nl, byBound:{ compute:h.bb[0], memory:h.bb[1], comm:h.bb[2], launch:h.bb[3] },
    byRole:Object.fromEntries(ROLES.map(k => [k, h.role && h.role[k] ? h.role[k][0] : 0])),
    byRoleBound:Object.fromEntries(ROLES.map(k => [k, h.role && h.role[k] ? { compute:h.role[k][1], memory:h.role[k][2], comm:h.role[k][3] } : null])) };
  const steps = [ { key:"ideal", t:I }, { key:"acc", d:R - I }, { key:"ser", d:s0 - R }, { key:"launch", d:H - s0 }, { key:"hga", t:H } ];
  const res = { compute:[ideal.compute, hga.compute], memory:[ideal.memory, hga.memory], comm:[ideal.comm, hga.comm], flops:[ideal.flops, hga.flops],
    bytes:[ideal.bytes, hga.bytes], commBytes:[ideal.commBytes, hga.commBytes], kernels:[ideal.kernels, hga.kernels] };
  const roles = ROLES.map(k => ({ role:k, ideal:ideal.byRole[k], hga:hga.byRole[k] })).filter(x => x.ideal > 0 || x.hga > 0);
  return { ideal, hga, waterfall:steps, resources:res, roles, ratio: I > 0 ? H/I : null };
}

/* 推理对照：data = 这个模型的导出（src/data/hga/<model>.js），返回能直接画的结构 */
function compare(data, q){
  const mk = q.model, engine = q.engine, sig = sigOf(engine, q.sig, mk), card = q.hw;
  if(!sig) throw new Error('signature "' + q.sig + '" not found for ' + engine);
  const cell = data && data.infer && data.infer[engine + "." + sig.id] && data.infer[engine + "." + sig.id][card];
  const out = { model:mk, hf:REAL[mk], engine, sig:sig.id, flags:sig.flags, hw:card, isl:q.isl, osl:GRID.osl, b:q.b, parallel:sig.hga };
  if(!cell) return Object.assign(out, { ok:false, reason:"missing" });
  if(!cell.ok) return Object.assign(out, { ok:false, reason:cell.reason || "refused" });
  const p = at(cell, q.isl, q.b);
  if(!p) return Object.assign(out, { ok:false, reason:"capacity" });
  const I = idealInfer(mk, card, engine, sig, q.isl, GRID.osl, q.b);
  const tokH = q.b * (sig.hga.dp_attention || 1) / sig.gpus / p.decode.t;
  return Object.assign(out, { ok:true, exact:p.exact, maxB:{ hga:p.maxB, ideal:I.decode.Bmax }, fits:{ hga:q.b <= p.maxB, ideal:q.b <= I.decode.Bmax },
    decode:phaseCmp(I.decode, p.decode), prefill:phaseCmp(I.prefill, p.prefill),
    throughput:{ ideal:I.decode.tokPerGpu, hga:tokH }, warn:I.decode.warn, hgaWarnings:cell.warn || [] });
}
/* 沿一个轴扫：axis = "b" 或 "isl"，返回每个网格点的理想 / HuggingArch 步长、吞吐（画曲线用） */
function curve(data, q){
  const xs = q.axis === "isl" ? GRID.isl : GRID.b;
  return xs.map(x => { const r = compare(data, Object.assign({}, q, q.axis === "isl" ? { isl:x } : { b:x }));
    return r.ok ? { x, decode:{ ideal:r.decode.ideal.t, hga:r.decode.hga.t }, prefill:{ ideal:r.prefill.ideal.t, hga:r.prefill.hga.t },
      tok:r.throughput, fits:r.fits, maxB:r.maxB } : { x, ok:false, reason:r.reason }; });
}

/* ---- 训练对照 ---- */
/* Scouter 的训练 op 和通信按 HuggingArch 的类别（ROLES，归类同 ROLE_OF）分开，一个 micro-batch、一个 stage、每趟一份（fwd / bwd / 重算）；
   通信只算露出来的那部分（Train.run 的 x.exp，按每趟的份额摊开），HuggingArch 那边的通信本来就是串行的 */
function trainRoles(r){
  const pp = r.pp || 1, z = () => Object.fromEntries(ROLES.map(k => [k, 0])), out = { fwd:z(), bwd:z(), rc:z() };
  for(const o of r.ops){ const k = ROLE_OF(o.key);
    out.fwd[k] += o.fwd.t/pp; out.bwd[k] += (o.dgrad.t + o.wgrad.t)/pp; if(o.rc) out.rc[k] += o.fwd.t/pp; }
  const raw = { fwd:0, bwd:0, rc:0 };
  for(const x of r.comm){ if(x.perStage || !(x.t > 0)) continue; const share = (x.exp == null ? x.t : x.exp)/x.t;
    for(const e of x.xs || []) raw[e.pass === "bwd" || e.pass === "rc" ? e.pass : "fwd"] += e.t*x.layers*share/pp; }
  const tot = raw.fwd + raw.bwd + raw.rc, k = tot > 0 ? r.exposed/pp/tot : 0;     // 露出的总量以 Train.run 为准（shared expert 的重叠还要再扣）
  for(const p of ["fwd", "bwd", "rc"]) out[p].comm += raw[p]*k;
  /* micro-batch 里另外几项，都记在 forward：FSDP 藏不住的 gather（第一层的 gather 在 forward 之前）和 PP 的点对点算通信；
     PP 时一段至少要跑完 LM head，这个下限比平均分的 stage 多出来的部分算“其余” */
  const fsdpX = r.fsdpExposedMB || 0;
  out.fwd.comm += fsdpX + (r.tMB - r.tMBstage - fsdpX);
  out.fwd.other += r.tMBstage - (r.compMB + r.exposed)/pp;
  return out;
}
function idealTrain(mk, hwKey, fw, sig, seq, mbs){
  const m = MODELS[mk], hw = HW[hwKey], topo = TOPO[Common.homeTopo(hwKey)];
  // Train 的 tok 是每卡每个 micro-batch 的 token（界面上的 max_tokens_per_gpu）：CP 把 mbs 条长 seq 的序列切到 cp 个 rank 上，每卡 seq·mbs/cp；
  // HuggingArch 的 micro-batch 是整条 seq·mbs（TrainingConfig 的 tokens），两边的 DP 份数和 micro-batch 个数才对得上
  const c = Object.assign({}, Train.BASE, Train.FW_DEF[fw] || {}, { fw, N:GRID.world, offload:false, S:seq, Gtok:GRID.gtok }, sig.cfg);
  c.tok = seq*mbs/(c.cp || 1);
  if(fw === "megatron") c.ep = sig.cfg.ep || 1;
  const r = Train.run(m, hw, topo, c, IDEAL);
  const mb = r.tMB, m0 = r.microsPerStep;
  return { t:r.stepT, micro:mb, m:m0, pipe:r.tPipe, bubble:r.tPipe - mb*m0, dp:r.tDPexp, dpSerial:r.tDP, opt:r.tOpt + (r.tSmallAR||0),
    fwd:r.tF/r.pp, bwd:(r.tD + r.tW)/r.pp, rc:(r.tR||0)/r.pp, mfu:r.mfu, tokPerGpu:r.tokPerGpu, mem:r.mem.used, fits:r.mem.used <= r.mem.total, warn:r.warn,
    roles:trainRoles(r) };
}
function compareTrain(data, q){
  const mk = q.model, fw = q.engine, sig = sigOf(fw, q.sig, mk), card = q.hw;
  if(!sig) throw new Error('signature "' + q.sig + '" not found for ' + fw);
  const cell = data && data.train && data.train[fw + "." + sig.id] && data.train[fw + "." + sig.id][card];
  const out = { model:mk, hf:REAL[mk], engine:fw, sig:sig.id, flags:sig.flags, hw:card, seq:q.seq, mbs:q.mbs, world:GRID.world, gtok:GRID.gtok, config:sig.train };
  if(!cell) return Object.assign(out, { ok:false, reason:"missing" });
  if(!cell.ok) return Object.assign(out, { ok:false, reason:cell.reason || "refused" });
  const p = atTrain(cell, q.seq, q.mbs, sig.train), I = idealTrain(mk, card, fw, sig, q.seq, q.mbs);
  const H = { t:p.st[5], micro:p.f[0] + p.b[0] + p.r[0], m:p.st[0], pipe:p.st[1], bubble:p.st[2], dp:p.st[4], dpSerial:p.st[3], opt:p.st[6],
    fwd:p.f[0], bwd:p.b[0], rc:p.r[0], mfu:p.st[8], tokPerGpu:p.st[7], mem:p.mem[4], fits:p.mem[4] <= p.mem[5],
    memParts:{ weights:p.mem[0], grads:p.mem[1], optimizer:p.mem[2], activations:p.mem[3] },
    pass:{ fwd:unpackPass(p.f), bwd:unpackPass(p.b), rc:unpackPass(p.r) }, dpParts:p.dp };
  /* 瀑布：理想步长 → micro-batch 本身（m × 每个 micro-batch 的 fwd + bwd + 重算）→ PP bubble → 露出来的 DP 通信 → 优化器 → HuggingArch */
  const steps = [ { key:"ideal", t:I.t }, { key:"micro", d:H.micro*H.m - I.micro*I.m }, { key:"bubble", d:H.bubble - I.bubble }, { key:"dp", d:H.dp - I.dp },
    { key:"opt", d:H.opt - I.opt }, { key:"hga", t:H.t } ];
  /* 导出取三位有效数字，HuggingArch 的 t_step = t_pipe + 露出的 DP + 优化器 在舍入后差一点点：0.5% 以内并进 micro-batch 那一项，再大就单列“其余” */
  const rest = H.t - (I.t + steps.slice(1, 5).reduce((s, x) => s + x.d, 0));
  if(Math.abs(rest) > 5e-3*H.t) steps.splice(5, 0, { key:"other", d:rest }); else steps[1].d += rest;
  return Object.assign(out, { ok:true, exact:p.exact, ideal:I, hga:H, waterfall:steps, ratio: I.t > 0 ? H.t/I.t : null,
    passes:[ { key:"fwd", ideal:I.fwd, hga:H.fwd }, { key:"bwd", ideal:I.bwd, hga:H.bwd }, { key:"rc", ideal:I.rc, hga:H.rc } ],
    roles:ROLES.map((k, j) => { const side = (f, b, rc) => ({ fwd:f, bwd:b, rc, t:f + b + rc });
      return { role:k, ideal:side(I.roles.fwd[k], I.roles.bwd[k], I.roles.rc[k]), hga:side(p.f[9 + j] || 0, p.b[9 + j] || 0, p.r[9 + j] || 0) }; })
      .filter(x => x.ideal.t > 0 || x.hga.t > 0),
    warn:I.warn, hgaWarnings:cell.warn || [] });
}
function curveTrain(data, q){
  return GRID.seq.map(seq => { const r = compareTrain(data, Object.assign({}, q, { seq }));
    return r.ok ? { x:seq, ideal:r.ideal.t, hga:r.hga.t, tok:{ ideal:r.ideal.tokPerGpu, hga:r.hga.tokPerGpu }, mfu:{ ideal:r.ideal.mfu, hga:r.hga.mfu } } : { x:seq, ok:false, reason:r.reason }; });
}

return { REAL, CARD, GRID, SIG, TRAIN, INFER, ROLES, POINT, PASS, isReal, sigsFor, sigOf, at, atTrain, compare, curve, compareTrain, curveTrain, idealInfer, idealTrain };
})();
