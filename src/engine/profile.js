/* 线上 profile（其他 · Profile 页）：一步的时间按线上 trace 的分法拆开，观测本身要多少开销，DP / EP 负载不均吃掉多少。
   只拆 Infer.step 的结果，不另写物理；唯一新加的配置是 dpSkew。分法对照 susun 的 CUPTI HES 博客和 pegainfer-project/kern：
   · kern 的三类：compute、comm（NCCL 或用 peer buffer 的 kernel）、idle（窗口里减掉 GPU 忙的时间）；
   · 博客 DuckDB 查询里的几类：MoE、attention、attention 投影、indexer、mHC、其他、MoE 里的等待（waiting in MoE）、步与步之间。
   这里的数都是"平均一个 rank 每步"的秒数。 */
const Profile = (function(){
const DEF = Object.freeze({ dpSkew:1.15 });
/* 类别的顺序也是画图的顺序。kern：这一类在 kern 的三类里算哪一类 */
const CATS = ["moe", "eplb", "attn", "proj", "idx", "mhc", "lin", "mlp", "other", "draft", "comm", "wait", "idle"];
const KERN = { moe:"compute", eplb:"compute", attn:"compute", proj:"compute", idx:"compute", mhc:"compute", lin:"compute", mlp:"compute",
  other:"compute", draft:"compute", comm:"comm", wait:"comm", idle:"idle" };
/* Infer 的算子 → 类别。attention 核心（含 CSA 的压缩）单列；attention 组里其余的是投影、norm 和 gate（博客的 "attention projections"） */
const catOf = o => o.key==="mhc" ? "mhc" : /^aidx/.test(o.key) ? "idx" : (o.key==="acore" || o.key==="acmp") ? "attn"
  : o.group==="attn" ? "proj" : o.group==="lin" ? "lin" : o.key==="dense" ? "mlp" : o.group==="moe" ? "moe" : "other";
/* MoE 之前、各 DP rank 各算各的那一段（DP attention 的 rank 在 MoE 的通信里会合） */
const SIDE = ["attn", "proj", "idx", "mhc", "lin"];
const moeT = r => r.ops.filter(o => o.key==="moe").reduce((s, o) => s + o.t, 0);

/* EPLB：Infer 按最忙的那张卡算 MoE（行数 × imb），用 imb = 1 再算一遍，差就是专家负载不均多出来的时间。
   DPLB：dpSkew = 最慢的 DP rank 的 attention 侧工作量 ÷ 平均。MoE 的通信要等所有 rank 到齐，平均每个 rank 等 (dpSkew − 1) × attention 侧时间；
   这一段在 trace 里算在通信 kernel 里（博客的 "waiting in MoE" 就是同一次调用的耗时减去各 rank 里最短的那个），一步也跟着变长 */
function split(m, hw, topo, c, g, p){
  const P = Object.assign({}, DEF, p||{}), r = Infer.step(m, hw, topo, c, g), o = Object.fromEntries(CATS.map(k => [k, 0]));
  for(const x of r.ops) if(!x.overlap) o[catOf(x)] += x.t;
  o.moe += r.exposedOverlap;                                   // 共享 expert 和融合的 a2a 露在 routed expert 外面的部分
  o.draft = (r.draftT||0) + (r.commitT||0);
  o.comm = r.tComm + r.exposedTbo;
  const ep = m.moe.E>0 && c.N>1 && r.moePar!=="tp" && !g.lb && (c.imb||1)>1;
  if(ep){ const r1 = Infer.step(m, hw, topo, Object.assign({}, c, { imb:1 }), g); o.eplb = Math.max(0, Math.min(o.moe, moeT(r) - moeT(r1))); o.moe -= o.eplb; }
  const busy = CATS.filter(k => KERN[k]!=="idle" && k!=="wait").reduce((s, k) => s + o[k], 0);
  o.idle = Math.max(0, r.stepT - busy);                        // graph 里 kernel 之间的空隙、graph launch、CPU 发射、DP 同步、露出来的 Engram 拷贝
  const skew = Math.max(1, +P.dpSkew || 1), dp = r.dpA>1 && !g.lb;
  o.wait = dp ? (skew - 1)*SIDE.reduce((s, k) => s + o[k], 0) : 0;
  const total = Math.max(r.stepT, busy) + o.wait;
  const kern = { compute:0, comm:0, idle:0 };
  for(const k of CATS) kern[KERN[k]] += o[k];
  return { r, total, stepT:r.stepT, dp, ep, skew, nKern:r.nKern, graph:r.graph,
    cats:CATS.filter(k => o[k]>0).map(k => ({ key:k, t:o[k], frac:o[k]/total, kern:KERN[k] })),
    kern:Object.fromEntries(Object.entries(kern).map(([k, v]) => [k, { t:v, frac:v/total }])),
    dpLoss:o.wait/total, eplbLoss:o.eplb/total };
}

/* 观测的开销：每步的 kernel 数（开 CUDA graph 时就是节点数）× 每个节点的观测开销。这是上限：kernel 长、GPU 跑在 CPU 前面时，
   一部分开销藏在后面（博客里 Qwen3-4B 并发 1 时每步多 115 µs，并发 48 时多 64 µs） */
function observer(sp, usPerNode){
  const t = sp.nKern*usPerNode*1e-6;
  return { nodes:sp.nKern, t, frac:t/sp.total };
}

const inert = (m, c) => { const o = {}; if(!((c.dpA||1)>1)) o.dpSkew = "noDp"; return o; };

return { DEF, CATS, KERN, catOf, split, observer, inert };
})();
