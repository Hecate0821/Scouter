/* 公共层：模型结构、硬件、拓扑、参数推导、roofline 与集合通信的基础函数。推理和训练各自的算子图在 infer.js / train.js。 */
const Common = (function(){

/* 每个元素的字节数（含块 scale）：fp8 是 block FP8 权重，每 128×128 一个 fp32 scale；fp8a 是 block FP8 激活，每 1×128 一个 fp32；
   mxfp8 / mxfp4 每 32 个一个 e8m0；nvfp4 每 16 个一个 e4m3（第二级 fp32 scale 另算） */
const DT = { fp32:4, bf16:2, fp8:1+4/16384, fp8a:1+4/128, mxfp8:1+1/32, mxfp4:0.53125, nvfp4:0.5625 };
const dtB = d => DT[d] ?? 2;
const peakKey = d => d==="fp32"?"bf16":d==="fp4"||d==="mxfp4"||d==="nvfp4"?"fp4":d==="fp8"||d==="fp8a"||d==="mxfp8"?"fp8":"bf16";

/* 结构字段来自 HF config 与三个引擎 / 两个训练框架的模型文件（交叉核对见“口径与来源”）。 */
const MODELS = {
  qwen35: { name:"Qwen3.5-35B-A3B", short:"Qwen3.5-35B", family:"qwen", D:2048, L:40, V:248320, mtp:1,
    full:{ n:10, kind:"gqa", Hq:16, Hkv:2, dh:256, gate:true },
    lin:{ n:30, kind:"GDN", Hk:16, Hv:32, dk:128, dv:128, conv:4 },
    moe:{ E:256, k:8, F:512, Fs:512, sharedGate:true, Dlat:0 }, dense:{ n:0, F:0 },
    wDefault:"bf16", official:Msg.of("engine_common.model.qwen35.official"),
    note:Msg.of("engine_common.model.qwen35.note") },
  kimik3: { name:"Kimi K3", short:"Kimi K3", family:"kimi", D:7168, L:93, V:163840, mtp:0,
    full:{ n:24, kind:"mla", H:96, qLora:1536, kvLora:512, nope:128, rope:64, vh:128, gate:true, nope_only:true },
    lin:{ n:69, kind:"KDA", Hk:96, Hv:96, dk:128, dv:128, conv:4 },
    moe:{ E:896, k:16, F:3072, Fs:6144, sharedGate:false, Dlat:3584 }, dense:{ n:1, F:33792 },
    attnRes:8, wDefault:"mxfp4", official:Msg.of("engine_common.model.kimik3.official"),
    note:Msg.of("engine_common.model.kimik3.note") },
  glm53f: { name:"GLM-5.3-Flash", short:"GLM-5.3-Flash", family:"glm", D:4096, L:45, V:154880, mtp:1,
    full:{ n:11, kind:"dsa", H:64, qLora:1536, kvLora:512, nope:256, rope:0, vh:256, gate:false, topk:2048, idxH:32, idxD:128, kpool:4 },
    lin:{ n:34, kind:"KDA", Hk:64, Hv:64, dk:128, dv:128, conv:4 },
    moe:{ E:288, k:8, F:2048, Fs:2048, sharedGate:false, Dlat:0 }, dense:{ n:3, F:12288 },
    mhc:4, wDefault:"fp8", official:Msg.of("engine_common.model.glm53f.official"),
    note:Msg.of("engine_common.model.glm53f.note") },
  musespark: { name:Msg.of("engine_common.model.musespark.name"), short:"Muse Spark", family:"muse", D:8192, L:62, V:202048, mtp:0,
    full:{ n:62, kind:"gqa", Hq:128, Hkv:16, dh:64, gate:true, gateHead:true, swa:46, W:2048, gEvery:4, gOff:1 },
    lin:{ n:0, kind:"GDN", Hk:0, Hv:0, dk:128, dv:128, conv:4 },
    moe:{ E:256, k:8, F:4096, Fs:0, sharedGate:false, Dlat:4096 }, dense:{ n:0, F:0 },
    wDefault:"nvfp4", official:Msg.of("engine_common.model.musespark.official"),
    note:Msg.of("engine_common.model.musespark.note") },
  qwen38_27b: { name:Msg.of("engine_common.model.qwen38_27b.name"), short:"Qwen3.8-27B", family:"qwen", D:5120, L:64, V:248320, mtp:1,
    full:{ n:16, kind:"gqa", Hq:24, Hkv:4, dh:256, gate:true },
    lin:{ n:48, kind:"GDN", Hk:16, Hv:48, dk:128, dv:128, conv:4 },
    moe:{ E:0, k:0, F:0, Fs:0, sharedGate:false, Dlat:0 }, dense:{ n:64, F:17408 },
    wDefault:"bf16", official:Msg.of("engine_common.model.qwen38_27b.official"),
    note:Msg.of("engine_common.model.qwen38_27b.note") }
};

/* 规格表 dense 数值（TFLOP/s、GB/s、GB）；nvl / ib / host 为单 GPU 单向带宽 */
const HW = {
  H100:{ name:"H100 SXM", sm:90, bf16:989, fp8:1979, fp4:1979, hbm:3350, mem:80, nvl:450, ib:50, host:25 },
  H200:{ name:"H200", sm:90, bf16:989, fp8:1979, fp4:1979, hbm:4800, mem:141, nvl:450, ib:50, host:25 },
  B200:{ name:"B200", sm:100, bf16:2250, fp8:4500, fp4:9000, hbm:8000, mem:180, nvl:900, ib:50, host:25 },
  B300:{ name:Msg.of("engine_common.hw.B300"), sm:103, bf16:2250, fp8:4500, fp4:13500, hbm:8000, mem:288, nvl:900, ib:100, host:25 },
  GB200:{ name:"GB200", grace:true, sm:100, bf16:2500, fp8:5000, fp4:10000, hbm:8000, mem:186, nvl:900, ib:50, host:200 },
  GB300:{ name:"GB300", grace:true, sm:103, bf16:2500, fp8:5000, fp4:15000, hbm:8000, mem:288, nvl:900, ib:100, host:200 }
};
const TOPO = {
  hgx8:{ name:Msg.of("engine_common.topo.hgx8"), gpus:8, domain:8 },
  nvl64:{ name:Msg.of("engine_common.topo.nvl64"), gpus:64, domain:72 }
};   // 只算这两种：所有并行组都在一个 NVLink 域内。跨机走 IB 的公式（link、Par.epLayer）还在，但没有拓扑会用到
const peakOf = (hw,d) => hw[peakKey(d)]*1e12;
/* roofline 的几个量（界面画图和标注也用这些，不另写公式）。peak 是 FLOP/s（peakOf 的结果），η 来自 g */
const compT = (flops, peak, g) => flops/(peak*g.etaC);                          // 按可达算力算的时间
const memT = (bytes, hw, g) => bytes/(hw.hbm*1e9*g.etaM);                       // 按可达 HBM 带宽算的时间
const ridge = (peak, hw, g) => peak*g.etaC/(hw.hbm*1e9*g.etaM);                 // 屋顶拐点（FLOP/B）
const attainable = (I, peak, hw, g) => Math.min(peak*g.etaC, hw.hbm*1e9*g.etaM*I);   // 强度 I 下能达到的 FLOP/s
const linkRidge = (peak, gbs, g) => peak*g.etaC/(gbs*1e9*g.etaAg);             // 每通信 1 字节要配多少 FLOPs 才藏得住（链路 GB/s）

/* 默认的效率系数与固定开销（每页共用，界面上可以改）：η 是可达峰值的比例，α 是集合通信延迟（µs），其余是 kernel / launch / 同步开销（µs） */
const freeze = o => { for(const v of Object.values(o)) if(v && typeof v==="object") freeze(v); return Object.freeze(o); };
const G0 = freeze({ etaC:0.7, etaM:0.8, etaN:0.7, etaAg:0.78, alphaNv:8, alphaIb:20, kernelUs:2, launchUs:4, graphLaunchUs:15, schedUs:0, dpSyncUs:100, a2aLLus:25, deepepUs:10, barrierUs:5 });
/* 理想 roofline：只看硬件峰值，可达比例都是 1、固定开销和延迟都是 0；kernel 序列、重叠规则还是所选引擎 / 框架自己的。
   BASIS 是界面和命令行的“口径”开关：ideal = 硬件屋顶，modeled = 上面的默认效率（实测走数据点库，不在计算核心里） */
const IDEAL = freeze(Object.fromEntries(Object.keys(G0).map(k => [k, k.startsWith("eta") ? 1 : 0])));
const BASIS = freeze({ ideal: IDEAL, modeled: G0 });

/* GQA 的 q + 输出门宽度：门按元素（Qwen）是 Hq·dh，按头（Muse Spark）是 Hq */
const qoW = f => f.Hq*f.dh + (f.gate ? (f.gateHead ? f.Hq : f.Hq*f.dh) : 0);
/* 滑窗层每个 query 平均看到的 key 数（整段长 S 的 causal）；全注意力是 S/2 */
const swaKeys = (S, W) => S>W ? W - W*W/(2*S) : S/2;

/* 参数量：按层类型拆开，推理和训练都从这里取 */
function derive(m){
  const D=m.D, f=m.full, l=m.lin, e=m.moe;
  const r = { nMoe: m.L-m.dense.n, De: e.Dlat||D };
  if(f.n>0){
    if(f.kind==="gqa"){
      r.pQKV = D*(qoW(f) + 2*f.Hkv*f.dh); r.pO = f.Hq*f.dh*D; r.pAttnRep = 0;
      r.kvTokElems = 2*f.Hkv*f.dh; r.Hattn=f.Hq; r.dqk=f.dh; r.dv=f.dh;
    } else {
      const dq=f.nope+f.rope;
      r.pAttnRep = D*(f.qLora + f.kvLora + f.rope);                   // fused_qkv_a，各 TP rank 复制
      r.pQKV = f.qLora*f.H*dq + f.kvLora*f.H*(f.nope+f.vh) + (f.gate? D*f.H*f.vh : 0);
      r.pO = f.H*f.vh*D;
      if(f.kind==="dsa") r.pIdx = f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH; else r.pIdx=0;
      r.kvTokElems = f.kvLora + f.rope;                              // K3 NoPE 仍存 64 维槽位（576）；GLM rope=0（512）
      r.Hattn=f.H; r.dqk=dq; r.dv=f.vh;
    }
  } else { r.pQKV=0; r.pO=0; r.pAttnRep=0; r.kvTokElems=0; r.pIdx=0; }
  r.pIdx = r.pIdx||0;
  r.pAttn = r.pQKV + r.pO + r.pAttnRep + r.pIdx;
  if(l.n>0){
    const inW = 2*l.Hk*l.dk + 2*l.Hv*l.dv + (l.kind==="KDA" ? l.Hk*l.dk + 2*l.Hv : 2*l.Hv); // q,k,v,z/gate + β,α (KDA: f 门低秩近似成 H·dk)
    r.linIn = inW; r.pLin = D*inW + l.Hv*l.dv*D;
    r.linState = l.Hv*l.dk*l.dv; r.linConvDim = 2*l.Hk*l.dk + l.Hv*l.dv;
  } else { r.linIn=0; r.pLin=0; r.linState=0; r.linConvDim=0; }
  r.pExpert = 3*r.De*e.F;
  r.pRouted = e.E*r.pExpert;
  r.pShared = 3*D*e.Fs + (e.sharedGate? D : 0);
  r.pRouter = D*e.E;
  r.pLat = e.Dlat ? 2*D*e.Dlat : 0;
  r.pDense = 3*D*m.dense.F;
  r.pEmb = m.V*D; r.pHead = m.V*D;
  r.total = f.n*r.pAttn + l.n*r.pLin + r.nMoe*(r.pRouted+r.pShared+r.pRouter+r.pLat) + m.dense.n*r.pDense + r.pEmb + r.pHead;
  r.active = f.n*r.pAttn + l.n*r.pLin + r.nMoe*(e.k*r.pExpert+r.pShared+r.pRouter+r.pLat) + m.dense.n*r.pDense + r.pHead;
  r.pExpertsAll = r.nMoe*r.pRouted;
  r.pDenseAll = r.total - r.pExpertsAll;
  return r;
}

/* 每层的类型：attn = full（全注意力）/ swa（滑窗）/ lin（线性注意力），mlp = moe / dense。
   全注意力层均匀插在线性层之间（第 every−1、2·every−1… 层，尾部不够时补齐）；滑窗层按 i % gEvery ≠ gOff 判断；前 dense.n 层是 dense FFN。 */
function layerPlan(m){
  const f=m.full, every = m.lin.n>0 ? Math.round(m.L/Math.max(1,f.n)) : 1, sw=f.swa||0;
  let left=f.n; const out=[];
  for(let i=0;i<m.L;i++){
    const full = m.lin.n===0 || ((i+1)%every===0 && left>0) || (m.L-i<=left); if(full) left--;
    const swa = full && sw>0 && i%(f.gEvery||4)!==(f.gOff||0);
    out.push({ attn: !full ? "lin" : swa ? "swa" : "full", mlp: i>=m.dense.n ? "moe" : "dense" });
  }
  return out;
}

/* 链路：组内 peer 在同一个 NVLink 域就走 NVLink，否则走 IB（取最慢的那条） */
function link(hw, topo, n, stride){
  if(n<=1) return { bw:Infinity, intra:true, alpha:0 };
  const span = n*(stride||1);
  const intra = span<=topo.domain;
  return { bw:(intra?hw.nvl:hw.ib)*1e9, intra };
}
/* ring all-gather / reduce-scatter：msg 为整份张量字节 */
function agT(msg, n, lk, g){ if(n<=1||msg<=0) return 0; return (lk.intra?g.alphaNv:g.alphaIb)*1e-6*Math.log2(n)+ msg*(n-1)/n/(lk.bw*(g.etaAg||g.etaN)); }
function arT(msg, n, lk, g, algo){
  if(n<=1||msg<=0) return 0; const a=(lk.intra?g.alphaNv:g.alphaIb)*1e-6;
  if(algo==="oneshot") return a + msg*(n-1)/(lk.bw*g.etaN);        // 每个 rank 读其余 n−1 份整消息，受单卡带宽限制
  if(algo==="twoshot") return 2*a + 2*msg*(n-1)/n/(lk.bw*g.etaN);
  return 2*a*Math.log2(n) + 2*msg*(n-1)/n/(lk.bw*g.etaN);         // NCCL ring
}
/* all-to-all：每卡发 sent 字节，(n−1)/n 离开本卡；组跨域时只按最慢链路 */
function a2aT(sent, n, lk, g, alphaUs){ if(n<=1||sent<=0) return 0; return (alphaUs ?? (lk.intra?g.alphaNv:g.alphaIb))*1e-6 + sent*(n-1)/n/(lk.bw*g.etaN); }

/* 单个算子的 roofline：t = max(FLOPs/峰值, 字节/带宽, kernel 数 × 最短 kernel 时间) */
function roof(flops, bytes, peak, hw, g, kernels){
  const tc = compT(flops, peak, g), tm = memT(bytes, hw, g), tf=(kernels||0)*g.kernelUs*1e-6;
  const t = Math.max(tc, tm, tf);
  return { t, tc, tm, tf, bound: t===tf&&tf>0 ? "launch" : tc>=tm ? "compute" : "memory", I: bytes>0?flops/bytes:Infinity };
}

return { DT, dtB, peakKey, peakOf, compT, memT, ridge, attainable, linkRidge, G0, IDEAL, BASIS, freeze, MODELS, HW, TOPO, derive, layerPlan, qoW, swaKeys, link, agT, arT, a2aT, roof };
})();
