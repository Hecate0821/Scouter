/* 架构页的计算：把一个模型拆成模块（每种层的 attention 模块、MLP 模块），给出每个模块“一层”的 FLOPs、字节、时间和 roofline 位置。
   不另写公式：推理走 Infer.step 的参考画像（ENGINES.ref：各引擎都默认开的优化，单卡、不补齐 CUDA graph），
   训练走 Train.run（Megatron 默认、单卡、不重算）；再按算子的 group / key / mod 归到模块上，除以该算子覆盖的层数。 */
const Arch = (function(){
const { derive, layerPlan, dtB, peakKey, peakOf, TOPO } = Common;

const ORDER = ["full", "swa", "lin", "moe", "dense"];
const isMlp = id => id==="moe" || id==="dense";

/* 模型里有哪些模块：id、属于 attention 还是 MLP、机制、层数 */
function present(m){
  const cnt = {};
  for(const x of layerPlan(m)){ cnt[x.attn]=(cnt[x.attn]||0)+1; cnt[x.mlp]=(cnt[x.mlp]||0)+1; }
  return ORDER.filter(id => cnt[id]>0).map(id => ({ id, part: isMlp(id) ? "mlp" : "attn", layers: cnt[id],
    kind: id==="full" ? m.full.kind : id==="swa" ? "swa" : id==="lin" ? m.lin.kind.toLowerCase() : id }));
}

/* 默认工作负载：精度跟着官方 checkpoint；chunk 大小按 SGLang / vLLM 的显存分档（< 160 GB 8192，否则 16384） */
function defaults(m){
  const e = m.wDefault || "bf16";
  return { phase:"decode", B:64, n:8192, C:0, S:8192, wDt: e==="fp8" ? "fp8" : "bf16", eDt: e, kvDt:"bf16", stateDt:"fp32", prec:"bf16" };
}
const chunkOf = (wl, hw) => wl.C>0 ? wl.C : (hw.mem>=160 ? 16384 : 8192);

function infCfg(wl, over){
  return Object.assign({ engine:"ref", phase:"decode", N:1, dpA:1, B:wl.B, n:wl.n, C:8192, P:8192, p:0,
    wDt:wl.eDt, eAct:"", dDt:wl.wDt, kvDt:wl.kvDt, stateDt:wl.stateDt, specK:0, specMethod:"mtp", accept:0, graph:false,
    imb:1, redundant:0, dcp:1, prefixCache:false, moePar:"tp", dispFp8:false, tbo:false, hostSyncUs:0,
    memFrac:0.9, reserveGB:4, bi:false, biFactor:1, implEff:1, reqPerChunk:1 }, over||{});
}
function trnCfg(wl){
  return { fw:"megatron", N:1, tp:1, cp:1, pp:1, vpp:1, ep:1, etp:1, dispatcher:"alltoall", cpType:"p2p", linCP:"headwise",
    tok:wl.S, S:wl.S, Gtok:wl.S, prec:wl.prec, recompute:"selective", rcModules:[], offload:false, overlapOffload:false, cpuAdamGBs:30,
    overlapGrad:true, overlapParam:true, epOverlap:false, sharedOverlap:false, tpOverlap:false, reshard:false, imb:1, hostSyncUs:0,
    workspaceGB:0, cpLB:true, hsdp:1, gradDt:"fp32", mokF:0.25, etaA:0 };
}

/* 算子 → 模块：attention 的投影是全局层和滑窗层共用的（同一个 op 覆盖 full.n 层），核心按 mod 分开 */
function modsOf(o){
  if(o.group==="attn") return o.mod ? [o.mod] : ["full","swa"];
  if(o.group==="lin") return ["lin"];
  if(o.group==="moe") return [o.key==="dense" ? "dense" : "moe"];
  return [];
}
const blank = () => ({ flops:0, bytes:0, t:0, tc:0, tm:0, tf:0, byBound:{ compute:0, memory:0, launch:0 }, ops:[] });
function addOp(acc, ids, o, x, scale){
  const L=o.layers||1, k=scale/L;
  for(const id of ids){ const a=acc[id]; if(!a) continue;
    a.flops+=x.flops*k; a.bytes+=x.bytes*k; a.t+=x.t*k; a.tc+=(x.tc||0)*k; a.tm+=(x.tm||0)*k; a.tf+=(x.tf||0)*k;
    a.byBound[x.bound in a.byBound ? x.bound : "memory"] += x.t*k;
    const prev=a.ops.find(p=>p.key===o.key && p.name===o.name);
    if(prev){ prev.flops+=x.flops*k; prev.bytes+=x.bytes*k; prev.t+=x.t*k; }
    else a.ops.push({ key:o.key, name:o.name, flops:x.flops*k, bytes:x.bytes*k, t:x.t*k, bound:x.bound }); }
}
function finish(acc){
  for(const id in acc){ const a=acc[id];
    a.I = a.bytes>0 ? a.flops/a.bytes : Infinity;
    a.perf = a.t>0 ? a.flops/a.t : 0;
    a.bound = Object.keys(a.byBound).reduce((b,k)=>a.byBound[k]>a.byBound[b]?k:b, "memory");
    for(const p of a.ops){ p.I = p.bytes>0 ? p.flops/p.bytes : Infinity; } }
  return acc;
}

/* 一层的模块统计。decode：B 条序列各 n 个 token 的上下文，走一步；prefill：长 n 的 prompt 按 chunk 走完（取全部 chunk 的和）；
   train：一条长 S 的序列 fwd + dgrad + wgrad。 */
function measure(m, hw, g, wl){
  const topo = TOPO.hgx8, acc = {};
  for(const p of present(m)) acc[p.id] = blank();
  if(wl.phase==="train"){
    const r = Train.run(m, hw, topo, trnCfg(wl), g);
    for(const o of r.ops){ const ids=modsOf(o); if(!ids.length) continue;
      for(const ps of ["fwd","dgrad","wgrad"]){ const x=o[ps]; if(!x || !(x.t>0 || x.fl>0)) continue;
        addOp(acc, ids, o, { flops:x.fl, bytes:x.by, t:x.t, tc:0, tm:0, tf:0, bound:x.bound }, 1); } }
    return { mods: finish(acc), tokens: wl.S };
  }
  if(wl.phase==="prefill"){
    const C = Math.min(chunkOf(wl,hw), wl.n), nCh = Math.ceil(wl.n/C), mx = wl.maxChunks || 24;   // chunk 太多时按等间距抽样（取各段中点）
    const pick = nCh<=mx ? [...Array(nCh).keys()] : [...Array(mx).keys()].map(i=>Math.floor((i+0.5)*nCh/mx));
    const scale = nCh/pick.length;
    for(const i of pick){ const p=i*C, cc=Math.min(C, wl.n-p);
      const r = Infer.step(m, hw, topo, infCfg(wl, { phase:"prefill", C:cc, P:wl.n, p }), g);
      for(const o of r.ops) addOp(acc, modsOf(o), o, o, scale); }
    return { mods: finish(acc), tokens: wl.n, chunk: C, chunks: nCh };
  }
  const r = Infer.step(m, hw, topo, infCfg(wl), g);
  for(const o of r.ops) addOp(acc, modsOf(o), o, o, 1);
  return { mods: finish(acc), tokens: wl.B };
}

/* 每个模块每层存多少：KV（每 token，随上下文增长）、递推状态（每条序列，常数）、权重 */
function facts(m, wl){
  const d=derive(m), f=m.full, l=m.lin, e=m.moe, kvb=dtB(wl.kvDt), sb=wl.stateDt==="bf16"?2:4, wb=dtB(wl.wDt), eb=dtB(wl.eDt);
  const out = {};
  if(f.n>0){
    const idx = f.kind==="dsa" ? (f.idxD+4)/(f.kpool||1) : 0;          // DSA：fp8 index key + scale，按池存
    const kvTok = d.kvTokElems*kvb + idx;
    const w = (d.pQKV+d.pO+d.pAttnRep+(d.pIdx||0))*wb;
    out.full = { kvTok, weights:w, params:d.pAttn };
    if(f.swa) out.swa = { kvTok: d.kvTokElems*kvb, window:f.W, weights:w, params:d.pAttn };
  }
  if(l.n>0) out.lin = { state: d.linState*sb + d.linConvDim*(l.conv-1)*2, weights:d.pLin*wb, params:d.pLin };
  if(d.nMoe>0 && e.E>0){ const per=d.pExpert*eb;
    out.moe = { expert: per, experts: e.E*per, shared: d.pShared*wb, other:(d.pRouter+d.pLat)*2, weights: e.E*per + d.pShared*wb + (d.pRouter+d.pLat)*2,
      params: d.pRouted+d.pShared+d.pRouter+d.pLat, active: e.k*d.pExpert+d.pShared+d.pRouter+d.pLat, cover: B => Infer.cover(e.k, e.E, B) }; }
  if(m.dense.n>0) out.dense = { weights: d.pDense*wb, params: d.pDense };
  return out;
}

/* roofline 的屋顶：用到的峰值（BF16 注意力核心 + 权重精度的 GEMM；训练按训练精度） */
function peaks(hw, wl){
  const ks = new Set(wl.phase==="train" ? ["bf16", peakKey(wl.prec)] : ["bf16", peakKey(wl.wDt), peakKey(wl.eDt)]);
  return ["bf16","fp8","fp4"].filter(k=>ks.has(k)).map(k=>({ k, v: peakOf(hw, k==="fp4"?"nvfp4":k) }));
}
/* 一个模块主要的 GEMM 精度（读数里的 ridge 用它）：注意力核心按 BF16；MLP 按 expert / dense 权重精度，训练按训练精度 */
function peakKeyFor(id, wl){
  if(!isMlp(id)) return "bf16";
  return peakKey(wl.phase==="train" ? wl.prec : id==="moe" ? wl.eDt : wl.wDt);
}

return { ORDER, present, defaults, chunkOf, measure, facts, peaks, peakKeyFor, isMlp };
})();
