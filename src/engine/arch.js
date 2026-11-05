/* 架构页的计算：把一个模型拆成模块（每种层的 attention 模块、MLP 模块），给出每个模块“一层”的 FLOPs、字节、时间和 roofline 位置。
   不另写公式：推理走 Infer.step 的参考画像（ENGINES.ref：各引擎都默认开的优化，单卡、不补齐 CUDA graph），
   训练走 Train.run（Megatron 默认、单卡、不重算）；再按算子的 group / key / mod 归到模块上，除以该算子覆盖的层数。 */
const Arch = (function(){
const { derive, layerPlan, dtB, peakKey, peakOf, ridge, TOPO, csaKv, csaLatB } = Common;

const ORDER = ["full", "swa", "lin", "moe", "dense"];
const isMlp = id => id==="moe" || id==="dense";

/* 模型里有哪些模块：id、属于 attention 还是 MLP、机制、层数 */
function present(m){
  const cnt = {};
  for(const x of layerPlan(m)){ cnt[x.attn]=(cnt[x.attn]||0)+1; cnt[x.mlp]=(cnt[x.mlp]||0)+1; }
  return ORDER.filter(id => cnt[id]>0).map(id => ({ id, part: isMlp(id) ? "mlp" : "attn", layers: cnt[id],
    kind: id==="full" ? m.full.kind : id==="swa" ? "swa" : id==="lin" ? m.lin.kind.toLowerCase() : id }));
}

/* 默认工作负载：精度跟着官方 checkpoint（dDefault / kvDefault 是 checkpoint 自己给的非 expert 权重和 KV 精度，没有时按 expert 精度推）；
   chunk 大小按 SGLang / vLLM 的显存分档（< 160 GB 8192，否则 16384） */
function defaults(m){
  const e = m.wDefault || "bf16";
  return { phase:"decode", B:64, n:8192, C:0, S:8192, wDt: m.dDefault || (e==="fp8" ? "fp8" : "bf16"), eDt: e, kvDt: m.kvDefault || "bf16", stateDt:"fp32", prec:"bf16" };
}
const chunkOf = (wl, hw) => wl.C>0 ? wl.C : (hw.mem>=160 ? 16384 : 8192);

/* 在默认配置上改出参考画像：单卡、引擎中立（ref），不补齐 CUDA graph、不算负载不均、不开前缀缓存和 FP8 dispatch、没有 host 同步 */
function infCfg(wl, over){
  return Object.assign({}, Infer.BASE, { engine:"ref", N:1, dpA:1, B:wl.B, n:wl.n, P:8192, moePar:"tp",
    wDt:wl.eDt, dDt:wl.wDt, kvDt:wl.kvDt, stateDt:wl.stateDt, graph:false, imb:1, prefixCache:false, dispFp8:false, hostSyncUs:0,
    memFrac:0.9, reserveGB:4, biFactor:1 }, over||{});
}
/* 训练：Megatron 默认，单卡、一条长 S 的序列，不切并行、不卸载、不重算 */
function trnCfg(wl){
  return Object.assign({}, Train.BASE, Train.FW_DEF.megatron, { fw:"megatron", N:1, tp:1, cp:1, pp:1, vpp:1, ep:1, etp:1,
    tok:wl.S, S:wl.S, Gtok:wl.S, prec:wl.prec, recompute:"selective", rcModules:[], offload:false, overlapOffload:false,
    reshard:false, imb:1, hostSyncUs:0, workspaceGB:0 });
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
    const prev=a.ops.find(p=>p.key===o.key && Msg.id(p.name)===Msg.id(o.name));
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
  if(f.n>0 && f.kind==="csa"){
    // CSA：压缩 KV + index K 只在 kv source 层存，这里摊到每个压缩层（kvTok）；滑窗 latent 每层都有，每条序列最多 W 个（swaTok）。
    // 权重：每层同形状的 q / 滑窗 KV / 输出投影（wo_a 是 f.oDt），indexer 和 compressor 摊到压缩层
    const kv=csaKv(m, wl.kvDt), P=d.csa, nF=P.filter(x=>x.m>0).length, nS=f.n-nF;
    const w = (d.pQKV+d.pO-d.pOa+d.pAttnRep)*wb + d.pOa*dtB(f.oDt), pBase = d.pQKV+d.pO+d.pAttnRep;
    const wIdx = d.nIdx*(f.qLora*f.idxH*f.idxD*wb + m.D*f.idxH*2) + P.filter(x=>x.kv).reduce((s,x)=>s + m.D*f.dh*(x.m>1?2:1)*dtB(x.m>1?f.cmpDt:"bf16") + f.dh*f.idxD*2, 0);
    if(nF) out.full = { kvTok:(kv.main+kv.index)/nF, swaTok:kv.swa, window:f.W, weights:w + wIdx/nF, params:pBase + d.pIdx*f.n/nF };
    if(nS) out.swa = { kvTok:kv.swa, window:f.W, weights:w, params:pBase };
  } else if(f.n>0){
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
  return ["bf16","fp8","fp4"].filter(k=>ks.has(k)).map(k=>({ k, v: peakOf(hw, k) }));
}
/* 一个模块主要的 GEMM 精度（读数里的 ridge 用它）：注意力核心按 BF16；MLP 按 expert / dense 权重精度，训练按训练精度 */
function peakKeyFor(id, wl){
  if(!isMlp(id)) return "bf16";
  return peakKey(wl.phase==="train" ? wl.prec : id==="moe" ? wl.eDt : wl.wDt);
}

/* 一步里这个模块处理的行数：decode 是 B 条序列各一个 token，prefill 是一个 chunk，训练是一条长 S 的序列 */
const rowsOf = (wl, hw) => wl.phase==="decode" ? wl.B : wl.phase==="prefill" ? chunkOf(wl,hw) : wl.S;

/* 机制量：示意图和读数用的都从这里取，同一个量只算一次。B 是一步的行数（默认 rowsOf），n 是上下文长度；
   DSA、线性注意力给的是每条序列的量。need / Bstar：GEMM 进入算力区所需的每个 expert 行数 / batch（ridge × 每参数字节 / 2） */
function explain(m, hw, g, wl, id, B){
  B = B ?? rowsOf(wl, hw);
  const f=m.full, l=m.lin, e=m.moe, n=Math.max(1,wl.n), kvb=dtB(wl.kvDt), x={ B, n };
  if((id==="full" || id==="swa") && f.kind==="csa"){
    /* CSA（每条序列、每层的平均）：attention 看 kc 个压缩 latent + ks 个滑窗 latent；index source 层打分的位置数 scored（自己的 n / m，
       候选层之后最多 blocks·bs 个），读 scored 个 index K */
    const kv=csaKv(m, wl.kvDt), P=derive(m).csa, ks=Math.min(n, f.W);
    Object.assign(x, { kvTok: facts(m, wl)[id].kvTok, swaB:kv.swa, latB:kv.lat, idxTok:kv.idx, ks, W:f.W });
    if(id==="swa") return Object.assign(x, { keys:ks, attB:ks*kv.swa, attF:4*f.H*f.dh*ks });
    const full=P.filter(q=>q.m>0), idx=P.filter(q=>q.idx), cap=f.cand ? f.cand.blocks*f.cand.bs : Infinity;
    const kc=full.reduce((s,q)=>s+Math.min(f.topk, n/q.m), 0)/full.length;
    const sc=q => n/q.m<=f.topk ? 0 : q.useCand ? Math.min(n/q.m, cap) : n/q.m, scored=idx.reduce((s,q)=>s+sc(q), 0)/Math.max(1, idx.length);
    return Object.assign(x, { kc, keys:kc+ks, nFull:full.length, nIdx:idx.length, nSrc:P.filter(q=>q.kv).length, scored, cap, keep:Math.min(1, f.topk/Math.max(1, n/2)),
      attB:kc*kv.lat + ks*kv.swa, idxB:scored*kv.idx*idx.length/full.length, attF:4*f.H*f.dh*(kc+ks), idxF:2*f.idxH*f.idxD*scored*idx.length/full.length });
  }
  if(id==="full" || id==="swa"){
    x.kvTok = facts(m, wl)[id].kvTok;
    if(id==="swa" || f.kind==="gqa") return Object.assign(x, { groups:f.Hq/f.Hkv, keys: id==="swa" ? Math.min(n,f.W) : n });
    const lat=f.kvLora+f.rope;
    Object.assign(x, { lat, absorbI:2*f.H*(lat+f.kvLora)/(lat*kvb), allB:n*lat*kvb });
    if(f.kind!=="dsa") return x;
    const k=Math.min(n,f.topk), pools=Math.ceil(n/(f.kpool||1));
    return Object.assign(x, { k, pools, keep:Math.min(1,f.topk/n), idxB:pools*(f.idxD+4), attB:k*lat*kvb, idxF:2*f.idxH*f.idxD*pools, attF:2*f.H*(lat+f.kvLora)*k });
  }
  if(id==="lin"){ const sb=wl.stateDt==="bf16"?2:4, st=l.Hv*l.dk*l.dv*sb, kvTokFull=f.n>0?derive(m).kvTokElems*kvb:0;
    return Object.assign(x, { sb, st, rwB:2*st*B, I:8/(2*sb), kvTokFull, vsKv:kvTokFull>0?st/kvTokFull:null, kvEq:n*2*(l.Hk*l.dk)*kvb }); }
  if(id==="moe"){ const cov=Infer.cover(e.k,e.E,B), hit=e.E*cov, eb=dtB(wl.eDt), need=ridge(peakOf(hw,wl.eDt),hw,g)*eb/2;
    return Object.assign(x, { cov, hit, rows:B*e.k/Math.max(1,hit), per:3*(e.Dlat||m.D)*e.F*eb, need, needB:need*e.E/e.k }); }
  const F=m.dense.F, wb=dtB(wl.wDt), rI=ridge(peakOf(hw,wl.wDt),hw,g);
  return Object.assign(x, { F, wB:3*m.D*F*wb, I:2*B*3*m.D*F/(3*m.D*F*wb + B*(m.D*2+3*F*2)), ridge:rI, Bstar:rI*wb/2 });
}

return { ORDER, present, defaults, chunkOf, rowsOf, measure, facts, explain, peaks, peakKeyFor, isMlp };
})();
