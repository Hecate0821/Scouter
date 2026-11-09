/* 推理核心：只描述 serving 引擎真正发出的 kernel 序列（paged KV、状态缓存、CUDA graph 补齐、采样、MTP 验证、DP-attention / EP 通信）。
   与训练核心不共享算子表。每条口径对应的仓库路径见 UI 的“口径与来源”。 */
const Infer = (function(){
const { dtB, peakOf, derive, qoW, capKeys, link, agT, arT, a2aT, roof, gK, csaKv, csaPlan, sm10x } = Common;

function range(a,b,s){ const r=[]; for(let x=a;x<=b;x+=s) r.push(x); return r; }
/* 引擎画像：来自三个引擎的默认值（research/inference_engines.md Part IV）。replay：DeepSeek-V4.1 的 decoder SWA bounded replay
   默认开（SGLang --enable-decoder-swa-bounded-replay，从最后一个 kv source 层起只跑每个请求最后 W 个 token，那一层的压缩 KV 仍按全部行写；
   vLLM swa_bounded_replay 从它的下一层起）；没有这项的引擎为空 */
const ENGINES = {
  sglang: { name:"SGLang", mixed:false, dcpComm:"a2a", dcpOutB:4, lmInGraph:true, samplerInGraph:false, specCap:48, defaultMoe:"tp",
    ladder:(hw,spec)=> spec ? [...range(1,8,1),...range(10,32,2),...range(40,64,4),...range(72,256,8)] : [1,2,4,8,12,...range(16,256,8),...range(272,496,16),512],
    maxGraph:(hw,spec)=> spec?48:512,
    arFusion:(m,c,T)=> (m.family==="qwen"||m.family==="kimi") && c.dpA===1 && T<=2048,
    stateDecode:()=>2, stateVerify:(m,N)=>N+3, prefillGraph:m=>m.family!=="kimi" && m.full.kind!=="csa", prefillPad:0.06,
    hMat:(m,hw)=> !(m.lin.kind==="GDN" && hw.sm<100), mlaReexpand:true, sharedOverlap:"graph", replay:"source",
    kernels:{ gqa:Msg.of("engine_infer.kernel.sglang.gqa"), mla:"trtllm_mla", dsa:Msg.of("engine_infer.kernel.sglang.dsa"), csa:Msg.of("engine_infer.kernel.sglang.csa"), gdn:"fused_recurrent_gated_delta_rule_packed_decode", kda:Msg.of("engine_infer.kernel.sglang.kda"), moe:Msg.of("engine_infer.kernel.sglang.moe"), ar:Msg.of("engine_infer.kernel.sglang.ar") } },
  vllm: { name:"vLLM", mixed:true, dcpComm:"ag_rs", dcpOutB:2, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"ag",
    ladder:(hw,spec)=> [1,2,4,...range(8,248,8),...range(256, hw.sm>=100?1024:512,16)],
    maxGraph:(hw)=> hw.sm>=100?1024:512,
    arFusion:(m,c,T,D,hw)=> m.family==="qwen" && c.dpA===1 && T*D*2 <= (hw.sm>=100?1e6:5e5),
    stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>true, prefillPad:0.06,
    hMat:m=> m.lin.kind==="GDN", mlaReexpand:true, sharedOverlap:"small", replay:"after",
    kernels:{ gqa:"SM100 FlashInfer trtllm-gen / SM90 FA3", mla:Msg.of("engine_infer.kernel.vllm.mla"), dsa:"FLASHINFER_MLA_SPARSE + DeepGEMM paged MQA logits", csa:"FLASHINFER_MLA_SPARSE_DSV41", gdn:"fused_recurrent_gated_delta_rule_packed_decode + causal_conv1d_update", kda:Msg.of("engine_infer.kernel.vllm.kda"), moe:Msg.of("engine_infer.kernel.vllm.moe"), ar:Msg.of("engine_infer.kernel.vllm.ar") } },
  tokenspeed: { name:"TokenSpeed", mixed:false, lmInGraph:true, samplerInGraph:true, specCap:Infinity, defaultMoe:"replicated",
    ladder:(hw,spec)=> [1,2,4,...range(8, spec?80:160, 8)],
    maxGraph:(hw,spec)=> spec?80:160,
    arFusion:(m,c,T)=> m.family!=="glm" && c.dpA===1 && T<=2048,
    stateDecode:m=> m.lin.kind==="KDA"?3:2, stateVerify:()=>3, prefillGraph:m=>m.full.kind!=="dsa", prefillPad:0.125,
    hMat:()=>false, mlaReexpand:true, dsaAbsorbPrefill:true, sharedOverlap:"graph",
    kernels:{ gqa:"triton_gqa_prologue + trtllm-gen / fa3", mla:Msg.of("engine_infer.kernel.tokenspeed.mla"), dsa:Msg.of("engine_infer.kernel.tokenspeed.dsa"), csa:"—", gdn:"triton_causal_conv1d_update + flashinfer_gdn_decode_step", kda:Msg.of("engine_infer.kernel.tokenspeed.kda"), moe:Msg.of("engine_infer.kernel.tokenspeed.moe"), ar:Msg.of("engine_infer.kernel.tokenspeed.ar") } },
  /* 架构页用的参考画像：只取各引擎都默认开的优化。全注意力 FlashAttention 类 kernel（FA3 / FA4 / FlashInfer / TRT-LLM FMHA）；
     MLA decode absorbed（FlashMLA 类），prefill 走 MHA 并把前缀 latent 重新展开（SGLang MHA_CHUNKED_KV、vLLM _compute_prefill_context）；
     DSA 按 DeepSeek 开源做法：DeepGEMM fp8 MQA logits 做 indexer，FlashMLA sparse 在 latent 上做 MQA，prefill 也 absorbed；
     线性注意力 FLA：decode fused_recurrent（读写一遍状态），prefill chunk（C=64，h 每个 chunk 写回 HBM）；MoE fused grouped GEMM。
     不补齐 CUDA graph（调用方关掉 graph），没有 AR 融合，shared expert 不藏。 */
  ref: { name:Msg.of("engine_infer.engine.ref"), mixed:false, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"tp",
    ladder:()=>[1,2,4,...range(8,1024,8)], maxGraph:()=>1024,
    arFusion:()=>false, stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>false, prefillPad:0,
    hMat:()=>true, mlaReexpand:true, dsaAbsorbPrefill:true, sharedOverlap:"none", replay:"source",
    kernels:{ gqa:"FlashAttention", mla:"FlashMLA", dsa:"FlashMLA sparse + DeepGEMM indexer", csa:"FlashMLA sparse (SWA + compressed)", gdn:"FLA fused_recurrent", kda:"FLA fused_recurrent (KDA)", moe:"fused MoE grouped GEMM", ar:"—" } },
  titan: { name:"TorchTitan TitanRL", mixed:true, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"standard",
    ladder:(hw)=> [1,2,4,...range(8,248,8),...range(256,512,16)], maxGraph:()=>512,
    arFusion:()=>false, stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>false, prefillPad:0,
    hMat:()=>true, mlaReexpand:false, sharedOverlap:"none",
    kernels:{ gqa:Msg.of("engine_infer.kernel.titan.gqa"), mla:Msg.of("engine_infer.kernel.titan.mla"), dsa:Msg.of("engine_infer.kernel.titan.dsa"), csa:"—", gdn:"attn_gym GDN", kda:"attn_gym KDA", moe:"torch._grouped_mm / Dist-MoE", ar:Msg.of("engine_infer.kernel.titan.ar") } }
};

const MOEPARS = ["tp","replicated","ag","standard","deepep","hybridep","fused"];
const padTo = (ladder, x) => { for(const v of ladder) if(v>=x) return v; return x; };
/* 每个算子查效率表用的 kernel 类别（Common.KEFF）：带权重的线性层是 gemm，attention 核心是 attn，MoE 是 grouped GEMM；
   elementwise、norm、线性注意力的扫描没有公开的单独效率，不查表 */
const KCLS = { moe:"moe", acore:"attn", aidx:"gemm" };
for(const k of ["aproj","aout","aqa","aqb","agp","akvb","auk","auv","aidxp","acmp","aoa","engram","lproj","lout","shared","dense","lm","latd","latu","router"]) KCLS[k]="gemm";
const cover = (k,E,T) => 1-Math.pow(1-k/E, Math.max(0,T));
/* 开投机时并发被引擎默认上限截住的警告；投机页按这句原文把它从警告栏里去掉（正文里交代） */
const specCapWarn = P => Msg.of("engine_infer.warn.specCap", { engine:P.name, cap:P.specCap });
/* expert GEMM 的激活精度。没指定（eAct 为空）时按 SGLang 的默认：MXFP4 权重在 Blackwell 上跑 MXFP8 激活（W4A8：
   FlashInfer trtllm_fp4_block_scale_moe / DeepGEMM fp8_fp4，mxfp4.py），MMA 是 FP8 档；Hopper 没有 FP4 MMA，FP4 权重
   走 W4A16（FlashInfer CUTLASS / Marlin），MMA 是 BF16 档；其余跟权重精度相同（NVFP4 在 Blackwell 上是 W4A4） */
const eActOf = (c, hw) => c.eAct==="same" ? c.wDt : c.eAct || (c.wDt==="mxfp4" && hw.sm>=100 ? "mxfp8" : (c.wDt==="mxfp4"||c.wDt==="nvfp4") && hw.sm<100 ? "bf16" : c.wDt);

/* 一次 forward（decode 或 prefill chunk）的逐算子时间 */
function step(m, hw, topo, c, g){
  const d=derive(m), P=ENGINES[c.engine], f=m.full, l=m.lin, e=m.moe, D=m.D;
  const N=c.N, dpA=Math.max(1,c.dpA), tpA=N/dpA, dec=c.phase==="decode";
  const wbE=dtB(c.wDt), wbD=dtB(c.dDt), kvb=dtB(c.kvDt), sb=c.stateDt==="bf16"?2:4;
  const pkD=peakOf(hw,c.dAct||c.dDt), pkE=peakOf(hw,eActOf(c,hw)), pkA=peakOf(hw,"bf16");   // eAct / dAct：expert / 其余线性层的激活精度（W4A16 / W4A8 时 MMA 跑在激活的峰值上）
  const lb=!!g.lb;                                         // 理想下限口径（Common.IDEAL 的注释）
  const slow=lb ? 1 : (c.bi?c.biFactor:1)/(c.implEff||1);
  /* DSpark 的验证长度由置信度头决定（技术报告 2.4.3：confidence-scheduled verification）：specVer 是平均送去验证的草稿比例，
     验证宽度 = 1 + round(k · specVer)；其余方法每步验证全部 k 个 */
  const specK = dec ? (c.specK||0) : 0, vK = specK>0 && c.specMethod==="dspark" ? Math.max(1, Math.min(specK, Math.round(specK*(c.specVer??1)))) : specK;
  const w = specK>0 ? vK+1 : 1;
  const warn=[];
  if(specK>0 && !c._mb && (c.accept||0)>vK) warn.push(Msg.of("engine_infer.warn.acceptVer",{a:c.accept,v:vK}).invalid());
  if(!Number.isInteger(tpA)) warn.push(Msg.of("engine_infer.warn.dpDivides",{dp:dpA,n:N}).invalid());
  // 批大小与 CUDA graph 补齐（TBO 的两个 micro-batch 由外层按补齐后的整批切好，经 c._mb 传进来）
  let Breq = dec ? c.B : 1;
  if(!c._mb && dec && specK>0 && Breq>P.specCap){ warn.push(specCapWarn(P)); Breq=P.specCap; }
  const ladder=P.ladder(hw,specK>0), maxG=P.maxGraph(hw,specK>0);
  let graph, rowsG, rowsR;
  if(c._mb) ({ Breq, graph, rowsG, rowsR } = c._mb);
  else if(dec){
    const units = c.engine==="vllm"||c.engine==="titan" ? Breq*w : Breq;
    graph = c.graph && units<=maxG;
    const unitsPad = graph && !lb ? padTo(ladder, units) : units;
    rowsG = c.engine==="vllm"||c.engine==="titan" ? unitsPad : unitsPad*w;
    rowsR = Breq*w;
  } else {
    graph = c.graph && P.prefillGraph(m) && c.C<=8192;
    rowsR = c.C; rowsG = graph && !lb ? Math.ceil(c.C*(1+P.prefillPad*0.5)) : c.C;
  }
  const Bp = dec ? rowsG/w : 1;   // 补齐后的请求数（只影响随行数缩放的算子）
  const mp0 = c.moePar||P.defaultMoe;
  /* DeepSeek-V4.1 在 SGLang 里还不支持的组合（srt/arg_groups/deepseek_v4_hook.py）：TBO、DSpark 以外的投机；Hopper 上没有 FP4 KV 的 kernel */
  if(f.kind==="csa" && !c._mb && !c._rp){
    if(c.engine==="sglang" && c.tbo && N>1) warn.push(Msg.of("engine_infer.warn.csaTbo").invalid());
    if(c.engine==="sglang" && specK>0 && c.specMethod!=="dspark") warn.push(Msg.of("engine_infer.warn.csaSpec").invalid());
    if((c.kvDt==="nvfp4"||c.kvDt==="mxfp8") && !sm10x(hw)) warn.push(Msg.of("engine_infer.warn.csaKvHopper"));   // V41 / V41_FP4 布局只有 SM100 的 kernel（Hopper 和 SM120 都没有）
  }
  /* bounded replay（prefill，见 ENGINES 的 replay）：模型切成前后两段各算一遍再相加。前段（到最后一个 kv source 层为止）跑整个 chunk；
     后段只跑每个结束的请求最后 W 个 token（c.reqPerChunk 个请求），按 eager 跑；SGLang 的切法下，最后那个 kv source 层的压缩 KV
     和 index K 仍按整个 chunk 写，记在前段。SGLang 开 DP attention 时只配 a2a backend none / megamoe，DeepEP 下不开 */
  const rpOn = f.kind==="csa" && !dec && !c._rp && !c._mb && c.replay!==false && !!P.replay && !(c.engine==="sglang" && dpA>1 && mp0==="deepep");
  if(f.kind==="csa" && !dec && !c._rp && !c._mb && c.replay!==false && P.replay && !rpOn) warn.push(Msg.of("engine_infer.warn.replayOff"));
  if(rpOn){
    const last = Math.max(...f.kvSrc), cut = P.replay==="source" ? last : last+1;
    const rd = Math.min(rowsR, f.W*(c.reqPerChunk??1));
    const enc = step(csaSlice(m,0,cut), hw, topo, Object.assign({}, c, { reqPerChunk:0, _rp:{ kvAlso:cut===last ? [last] : [] } }), g);
    if(rd<=0) return Object.assign(enc, { warn:warn.concat(enc.warn), replay:{ cut, rows:0 } }, csaKvOut(m, c));
    const tail = step(csaSlice(m,cut,m.L), hw, topo, Object.assign({}, c, { C:rd, p:(c.p||0)+rowsR-rd, graph:false, _rp:{ kvSkip:cut===last ? [last] : [] } }), g);
    return Object.assign(seqMerge(enc, tail, c, g, warn), { replay:{ cut, rows:rd } }, csaKvOut(m, c));
  }
  /* TBO（two-batch overlap，SGLang --enable-two-batch-overlap、vLLM --enable-dbo）：只对走单独 a2a 的 MoE 后端，整批切成两个
     micro-batch 依次跑完整的层序列。每个 micro-batch 各读一遍权重（不算两次之间 L2 里的复用）、各发一遍 kernel 和 a2a（固定延迟也是两份）；
     一个 micro-batch 的 a2a 只能藏在另一个的计算后面，能藏的比例是 g.etaTbo。prefill 按 token 切，后一半的前缀多出前一半。
     下限口径不切：整批的 a2a 全部可以藏在整批的计算后面（切开只会多读权重、多发 kernel），所以理想仍不慢于建模 */
  if(c.tbo && !c._mb && !lb && N>1 && derive(m).nMoe>0 && ["standard","deepep","hybridep"].includes(mp0) && (dec ? Breq>=2 : rowsR>=2)){
    const half = (x, first) => first ? Math.ceil(x/2) : Math.floor(x/2);
    const mb = first => Object.assign({}, c, { _mb:{ Breq:dec ? half(Breq,first) : 1, graph, rowsG:half(rowsG,first), rowsR:half(rowsR,first) } },
      dec ? {} : { p:(c.p||0) + (first ? 0 : half(rowsR,true)), reqPerChunk:first ? 0 : c.reqPerChunk??1 });
    return tboMerge(step(m,hw,topo,mb(true),g), step(m,hw,topo,mb(false),g), c, g, { warn, Breq, graph, rowsG, rowsR, Bp });
  }
  const ops=[], comm=[];
  const pkName = pk => pk===pkA ? "bf16" : pk===peakOf(hw,"fp8") ? "fp8" : pk===peakOf(hw,"fp4") ? "fp4" : "bf16";
  const op=(key,name,group,layers,flops,bytes,peak,kern,extra)=>{ if(layers<=0) return;
    const gk=gK(g,KCLS[key],pkName(peak),hw), r=roof(flops,bytes,peak,hw,gk,kern); r.t*=slow; r.tc*=slow; r.tm*=slow;
    ops.push(Object.assign({key,name,group,layers,flops:flops*layers,bytes:bytes*layers,t:r.t*layers,tc:r.tc*layers,tm:r.tm*layers,tf:r.tf*layers,bound:r.bound,I:r.I,kernels:(kern||0)*layers,
      cls:KCLS[key]||null,etaC:gk.etaC,etaM:gk.etaM},extra||{})); };
  const cm=(key,name,layers,t,bytes,extra)=>{ if(layers<=0||t<=0) return; comm.push(Object.assign({key,name,layers,t:t*layers,bytes:bytes*layers},extra||{})); };
  const lkA=link(hw,topo,tpA,1), lkN=link(hw,topo,N,1);
  const ctx = dec ? c.n : c.p;              // decode：已有上下文；prefill：本 chunk 之前的前缀

  /* ---- 层边界：AR(+add+RMSNorm) ---- */
  const nMoe=d.nMoe, moePar=c.moePar||P.defaultMoe;
  if(nMoe>0 && !MOEPARS.includes(moePar)) warn.push(Msg.of("engine_infer.warn.moePar",{v:String(moePar),opts:MOEPARS.join(" / ")}).invalid());
  // EP 合回来的只有本卡那份 token，TP 组内靠 all-gather 拼齐，不是 AR；shared expert 还是 TP 切的，那一层仍要 AR
  const epGather = tpA>1 && !e.Fs && moePar!=="tp" && moePar!=="replicated" ? nMoe : 0;
  const boundaries = m.L*2-epGather;
  // 只有 PCIe 的机器上超过 2 张卡时，vLLM 和 SGLang 都关掉自定义 all-reduce，退回 NCCL 环（topo.pcieOnly：PCIe 8 卡拓扑和集群页的 PCIe 机内互联）；
  // AR + RMSNorm 融合用的是同一套 IPC 缓冲，跟着一起关
  const pcieRing = topo.pcieOnly && tpA>2;
  const fused = tpA>1 && !pcieRing && P.arFusion(m,c,rowsG,D,hw);
  const arAlgo = pcieRing ? "ring" : rowsG*D*2 <= 512*1024 ? "oneshot" : "twoshot";
  if(tpA>1){
    cm("ar", fused?Msg.of("engine_infer.comm.arFused"):Msg.of("engine_infer.comm.ar"), boundaries, arT(rowsG*D*2,tpA,lkA,g,arAlgo), rowsG*D*2*(tpA-1)/tpA);
    if(epGather) cm("epag",Msg.of("engine_infer.comm.epag"),epGather,agT(rowsG*D*2,tpA,lkA,g),rowsG*D*2*(tpA-1)/tpA);
    op("norm",Msg.of("engine_infer.op.norm"),"misc",fused?epGather:m.L*2,8*rowsG*D,3*rowsG*D*2,pkA,1);
  } else op("norm",Msg.of("engine_infer.op.norm"),"misc",boundaries,8*rowsG*D,3*rowsG*D*2,pkA,1);
  /* mHC：hc 路残差流的混合，加上系数投影——把展平的 hc·D 投成 (2 + hc)·hc 个系数（pre、post 各 hc 个，res 是 hc × hc），
     权重是 fp32（DeepSeek-V4.1 checkpoint 的 hc_*_fn：每个子层 [24, 4·D]），每步按权重读一遍 */
  if(m.mhc){ const hcN=(2+m.mhc)*m.mhc;
    op("mhc",Msg.of("engine_infer.op.mhc"),"misc",m.L*2,4*m.mhc*rowsG*D + 2*hcN*m.mhc*D*rowsG, 2*m.mhc*rowsG*D*2 + hcN*m.mhc*D*4, pkA,2); }
  // AttnRes：每层两次（attention 前、MLP 前；SGLang kimi_k3.py 的 attn_res / mlp_res_proj），读 bank 里之前各块的快照和当前流，写一行
  if(m.attnRes) op("attnres",Msg.of("engine_infer.op.attnres"),"misc",m.L*2,2*(m.attnRes/2+1)*rowsG*D,(m.attnRes/2+2)*rowsG*D*2,pkA,1);
  /* Engram（DeepSeek-V4.1）：这几层每个 token 按 n-gram 哈希查 (ngram − 1)·heads 行（c.engramDt：checkpoint 的 fp8 或 MXFP4，
     每 32 个一个 ue8m0 scale），wkv 投成 hc 份 key 和一份 value，按门加进残差。查表的方式见 engramPlan */
  if(d.nEng>0){ const eg=m.engram, cols=d.engCols, hc=m.mhc||1, ep=engramPlan(m, hw, c, topo, N, dpA, lkN, warn), eb=dtB(ep.dt);
    const vb=rowsR*cols*eg.hd*eb, Tall=rowsR*dpA;                                     // 本卡 token 要的行；整个组的 token
    // 本卡自己查的行：local 查自己 token 的全部行；all_reduce / reduce_scatter 查自己那份行、全组的 token；all_to_all 查自己那几列、全组的 token
    const own = ep.look==="local" ? (ep.host ? vb : vb/N) : ep.look==="peer_read" ? vb/N : Tall*cols*eg.hd*eb/N;
    op("engram",Msg.of("engine_infer.op.engram",{n:cols}),"misc",d.nEng, 2*rowsG*cols*eg.hd*D*(hc+1) + 8*rowsG*hc*D, d.pEngW1*wbD + (ep.host ? 0 : own) + rowsG*(cols*eg.hd*2 + D*(hc+1)*2 + 2*hc*D*2), pkD,3);
    if(ep.host) cm("enghost",Msg.of(ep.stage?"engine_infer.comm.engHostStage":"engine_infer.comm.engHost"),d.nEng,(lb?0:g.kernelUs*1e-6) + own/(hw.host*1e9*g.etaN), own, ep.stage ? {overlap:"stage"} : null);
    const vAll=rowsG*dpA*cols*eg.hd*2, vA2a=Tall/N*cols*eg.hd*2;                      // 查回来的行按 bf16 交换
    if(ep.look==="peer_read") cm("engpeer",Msg.of("engine_infer.comm.engPeer"),d.nEng,a2aT(vb,N,lkN,g),vb*(N-1)/N);
    else if(ep.look==="all_reduce") cm("engar",Msg.of("engine_infer.comm.engAr"),d.nEng,dpA===1 ? arT(vAll,N,lkN,g,vAll<=512*1024?"oneshot":"twoshot") : arT(vAll,N,lkN,g),vAll*(N-1)/N);
    else if(ep.look==="reduce_scatter") cm("engrs",Msg.of("engine_infer.comm.engRs"),d.nEng,agT(vAll,N,lkN,g),vAll*(N-1)/N);
    else if(ep.look==="all_to_all") cm("enga2a",Msg.of("engine_infer.comm.engA2a"),d.nEng,a2aT(vA2a,N,lkN,g),vA2a*(N-1)/N);
  }

  /* ---- 全注意力层 ---- */
  let kvTokRank=0, idxTok=0;
  // DCP（见下面全注意力层之后的通信）：每种注意力记下 q 头数、每头 q / 输出的宽度、prefill 要收回来的每 token KV
  // 和 inert 一致：没有全注意力层、attention 不切 TP、GQA 而引擎不是 vLLM（SGLang 的 DCP 只做 MLA / CSA）时 dcp 不起作用
  const dcp = f.n>0 && tpA>1 && !(f.kind==="gqa" && c.engine!=="vllm") ? Math.max(1, Math.round(c.dcp||1)) : 1;
  let dcpH=0, dcpQ=0, dcpO=0, kvPre=0;
  if(dcp>1 && !c._mb && !c._rp){
    if(tpA % dcp) warn.push(Msg.of("engine_infer.warn.dcpDivides",{dcp,tp:tpA}).invalid());
    if(f.kind==="gqa" && dcp>Math.max(1, tpA/f.Hkv)) warn.push(Msg.of("engine_infer.warn.dcpGqa",{dcp,max:Math.max(1, Math.floor(tpA/f.Hkv)),hkv:f.Hkv,tp:tpA}).invalid());
    if(f.kind==="csa" && c.engine==="vllm") warn.push(Msg.of("engine_infer.warn.dcpCsa").invalid());
  }
  if(f.n>0 && f.kind==="csa"){
    /* DeepSeek-V4.1（Common.csaPlan）：每层 q 走 wq_a → q_norm → wq_b，滑窗 KV 走 wkv → kv_norm（每层都写）；
       kv source 层再跑 compressor（m 个 token 的 wkv + wgate 做 softmax 池化，写一份压缩 latent 和 index K），
       index source 层算 indexer 的 q 和权重、对压缩位置打分选 top-k（候选层之后只在候选块里打分）；
       attention 在 ≤ top-k 个压缩 latent + 最近 W 个滑窗 latent 上做 MQA（K、V 都是这个 512 维 latent），输出走分组的 wo_a 再 wo_b */
    const Pl=d.csa, H=f.H, Hl=H/tpA, dh=f.dh, kv=csaKv(m, c.kvDt), W=f.W, rp=c._rp||{};
    if(tpA>f.oG) warn.push(Msg.of("engine_infer.warn.csaGroups",{g:f.oG,tp:tpA}));
    kvTokRank = kv.lat/dcp; dcpH=Hl; dcpQ=dh; dcpO=dh;           // prefill 不收前缀：滑窗在本地，压缩 latent 的收集没有建模
    op("aqa",Msg.of("engine_infer.op.csaQa"),"attn",f.n,2*rowsG*D*(f.qLora+dh), D*(f.qLora+dh)*wbD + rowsG*(D+f.qLora+dh)*2, pkD,1);
    op("anorm",Msg.of("engine_infer.op.csaNorm"),"attn",f.n,8*rowsG*(f.qLora+dh) + 6*rowsG*(Hl+1)*f.rope, rowsG*(f.qLora+dh)*2*2 + rowsR*kv.swa, pkA,2);
    op("aqb",Msg.of("engine_infer.op.aqb"),"attn",f.n,2*rowsG*f.qLora*Hl*dh, f.qLora*Hl*dh*wbD + rowsG*(f.qLora+Hl*dh)*2, pkD,1);
    // compressor：按 ratio 分组；m > 1 时 wkv、wgate 两个 fp32 权重（按 bf16 的算力档算），m = 1 只有一个 bf16 的 wkv
    const src = Pl.filter(x=>x.kv && !(rp.kvSkip||[]).includes(x.i)).concat((rp.kvAlso||[]).map(i=>({ i, m:f.cr[i], kv:true })));
    for(const mm of [...new Set(src.map(x=>x.m))]){
      const L1=src.filter(x=>x.m===mm).length, two=mm>1?2:1, cr=rowsR/mm;
      op("acmp",Msg.of("engine_infer.op.csaCmp",{m:mm}),"attn",L1, 2*rowsG*D*dh*two + 2*cr*dh*f.idxD + 10*rowsG*dh*two,
        D*dh*two*dtB(mm>1?f.cmpDt:"bf16") + dh*f.idxD*2 + rowsG*(D+dh*two)*2 + cr*(kv.lat+kv.idx), pkA,3,{mod:"full"});
    }
    // indexer：q 和每头权重；打分按层分三类：自己的压缩位置（n / m）、候选层（同上，再按块取最大分挑候选块：多读一遍分数）、只在候选块里重打分（≤ blocks·bs）
    const nI = Pl.filter(x=>x.idx).length, cap = f.cand ? f.cand.blocks*f.cand.bs : Infinity;
    op("aidxp",Msg.of("engine_infer.op.csaIdxp"),"attn",nI,2*rowsG*(f.qLora*f.idxH*f.idxD + D*f.idxH), f.qLora*f.idxH*f.idxD*wbD + D*f.idxH*2 + rowsG*(f.qLora+D+f.idxH*f.idxD+f.idxH)*2, pkD,2,{mod:"full"});
    const kinds = [["own",x=>x.idx&&!x.cand&&!x.useCand],["cand",x=>x.cand],["re",x=>x.useCand]];
    for(const [what, sel] of kinds){ const grp=Pl.filter(sel); if(!grp.length) continue;
      for(const mm of [...new Set(grp.map(x=>x.m))]){ const L1=grp.filter(x=>x.m===mm).length;
        const all = dec ? ctx/mm : (ctx+rowsR)/mm, np = what==="re" ? (dec ? Math.min(all, cap) : capKeys(ctx/mm, rowsR/mm, cap)) : (dec ? all : (ctx+rowsR/2)/mm);
        if(all<=f.topk) continue;                                    // 压缩位置不超过 top-k：全选，不用打分
        const rd = dec ? Breq*(what==="re" ? Math.min(all, cap) : all) : all;
        op("aidx",Msg.of("engine_infer.op.csaIdx."+what,{m:mm}),"attn",L1, 2*(f.idxH*f.idxD+f.idxH)*np*rowsR, rd*kv.idx + (what==="cand" ? 3 : 2)*4*rowsR*np, peakOf(hw,f.idxDt),2,{mod:"full",scalesWith:dec?"real":undefined}); }
    }
    // attention 核心：按 ratio 分组（m = 0 只有滑窗）。每个 query 看 min(top-k, n / m) 个压缩 latent + min(W, n) 个滑窗 latent
    for(const mm of [...new Set(Pl.map(x=>x.m))]){ const L1=Pl.filter(x=>x.m===mm).length, mod = mm>0 ? "full" : "swa";
      const kc = mm>0 ? (dec ? Math.min(f.topk, ctx/mm) : capKeys(ctx/mm, rowsR/mm, f.topk)) : 0, ks = dec ? Math.min(W, ctx) : capKeys(ctx, rowsR, W);
      const tag = Msg.of(mm>0 ? "engine_infer.op.acoreCsa" : "engine_infer.op.acoreCsaSwa",{kernel:P.kernels.csa,m:mm,w:W,k:f.topk});
      if(dec) op("acore",tag,"attn",L1, 4*Hl*dh*(kc+ks)*Breq*w, Breq*(kc*kv.lat/dcp + ks*kv.swa) + rowsR*Hl*dcp*dh*2*2, pkA,2,{scalesWith:"real",mod});
      else op("acore",tag,"attn",L1, 4*Hl*dh*(kc+ks)*rowsR, (mm>0 ? Math.min((ctx+rowsR)/mm, rowsR*kc)*kv.lat : 0) + (Math.min(ctx,W)+rowsR)*kv.swa + rowsR*Hl*dh*2*2, pkA,2,{mod});
    }
    op("aoa",Msg.of("engine_infer.op.csaOa",{g:f.oG,r:f.oLora}),"attn",f.n,2*rowsG*H*dh*f.oLora/tpA, H*dh*f.oLora/tpA*dtB(f.oDt) + rowsG*(Hl*dh+f.oG*f.oLora/tpA)*2, peakOf(hw,f.oDt),1);
    op("aout",Msg.of("engine_infer.op.csaOb"),"attn",f.n,2*rowsG*f.oG*f.oLora*D/tpA, f.oG*f.oLora*D/tpA*wbD + rowsG*(f.oG*f.oLora/tpA+D)*2, pkD,1);
  } else if(f.n>0){
    if(f.kind==="gqa"){
      const qo=qoW(f), kvo=2*f.Hkv*f.dh, Hl=f.Hq/tpA, kvHeads=Math.max(1,f.Hkv/tpA);
      const nSwa=f.swa||0, nGlob=f.n-nSwa;
      kvTokRank = 2*kvHeads*f.dh*kvb/dcp; dcpH=Hl; dcpQ=f.dh; dcpO=f.dh; kvPre=2*kvHeads*f.dh*kvb;
      op("aproj",f.gate?Msg.of("engine_infer.op.aprojGate"):Msg.of("engine_infer.op.aproj"),"attn",f.n,2*rowsG*D*(qo+kvo)/tpA, D*(qo+kvo)/tpA*wbD + rowsG*(D+(qo+kvo)/tpA)*2, pkD,1);
      op("aprol",Msg.of("engine_infer.op.aprol"),"attn",f.n,10*rowsG*(Hl+2*kvHeads)*f.dh, rowsG*(Hl+2*kvHeads)*f.dh*2*2 + rowsR*kvTokRank, pkA, c.engine==="sglang"&&f.gate?2:1);
      // 滑窗层只读最近 W 个 token 的 KV；全局层读整段
      const core=(layers,keyCap,tag)=>{ const cx=Math.min(ctx,keyCap);
        const mod = keyCap<Infinity ? "swa" : "full";     // 架构页按层类型拆模块用
        if(dec) op("acore",Msg.of("engine_infer.op.acoreDecode",{kernel:P.kernels.gqa,layers:tag}),"attn",layers, 4*Hl*f.dh*cx*Breq*w, Breq*cx*kvTokRank + rowsR*Hl*dcp*f.dh*2*2, pkA,1,{scalesWith:"real",mod});
        else op("acore",Msg.of("engine_infer.op.acorePrefill",{layers:tag}),"attn",layers, 4*Hl*f.dh*rowsR*capKeys(ctx,rowsR,keyCap), (cx+rowsR)*kvTokRank*dcp + rowsR*Hl*f.dh*2*2, pkA,1,{mod}); };
      core(nGlob, Infinity, nSwa?Msg.of("engine_infer.op.layersGlobal"):"");
      if(nSwa) core(nSwa, f.W, Msg.of("engine_infer.op.layersSwa",{w:f.W}));
      if(f.gate) op("agate",f.gateHead?Msg.of("engine_infer.op.agateHead"):Msg.of("engine_infer.op.agate"),"attn",f.n,2*rowsG*Hl*f.dh,3*rowsG*Hl*f.dh*2,pkA,1);
      op("aout",Msg.of("engine_infer.op.aout"),"attn",f.n,2*rowsG*f.Hq*f.dh*D/tpA, f.Hq*f.dh*D/tpA*wbD + rowsG*(f.Hq*f.dh/tpA+D)*2, pkD,1);
    } else {
      const H=f.H, Hl=H/tpA, dq=f.nope+f.rope, lat=f.kvLora+f.rope;
      kvTokRank = lat*kvb/dcp; dcpH=Hl; dcpQ=lat; dcpO=f.kvLora; kvPre=lat*kvb;   // latent 无头维度，TP 切不开，只能靠 DCP
      op("aqa",Msg.of("engine_infer.op.aqa"),"attn",f.n,2*rowsG*D*(f.qLora+lat), D*(f.qLora+lat)*wbD + rowsG*(D+f.qLora+lat)*2, pkD,1);
      op("anorm",Msg.of("engine_infer.op.anorm"),"attn",f.n,8*rowsG*(f.qLora+f.kvLora), rowsG*(f.qLora+f.kvLora)*2*2 + rowsR*kvTokRank, pkA,2);
      op("aqb",Msg.of("engine_infer.op.aqb"),"attn",f.n,2*rowsG*f.qLora*Hl*dq, f.qLora*Hl*dq*wbD + rowsG*(f.qLora+Hl*dq)*2, pkD,1);
      if(f.gate) op("agp",Msg.of("engine_infer.op.agp"),"attn",f.n,2*rowsG*D*Hl*f.vh, D*Hl*f.vh*wbD, pkD,1);
      if(f.kind==="dsa"){
        const ip=f.idxH*f.idxD+f.idxD+f.idxH, kp=f.kpool||1;
        idxTok = (f.idxD+4)/kp;                                       // fp8 index-K + scale，按池存
        op("aidxp",Msg.of("engine_infer.op.aidxp"),"attn",f.n,2*rowsG*(f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH), (f.qLora*f.idxH*f.idxD+D*f.idxD+D*f.idxH)*wbD + rowsG*ip*2, pkD,2);
        const nk = dec ? ctx : ctx+rowsR/2;
        if(dec ? ctx>f.topk : ctx+rowsR>f.topk)
          op("aidx",Msg.of("engine_infer.op.aidx",{pools:kp}),"attn",f.n, 2*f.idxH*f.idxD*(nk/kp)*rowsR, (dec?Breq*ctx:(ctx+rowsR))*idxTok + 2*4*rowsR*nk/kp, peakOf(hw,"fp8"),2,{scalesWith:dec?"real":undefined});
      }
      // absorb：q 乘 W_uk 到 latent，attention 在 latent 上做 MQA，再乘 W_uv
      const absorb = dec || f.kind==="dsa" && !!P.dsaAbsorbPrefill;
      if(absorb){
        op("auk",Msg.of("engine_infer.op.auk"),"attn",f.n,2*rowsG*Hl*f.nope*f.kvLora, Hl*f.nope*f.kvLora*2 + rowsG*Hl*(f.nope+f.kvLora)*2, pkA,1);
        const nKV = f.kind==="dsa" ? (dec ? Math.min(ctx, f.topk+3) : capKeys(ctx, rowsR, f.topk+3)) : (dec?ctx:ctx+rowsR/2);
        const tag = f.kind==="dsa" ? Msg.of("engine_infer.op.acoreDsa",{kernel:P.kernels.dsa}) : Msg.of("engine_infer.op.acoreMla",{kernel:P.kernels.mla});
        if(dec) op("acore",tag,"attn",f.n, 2*Hl*(lat+f.kvLora)*nKV*Breq*w, Breq*nKV*kvTokRank + rowsR*Hl*dcp*(lat+f.kvLora)*2, pkA,2,{scalesWith:"real"});
        else op("acore",tag,"attn",f.n, 2*Hl*(lat+f.kvLora)*nKV*rowsR, (ctx+rowsR)*kvTokRank*dcp, pkA,2);
        op("auv",Msg.of("engine_infer.op.auv"),"attn",f.n,2*rowsG*Hl*f.kvLora*f.vh, Hl*f.kvLora*f.vh*2 + rowsG*Hl*(f.kvLora+f.vh)*2, pkA,1);
      } else {
        // prefill：MHA 形式，kv_b 把新 token 展开成每头 K/V；mlaReexpand 的引擎每个 chunk 还把整个前缀的 latent 重新展开
        // （SGLang MHA_CHUNKED_KV、vLLM MLAAttention._compute_prefill_context）
        const reexp = P.mlaReexpand && ctx>0;
        const expTok = rowsR + (reexp?ctx:0);
        op("akvb",reexp?Msg.of("engine_infer.op.akvbReexp"):Msg.of("engine_infer.op.akvb"),"attn",f.n,2*expTok*f.kvLora*Hl*(f.nope+f.vh), f.kvLora*Hl*(f.nope+f.vh)*wbD + expTok*(f.kvLora + Hl*(f.nope+f.vh))*2, pkD,1);
        const nKV = f.kind==="dsa" ? capKeys(ctx, rowsR, f.topk) : ctx+rowsR/2;
        op("acore",f.kind==="dsa"&&ctx+rowsR>f.topk?Msg.of("engine_infer.op.acoreSparsePrefill"):Msg.of("engine_infer.op.acoreMhaPrefill",{qk:dq,v:f.vh}),"attn",f.n, 2*Hl*(dq+f.vh)*rowsR*nKV, expTok*Hl*(dq+f.vh)*2 + rowsR*Hl*(dq+f.vh)*2, pkA,1);
      }
      if(f.gate) op("agate",Msg.of("engine_infer.op.agate"),"attn",f.n,2*rowsG*Hl*f.vh,3*rowsG*Hl*f.vh*2,pkA,1);
      op("aout",Msg.of("engine_infer.op.aout"),"attn",f.n,2*rowsG*H*f.vh*D/tpA, H*f.vh*D/tpA*wbD + rowsG*(Hl*f.vh+D)*2, pkD,1);
    }
  }

  /* DCP（decode context parallel：vLLM -dcp，SGLang --dcp-size）：KV 按 token 轮流放在 attention TP 组里的 dcp 张卡上（token i 在 i mod dcp），
     MLA 那一个 latent 头不再每卡存一份；GQA 只切 TP 大于 KV 头数时被复制的那部分（dcp ≤ tp / Hkv）。decode 每个全注意力层：
     q 先在 DCP 组里 all-gather（每卡从 Hl 个头变成 Hl·dcp 个），各卡拿这些头对自己那份 token 做 attention，再按 LSE 把部分输出合起来。
     合法有两种（c.dcpComm，空 = 引擎默认 P.dcpComm）：ag_rs = LSE all-gather + 输出按头 reduce-scatter（vLLM 默认 bf16，SGLang 是 fp32）；
     a2a = 输出和 LSE 打包成 [dcp, B, Hl, D + 2] 的 bf16 做一次 all-to-all（SGLang 默认，vLLM PR 34883）。
     prefill：前缀分在各卡，每块先把前缀的 KV all-gather 回来再算（attention 看整个前缀，所以上面 prefill 的 KV 读按整份算） */
  if(dcp>1 && dcpH>0){
    const lkD=link(hw,topo,dcp,1), Hg=dcpH*dcp, algo=c.dcpComm||P.dcpComm||"ag_rs";
    if(dec){
      const q=rowsG*Hg*dcpQ*2;
      cm("dcpq",Msg.of("engine_infer.comm.dcpQ",{dcp}),f.n,agT(q,dcp,lkD,g),q*(dcp-1)/dcp);
      if(algo==="a2a"){ const pk=rowsG*Hg*(dcpO+2)*2; cm("dcpo",Msg.of("engine_infer.comm.dcpA2a"),f.n,a2aT(pk,dcp,lkD,g),pk*(dcp-1)/dcp); }
      else { const ls=rowsG*Hg*4, ob=P.dcpOutB||2, o=rowsG*Hg*dcpO*ob;
        cm("dcplse",Msg.of("engine_infer.comm.dcpLse"),f.n,agT(ls,dcp,lkD,g),ls*(dcp-1)/dcp);
        cm("dcpo",Msg.of("engine_infer.comm.dcpRs",{bits:ob*8}),f.n,agT(o,dcp,lkD,g),o*(dcp-1)/dcp); }
    } else if(ctx>0 && kvPre>0){ const kb=ctx*kvPre; cm("dcpkv",Msg.of("engine_infer.comm.dcpKv"),f.n,agT(kb,dcp,lkD,g),kb*(dcp-1)/dcp); }
  }

  /* ---- 线性注意力层（GDN / KDA） ---- */
  let Sl=0, Cl=0;
  if(l.n>0){
    const Hv=l.Hv/tpA; Sl = Hv*l.dk*l.dv*sb; Cl = d.linConvDim/tpA*(l.conv-1)*2;
    const wIn = D*d.linIn/tpA + d.linFa + d.linFb/tpA;   // KDA 的 f_a 每个 rank 都有一份
    op("lproj",Msg.of("engine_infer.op.lproj"),"lin",l.n,2*rowsG*wIn, wIn*wbD + rowsG*(D+d.linIn/tpA)*2, pkD, l.kind==="KDA"?2:1);
    if(dec){
      const mult = specK>0 ? P.stateVerify(m,w) : P.stateDecode(m);
      op("lconv",Msg.of("engine_infer.op.lconvDecode"),"lin",l.n,2*l.conv*rowsR*d.linConvDim/tpA, Breq*Cl*2 + rowsR*d.linConvDim/tpA*2*2, pkA,1,{scalesWith:"real"});
      op("lcore",Msg.of("engine_infer.op.lcoreDecode",{kind:l.kind,kernel:l.kind==="KDA"?P.kernels.kda:P.kernels.gdn}),"lin",l.n, 8*Hv*l.dk*l.dv*rowsR, mult*Breq*Sl + rowsR*Hv*(2*l.dk+2*l.dv)*2, pkA, c.engine==="tokenspeed"&&l.kind==="KDA"?2:1,{scalesWith:"real",stateMult:mult});
    } else {
      const C=64, hm=P.hMat(m,hw);
      op("lconv",Msg.of("engine_infer.op.lconvPrefill"),"lin",l.n,2*l.conv*rowsR*d.linConvDim/tpA, rowsR*d.linConvDim/tpA*2*2, pkA,1);
      op("lcore",hm?Msg.of("engine_infer.op.lcorePrefillH",{kind:l.kind}):Msg.of("engine_infer.op.lcorePrefillFused",{kind:l.kind}),"lin",l.n,
         rowsR*Hv*(6*l.dk*l.dv + C*(6*l.dk+4*l.dv) + C*C/3), rowsR*Hv*(2*l.dk+2*l.dv+l.dk)*2*2 + (hm? 2*rowsR*Hv*l.dk*l.dv*2/C : 0) + 2*Sl, pkA, hm?6:3);
    }
    op("lnorm",Msg.of("engine_infer.op.lnorm"),"lin",l.n,10*rowsG*Hv*l.dv,3*rowsG*Hv*l.dv*2,pkA,1);
    op("lout",Msg.of("engine_infer.op.lout"),"lin",l.n,2*rowsG*l.Hv*l.dv*D/tpA, l.Hv*l.dv*D/tpA*wbD + rowsG*(Hv*l.dv+D)*2, pkD,1);
  }

  /* ---- MoE 层 ---- */
  const De=d.De;
  const Tg = rowsG*dpA;                     // MoE 一侧看到的全局 token（含补齐行）
  const Tloc = rowsG;
  const routerT = moePar==="replicated" ? Tg : Tloc;
  op("router",Msg.of("engine_infer.op.router"),"moe",nMoe,2*routerT*D*e.E + 8*routerT*e.E, D*e.E*2 + routerT*(D*2+e.E*4), pkA, 2);
  if(e.Dlat) op("latd",Msg.of("engine_infer.op.latd",{dim:e.Dlat}),"moe",nMoe,2*Tloc*D*e.Dlat, D*e.Dlat*2 + Tloc*(D+e.Dlat)*2, pkA,1);
  const red=moePar==="tp"||N<=1 ? 0 : c.redundant||0, Eloc=(e.E+red)/N;   // tp 切的每卡都有全部 expert 的切片，没有“多放几个”这回事
  // 碰到的 expert：建模按均匀路由的期望覆盖；下限只算一定会碰到的——每个 token 选 k 个不同的 expert，所有 token 可以选同一组，
  // TP 切时每卡是这 k 个的切片，按 expert 切时最忙的 rank 至少有 ceil(k / N) 个
  const cv = lb ? (moePar==="tp" ? Math.min(1, e.k/e.E) : Math.min(1, Math.ceil(e.k/N)/Eloc)) : cover(e.k,e.E,Tg);
  const touched = moePar==="tp" ? e.E*cv : Eloc*cv;   // tp：每卡持有全部 expert 的 F/N 切片
  const wSlice = moePar==="tp" ? 3*De*e.F/N : 3*De*e.F;
  const rowsExp = Tg*e.k/N*(moePar==="tp" || lb || N<=1 ? 1 : c.imb||1);   // tp：每卡都有全部 expert 的切片，负载天然均衡；单卡没有组间不均
  const eA = eActOf(c,hw);
  if(c.qSep && eA!=="bf16") op("moeq",Msg.of("engine_infer.op.moeq"),"moe",nMoe,4*Tloc*De, Tloc*De*(2+dtB(eA)), pkA,1);
  // SM100 上 FP4 MoE（TRT-LLM-gen）的 hidden 维按 256 补零（vLLM PR 39510）：补出来的部分照样算、照样读
  const dPad = !lb && (c.wDt==="nvfp4"||c.wDt==="mxfp4") && sm10x(hw) ? Math.ceil(De/256)*256/De : 1;
  op("moe",Msg.of("engine_infer.op.moe",{kernel:P.kernels.moe}),"moe",nMoe, 2*rowsExp*3*De*e.F*dPad, touched*wSlice*wbE*dPad + rowsExp*(De*2+3*e.F*2/(moePar==="tp"?N:1)), pkE, 3,{touched,cover:cv,dPad,
    rowsPer: moePar==="tp" ? Tg*e.k/Math.max(1e-9, e.E*cv) : rowsExp/Math.max(1e-9, touched)});   // 每个被碰到的 expert 的 GEMM 有多少行（tp 切的是列，行数不变）
  if(e.Fs>0){ const Fl=e.Fs/tpA;
    op("shared",e.sharedGate?Msg.of("engine_infer.op.sharedGate"):Msg.of("engine_infer.op.shared"),"moe",nMoe,6*Tloc*D*Fl, 3*D*Fl*wbD + Tloc*(D+3*Fl)*2, pkD,2,
      {overlap: (P.sharedOverlap==="graph"&&graph&&dec) || (P.sharedOverlap==="small"&&Tloc<=256) ? "moe" : false}); }
  if(e.Dlat){
    // 算法和层边界的 AR 一样按消息大小选：大消息（prefill）用 one-shot 要读 N−1 份整消息，会比 two-shot 慢好几倍
    if(tpA>1){ const lb=Tloc*e.Dlat*2; cm("latar",Msg.of("engine_infer.comm.latar",{dim:e.Dlat}),nMoe,arT(lb,tpA,lkA,g,lb<=512*1024?"oneshot":"twoshot"),lb*(tpA-1)/tpA); }
    // SGLang K3 在 TP8、不超过 12 行时把复制的 latent up 换成按列切的 GEMV + multicast all-gather（k3_ar_fusion.gemm_ag_up_proj）
    const latAg = c.engine==="sglang" && m.family==="kimi" && fused && tpA===8 && sm10x(hw) && Tloc<=12, lw = latAg ? tpA : 1;
    op("latu",Msg.of("engine_infer.op.latu",{dim:e.Dlat}),"moe",nMoe,2*Tloc*e.Dlat*D/lw, e.Dlat*D*2/lw + Tloc*(D+e.Dlat)*2, pkA,2);
    if(latAg) cm("latag",Msg.of("engine_infer.comm.latag"),nMoe,agT(Tloc*D*2,tpA,lkA,g),Tloc*D*2*(tpA-1)/tpA);
  }
  if(m.dense.n>0) op("dense",Msg.of("engine_infer.op.dense"),"moe",m.dense.n,6*rowsG*D*m.dense.F/tpA, 3*D*m.dense.F/tpA*wbD + rowsG*(D+3*m.dense.F/tpA)*2, pkD,2);
  // MoE 通信
  const lkE = link(hw,topo,N,1);
  if(N>1){
    if(moePar==="tp" || moePar==="replicated"){
      if(dpA>1){ cm("dpg",Msg.of("engine_infer.comm.dpg"),nMoe,agT(Tg*De*2,dpA,lkE,g),Tg*De*2*(dpA-1)/dpA); cm("dps",Msg.of("engine_infer.comm.dps"),nMoe,agT(Tg*De*2,dpA,lkE,g),Tg*De*2*(dpA-1)/dpA); }
      else if(!e.Dlat) {/* MoE 结果的 AR 并进层边界的 AR（已计入） */}
    } else if(moePar==="ag"){
      cm("dpg",Msg.of("engine_infer.comm.epAg"),nMoe,agT(Tg*De*2,N,lkE,g),Tg*De*2*(N-1)/N);
      cm("dps",Msg.of("engine_infer.comm.epRs"),nMoe,agT(Tg*De*2,N,lkE,g),Tg*De*2*(N-1)/N);
    } else {
      const Tr = Tg/N;                                                 // 每卡只发自己那份 token：TP 组内按 rank 切开
      const sentD = Tr*e.k*De*(moePar==="deepep"&&c.dispFp8 ? 1+4/128 : 2), sentC = Tr*e.k*De*2;
      const al = moePar==="deepep" ? (dec?g.a2aLLus:g.alphaNv) : undefined;
      const fusedA2A = moePar==="fused";
      const tag = moePar==="deepep" ? (dec?Msg.of("engine_infer.comm.a2a.deepepLL"):Msg.of("engine_infer.comm.a2a.deepepNormal")) : moePar==="standard" ? Msg.of("engine_infer.comm.a2a.standard") : moePar==="hybridep" ? Msg.of("engine_infer.comm.a2a.hybridep") : Msg.of("engine_infer.comm.a2a.fused");
      cm("disp",Msg.of("engine_infer.comm.dispatch",{a2a:tag}),nMoe,a2aT(sentD,N,lkE,g,al),sentD*(N-1)/N,{overlap:fusedA2A?"moe":(c.tbo?"tbo":false)});
      cm("comb",Msg.of("engine_infer.comm.combine",{a2a:tag}),nMoe,a2aT(sentC,N,lkE,g,al),sentC*(N-1)/N,{overlap:fusedA2A?"moe":(c.tbo?"tbo":false)});
      if(moePar==="standard" && !lb) cm("cnt",Msg.of("engine_infer.comm.cnt"),nMoe,(lkE.intra?g.alphaNv:g.alphaIb)*1e-6 + (c.hostSyncUs??20)*1e-6, N*e.E*8/N);
    }
  }

  /* ---- 每步一次：lm_head、logits 收集、采样 ---- */
  const lmRows = dec ? (P.lmInGraph&&graph ? rowsG : rowsR) : c.reqPerChunk??1;   // prefill：只有收尾的 chunk 才要 lm_head 和采样，每个结束的请求一行
  if(lmRows>0){
  op("lm",dec&&P.lmInGraph&&graph?Msg.of("engine_infer.op.lmGraph"):Msg.of("engine_infer.op.lm"),"lm",1,2*lmRows*D*m.V/tpA, D*m.V/tpA*wbD + lmRows*m.V/tpA*4, pkD,1);
  if(tpA>1) cm("lmg",Msg.of("engine_infer.comm.lmg"),1,agT(lmRows*m.V*4,tpA,lkA,g),lmRows*m.V*4*(tpA-1)/tpA);
  const sampRows = dec ? (P.samplerInGraph&&graph ? rowsG : rowsR) : lmRows;
  op("samp",P.samplerInGraph?Msg.of("engine_infer.op.sampGraph"):Msg.of("engine_infer.op.sampEager"),"lm",1,5*sampRows*m.V,4*sampRows*m.V*4,pkA,3);
  }

  /* ---- 汇总 ---- */
  const F = finish(ops, comm, c, g, graph, dpA);
  // 每 token 的 KV（单个 attention rank，整个模型）：kvFull 随上下文增长，kvSwa 每个请求只留最近 kvW 个 token
  const kvAgg = f.kind==="csa" ? csaKvOut(m, c) : { kvFull: f.n>0 ? (f.n-(f.kind==="gqa" ? f.swa||0 : 0))*(kvTokRank+idxTok) : 0,
    kvSwa: f.kind==="gqa" ? (f.swa||0)*kvTokRank : 0, kvW: f.kind==="gqa" && f.swa ? f.W : 0 };
  return Object.assign({ ops, comm }, F, { graph, rowsG, rowsR, Breq, Bp, w, specK, tpA, dpA, N, Tg, kvTokRank, idxTok, Sl, Cl, fused, warn, moePar, P }, kvAgg);
}

/* 一步的汇总：算子和通信的时间怎么叠起来。step 和 mixStep 共用 */
function finish(ops, comm, c, g, graph, dpA){
  const lb = !!g.lb;
  const moeT = ops.filter(o=>o.key==="moe").reduce((s,o)=>s+o.t,0);
  let tOps=0;
  for(const o of ops) if(o.overlap!=="moe") tOps+=o.t;
  const sharedT = ops.filter(o=>o.overlap==="moe").reduce((s,o)=>s+o.t,0);
  let tComm=0, fusedComm=0, tboComm=0, stageComm=0;
  for(const x of comm){ if(x.overlap==="moe") fusedComm+=x.t; else if(x.overlap==="tbo") tboComm+=x.t; else if(x.overlap==="stage") stageComm+=x.t; else tComm+=x.t; }
  const exposedOverlap = Math.max(0, sharedT + fusedComm - moeT);  // 共享 expert 与融合 a2a 都藏在 routed expert 后面
  const tComp = tOps;
  // TBO：建模口径在 step 里切成两个 micro-batch（tboMerge 重算这一项）；下限口径不切，整批的 a2a 全部可以藏在整批的计算后面
  const exposedTbo = c.tbo && lb ? Math.max(0,tboComm - tComp) : tboComm;
  // Engram staging：拷贝在上一步 forward 期间进行，按上一步和这一步一样长算，只有比整步还长的部分露出来
  const exposedStage = Math.max(0, stageComm - (tComp + tComm + exposedOverlap + exposedTbo));
  let gpu = tComp + tComm + exposedOverlap + exposedTbo + exposedStage;
  const nKern = ops.reduce((s,o)=>s+o.kernels,0) + comm.length*1;
  // graph 里 kernel 之间的间隔：开 PDL（programmatic dependent launch）时下一个 kernel 的开头和上一个的结尾重叠，按全部去掉算（上限）
  const gapT = graph && !c.pdl ? nKern*g.gapUs*1e-6 : 0;
  gpu += gapT;
  // 下限的资源守恒：关键路径之外，计算、HBM、通信各自的总量也是下限（重叠的算子不能各自吃满同一份资源）
  let bound = null;
  if(lb){ bound = { path:gpu, compute:ops.reduce((s,o)=>s+o.tc,0), hbm:ops.reduce((s,o)=>s+o.tm,0), comm:comm.reduce((s,x)=>s+x.t,0) };
    gpu = Math.max(bound.path, bound.compute, bound.hbm, bound.comm); bound.by = Object.keys(bound).find(k => bound[k]===gpu); }
  const cpuLaunch = graph ? g.graphLaunchUs*1e-6 : nKern*g.launchUs*1e-6;
  const dpSync = (c.engine==="vllm"||c.engine==="titan") && dpA>1 ? g.dpSyncUs*1e-6 : 0;
  const stepT = Math.max(gpu + (graph?g.graphLaunchUs*1e-6:0), cpuLaunch, g.schedUs*1e-6) + dpSync;
  return { tComp, tComm, exposedOverlap, exposedTbo, exposedStage, gpu, gapT, stepT, bound, nKern, cpuLaunch, dpSync };
}

/* Engram 表放哪、怎么查（SGLang --engram-config，docs/advanced_features/engram.mdx）：
   · placement（engramAt）：gpu 显存；host 主机的 pinned 内存，GPU 原地读，不占显存
   · replication（engramRepl，只对 host）：none 一份表按行切在 TP 组的 N 张卡上，每卡一份分片；per_node 每台机器一份整表，组内各卡映射同一份
     （要求 TP 组在一台机器里，--nnodes 1）。GPU 表总是 none
   · lookup（engramLookup，只在 none 时）：auto 在 GPU 表、组在一个 NVLink 域时取 peer_read（每卡直接从持有那份的卡上读自己 token 的行，
     不走 collective）；否则 TP attention 时 all_reduce，DP attention 时列数能整除 N 就 all_to_all（只搬 1 / N 的字节），不能整除时 all_reduce
   · staging（engramStage，SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING，srt/layers/engram_staging.py）：host + per_node、没有 ATS（非 Grace）时，
     调度器在上一步 forward 期间把 prefill 批要的行收进 pinned 内存、在旁路 stream 上拷到 GPU，这一步只剩反量化
   vLLM 走自己的实现（vllm/config/engram.py：cpu_offload 默认开、DP 副本间共享主机表），这里不细分，仍按此前的近似：
   host 时每卡直接读自己 token 的行，GPU 时 TP attention 组内 all-reduce、DP attention 时 all-to-all */
function engramPlan(m, hw, c, topo, N, dpA, lkN, warn){
  const host=c.engramAt==="host", sg=c.engine==="sglang", cols=derive(m).engCols, dt=c.engramDt||"mxfp8";
  const repl = host ? (sg ? c.engramRepl||"none" : "per_node") : "none";
  if(sg && host && repl==="per_node" && N>(topo.node||topo.gpus)) warn.push(Msg.of("engine_infer.warn.engNode",{n:N,node:topo.node||topo.gpus}).invalid());
  let look = "local";
  if(N>1 && repl==="none"){
    const div = cols%N===0, fallback = dpA===1 ? "all_reduce" : div ? "all_to_all" : "all_reduce";
    const want = sg ? c.engramLookup||"auto" : (dpA===1 ? "all_reduce" : "all_to_all");
    look = want==="auto" ? (!host && lkN.intra ? "peer_read" : fallback) : want;
    if(look==="peer_read" && (host || !lkN.intra)){ warn.push(Msg.of("engine_infer.warn.engPeer").invalid()); look=fallback; }
    if((look==="all_to_all"||look==="reduce_scatter") && dpA===1){ warn.push(Msg.of("engine_infer.warn.engDp",{v:look}).invalid()); look="all_reduce"; }
    if(look==="all_to_all" && !div){ warn.push(Msg.of("engine_infer.warn.engCols",{cols,n:N}).invalid()); look="all_reduce"; }
  }
  const stageAsk = sg && host && repl==="per_node" && !!c.engramStage;
  if(stageAsk && hw.grace) warn.push(Msg.of("engine_infer.warn.engStageAts"));
  return { host, repl, look, dt, stage: stageAsk && !hw.grace && c.phase!=="decode" };
}

/* bounded replay 用：取模型的第 a 到 b − 1 层（层号仍按整个模型算，Common.csaPlan 用 full.from）；Engram 和 dense 层按层号筛 */
function csaSlice(m, a, b){
  const eg = m.engram, keep = eg ? eg.layers.map(i => i>=a && i<b) : null;
  return Object.assign({}, m, { L:b-a, full:Object.assign({}, m.full, { n:b-a, from:a }), dense:{ n:Math.max(0, Math.min(b, m.dense.n)-a), F:m.dense.F },
    engram: eg ? Object.assign({}, eg, { layers:eg.layers.filter((_,j)=>keep[j]), rows:eg.rows.filter((_,j)=>keep[j]) }) : eg });
}
/* CSA 的 KV 汇总（整个模型）：压缩 latent + index K 随上下文增长（DCP 切开），每层的滑窗 KV 每个请求最多 W 个 token */
function csaKvOut(m, c){ const kv = csaKv(m, c.kvDt);
  return { kvFull:(kv.main + kv.index)/Math.max(1, c.dcp||1), kvSwa:kv.swaTok, kvW:kv.W, kvLayers:{ full:kv.src, swa:kv.layers } }; }

/* 两步合成一步：同一个算子 / 通信（key、mod、名字都相同）的时间、字节、kernel 数相加；layers 为 true 时层数也相加（前后两段），
   否则不变（TBO 的两个 micro-batch 跑的是同一些层） */
function mergeOps(xs, ys, layers){
  const id = o => o.key+"|"+(o.mod||"")+"|"+JSON.stringify(o.name), out=[], at=new Map();
  for(const x of xs.concat(ys)){ const k=id(x), o=at.get(k);
    if(!o){ const n=Object.assign({}, x); at.set(k,n); out.push(n); continue; }
    for(const f of ["flops","bytes","t","tc","tm","tf","kernels"].concat(layers ? ["layers"] : [])) if(typeof x[f]==="number") o[f]+=x[f]; }
  for(const o of out) if(typeof o.flops==="number" && o.bytes>0) o.I=o.flops/o.bytes;
  return out;
}
const dedupWarn = ws => { const seen=new Set(); return ws.filter(w => { const k=JSON.stringify(w); if(seen.has(k)) return false; seen.add(k); return true; }); };
/* bounded replay 的前后两段依次跑：时间、kernel 数、launch 开销都相加，DP 同步只算一次；下限口径两段各自的下限相加 */
function seqMerge(a, b, c, g, warn){
  const sum = k => (a[k]||0)+(b[k]||0);
  const bound = a.bound && b.bound ? (x => Object.assign(x, { by:Object.keys(x).reduce((p,k)=>x[k]>x[p]?k:p, "path") }))(
    { path:a.bound.path+b.bound.path, compute:a.bound.compute+b.bound.compute, hbm:a.bound.hbm+b.bound.hbm, comm:a.bound.comm+b.bound.comm }) : null;
  return Object.assign({}, a, { ops:mergeOps(a.ops,b.ops,true), comm:mergeOps(a.comm,b.comm,true), tComp:sum("tComp"), tComm:sum("tComm"),
    exposedOverlap:sum("exposedOverlap"), exposedTbo:sum("exposedTbo"), exposedStage:sum("exposedStage"), gpu:sum("gpu"), gapT:sum("gapT"), nKern:sum("nKern"), cpuLaunch:sum("cpuLaunch"),
    stepT:a.stepT + b.stepT - (b.dpSync||0), bound, warn:dedupWarn(warn.concat(a.warn, b.warn)) });
}

/* 混合步（chunked prefill 和 decode 放进同一次 forward）：decode 的 B 条请求各一行，加上一块 Cp 个 token 的 prefill（前缀 p）。
   GEMM、MoE、norm 和通信按 B·w + Cp 行算一遍（权重只读一遍、a2a 只发一遍），从“Cp + decode 行”的 prefill 步里取；
   和上下文有关的 attention 部分（读 KV 的核心、indexer 打分、MLA 的吸收 / 展开、CSA 的压缩、线性注意力的卷积和状态）两边各算各的：
   decode 那份取 decode 步，prefill 那份取单独一块 Cp 的 prefill 步；lm_head 和采样只给 decode 的行（这一块不是最后一块），DCP 的 q 收集和输出合并也只有 decode 的行。
   三步都不开 CUDA graph 补齐 decode 的行（vLLM 的混合批走 piecewise graph，attention 在 graph 外），投机和 TBO 不进混合步；
   DeepSeek-V4.1 的 bounded replay 只对 prefill 的最后几个 token 有效，混合步按关掉算 */
const MIXK = new Set(["acore","aidx","auk","auv","akvb","acmp","lconv","lcore","lm","samp"]), MIXC = new Set(["lmg","dcpq","dcplse","dcpo"]);
function mixStep(m, hw, topo, c, g, Cp, p){
  const base = Object.assign({}, c, { specK:0, tbo:false, replay:false });
  const dec = step(m,hw,topo,Object.assign({}, base, { phase:"decode", graph:false }), g);
  const rows = dec.rowsR + Cp;
  const all = step(m,hw,topo,Object.assign({}, base, { phase:"prefill", C:rows, p, reqPerChunk:0 }), g);
  const pre = step(m,hw,topo,Object.assign({}, base, { phase:"prefill", C:Cp, p, reqPerChunk:0, graph:false }), g);
  const pick = (r, ctxOps) => r.ops.filter(o => MIXK.has(o.key)===ctxOps);
  const ops = pick(all,false).concat(pick(pre,true), pick(dec,true));
  const comm = all.comm.filter(x => !MIXC.has(x.key)).concat(dec.comm.filter(x => MIXC.has(x.key)));
  const F = finish(ops, comm, base, g, all.graph, all.dpA);
  return Object.assign({}, all, { ops, comm }, F, { mix:{ B:dec.Breq, Cp, p, rows, dec:dec.stepT, pre:pre.stepT }, Breq:dec.Breq, warn:dedupWarn(all.warn.concat(pre.warn, dec.warn)) });
}

/* TBO 的两个 micro-batch 合成一步：同一个算子 / 通信的时间和字节相加（层数不变），kernel 数相加；
   a 的 a2a 藏在 b 的计算后面（最多 g.etaTbo 那么多），b 的藏在 a 的后面 */
function tboMerge(a, b, c, g, full){
  const ops=mergeOps(a.ops,b.ops), comm=mergeOps(a.comm,b.comm);
  const tbo = r => r.comm.filter(x=>x.overlap==="tbo").reduce((s,x)=>s+x.t,0);
  const aA=tbo(a), aB=tbo(b);
  const exposedTbo = Math.max(0, aA - g.etaTbo*b.tComp) + Math.max(0, aB - g.etaTbo*a.tComp);
  const tComp=a.tComp+b.tComp, tComm=a.tComm+b.tComm, exposedOverlap=a.exposedOverlap+b.exposedOverlap, exposedStage=a.exposedStage+b.exposedStage, nKern=a.nKern+b.nKern;
  const graph=full.graph, gapT = graph && !c.pdl ? nKern*g.gapUs*1e-6 : 0;
  const gpu = tComp + tComm + exposedOverlap + exposedTbo + exposedStage + gapT;
  const cpuLaunch = graph ? g.graphLaunchUs*1e-6 : nKern*g.launchUs*1e-6;
  const dpSync = (c.engine==="vllm"||c.engine==="titan") && a.dpA>1 ? g.dpSyncUs*1e-6 : 0;
  const stepT = Math.max(gpu + (graph?g.graphLaunchUs*1e-6:0), cpuLaunch, g.schedUs*1e-6) + dpSync;
  const warn=dedupWarn(full.warn.concat(a.warn, b.warn));
  return Object.assign({}, a, { ops, comm, tComp, tComm, exposedOverlap, exposedTbo, exposedStage, gpu, gapT, stepT, bound:null, graph, nKern, cpuLaunch, dpSync, warn,
    rowsG:full.rowsG, rowsR:full.rowsR, Breq:full.Breq, Bp:full.Bp, Tg:full.rowsG*a.dpA,
    mbs:{ rows:[a.rowsG, b.rowsG], comp:[a.tComp, b.tComp], a2a:[aA, aB] } });
}

/* 投机解码的 draft 一项，按方法分开算。MTP：目标模型一层的平均代价（第一步按验证宽度，之后每步 1 行）+ 每步一次 lm_head。
   另外三种按各自的结构：EAGLE-3 每步一层 dense + 裁剪词表的 lm_head；DFlash 一个约 5 层的小模型对 B·(k+1) 行做一次 forward，
   上下文 KV 由目标模型的特征投影出来；DSpark 同 DFlash（B·k 行），外加 k 次 [B, r] × [r, V] 的 Markov 偏置。draft 层按 dense（≈ 12·D² 参数）、8 个 KV 头 × 128 算。 */
const SPEC = { mtp:{ name:"MTP" }, eagle3:{ name:"EAGLE-3", layers:1, vocab:32000 }, dflash:{ name:"DFlash", layers:5 }, dspark:{ name:"DSpark", layers:5, rank:256 } };
const DRAFT_KVW = 1024;
function draftCost(m, hw, c, g, r, base1){
  const k=r.specK, B=r.Breq, D=m.D, tp=r.tpA, S=SPEC[c.specMethod]?c.specMethod:"mtp";
  if(S==="mtp"){ const perLayerW=(r.gpu-opT(r,"lm")-opT(r,"samp"))/m.L, perLayer1=(base1.gpu-opT(base1,"lm")-opT(base1,"samp"))/m.L;
    return { t: perLayerW + (k-1)*perLayer1 + k*opT(base1,"lm") }; }
  const pk=peakOf(hw,"bf16"), R=(fl,by,kern)=>roof(fl,by,pk,hw,g,kern||1).t, n=c.n, Wl=12*D*D, kvw=DRAFT_KVW;
  const arc=base1.comm.filter(x=>x.key==="ar"), ar1=tp>1&&arc.length ? arc.reduce((s,x)=>s+x.t,0)/arc.reduce((s,x)=>s+x.layers,0) : 0;
  const layer=rows=>R(2*rows*Wl/tp, Wl*2/tp + rows*D*2*6, 6) + R(4*rows*n*D/tp, B*n*2*kvw*2/tp) + 2*ar1;      // 一层 dense + 读这一层的上下文 KV
  const lm=(rows,V)=>R(2*rows*D*V/tp, D*V*2/tp + rows*V*4/tp);
  if(S==="dspark" && m.drafter){
    // 模型自带的 DSpark（ownDraftW 的注释）：B·k 行一次 forward；MoE 按目标的切法摊在 N 张卡上，expert 按期望覆盖读
    const dr=m.drafter, f=m.full, de=dr.moe, wbD=dtB(c.dDt), wbE=dtB(c.wDt), kv=csaKv(m, c.kvDt), N=r.N, rows=B*k, Tg=rows*r.dpA;
    const pkD=peakOf(hw,c.dAct||c.dDt), pkE=peakOf(hw,eActOf(c,hw)), Rk=(fl,by,p,kern)=>roof(fl,by,p,hw,g,kern||1).t;
    const Wq=D*(f.qLora+f.dh)+f.qLora*f.H*f.dh, Wob=f.oG*f.oLora*D, Woa=f.H*f.dh*f.oLora, keys=Math.min(f.W, n+k);
    const attn = Rk(2*rows*(Wq+Wob)/tp, (Wq+Wob)*wbD/tp + rows*D*2*6, pkD, 4) + Rk(2*rows*Woa/tp, Woa*dtB(f.oDt)/tp + rows*(f.H*f.dh+f.oG*f.oLora)*2/tp, peakOf(hw,f.oDt), 1)
      + Rk(4*rows*f.H/tp*f.dh*keys, B*keys*kv.swa + rows*f.H/tp*f.dh*2*2, pk, 2);
    const moe = Rk(2*Tg*de.k*3*D*de.F/N, de.E*cover(de.k,de.E,Tg)*3*D*de.F*wbE/N + Tg*de.k/N*D*2*2, pkE, 3)
      + Rk(2*rows*3*D*de.Fs/tp, 3*D*de.Fs*wbD/tp + rows*D*2*2, pkD, 2) + Rk(2*rows*D*de.E, D*de.E*2 + rows*(D*2+de.E*4), pk, 2);
    const proj = Rk(2*B*(k+1)*dr.feats*D*D, dr.feats*D*D*wbD + B*(k+1)*(dr.feats+1)*D*2, pkD, 1);
    const t = proj + dr.layers*(attn + moe + 2*ar1) + lm(B*k, m.V) + k*R(2*B*dr.rank*m.V, dr.rank*m.V*2 + B*m.V*4);
    return { t, rows };
  }
  if(S==="eagle3"){ const w=k+1, Vd=SPEC.eagle3.vocab;
    const t=R(2*B*w*3*D*D/tp, 3*D*D*2/tp) + layer(B*w) + (k-1)*layer(B) + k*lm(B,Vd);
    return { t, rows:B*w }; }
  const L=SPEC[S].layers, rows=B*(S==="dspark"?k:k+1), ctxRows=B*(k+1);
  const ctx=R(2*ctxRows*5*D*D/tp, 5*D*D*2/tp) + R(2*ctxRows*D*L*2*kvw/tp, D*L*2*kvw*2/tp);   // 目标特征 → 每层上下文 K / V
  let t=ctx + L*layer(rows) + lm(B*k,m.V);
  if(S==="dspark"){ const rk=SPEC.dspark.rank; t+=k*R(2*B*rk*m.V, rk*m.V*2 + B*m.V*4); }
  return { t, rows };
}

/* DeepSeek-V4.1 自带的 DSpark（m.drafter，HF config 的 dspark_* 与 inference/model.py 的 DSparkBlock）每卡的权重：读目标 feats 层的特征，
   main_proj（feats·D → D）一次投影；layers 个块，每块是和目标同形状的滑窗 attention（DSparkAttention 继承 Attention）+ E 选 k 的 MoE
   （一个 shared expert，routed expert 和目标一样是 c.wDt、按 N 张卡切）；embedding / lm_head 和目标共用；Markov 偏置 2·rank·V（bf16） */
function ownDraftW(m, c, tpA, N){ const dr=m.drafter, f=m.full, de=dr.moe, D=m.D, wbD=dtB(c.dDt);
  const Wat = D*(f.qLora+f.dh) + f.qLora*f.H*f.dh + f.oG*f.oLora*D, Woa = f.H*f.dh*f.oLora;
  return dr.layers*((Wat*wbD + Woa*dtB(f.oDt))/tpA + de.E*3*D*de.F*dtB(c.wDt)/N + 3*D*de.Fs*wbD/tpA + D*de.E*2) + dr.feats*D*D*wbD + 2*dr.rank*m.V*2;
}

/* 投机解码：verify 步 + draft + 提交 */
function decode(m, hw, topo, c, g){
  const r = step(m,hw,topo,c,g);
  r.draftT=0; r.commitT=0;
  if(r.specK>0){
    const base1 = step(m,hw,topo,Object.assign({},c,{specK:0}),g);
    r.draftT = draftCost(m,hw,c,g,r,base1).t;
    if(m.lin.n>0) r.commitT = c.engine==="tokenspeed" ? g.launchUs*1e-6*2 + 2*r.Breq*r.Sl*m.lin.n/(hw.hbm*1e9*g.etaM) : 2*r.Breq*r.Sl*m.lin.n/(hw.hbm*1e9*g.etaM);
    r.stepT += r.draftT + r.commitT;
  }
  r.tokPerStep = r.Breq*(1+(r.specK>0?c.accept:0));
  r.tpot = r.stepT/(1+(r.specK>0?c.accept:0));
  /* HBM 的光速：这一步每张卡读写的字节全按峰值带宽走一遍（不算 η、计算、通信和 launch），摊到每个提交的 token。
     batch 1 时就是“每卡那份权重 + KV ÷ HBM 带宽”，即 tok/s/user ≤ N·HBM / 模型字节 */
  r.solT = r.ops.reduce((s,o)=>s+o.bytes,0)/(hw.hbm*1e9)/(1+(r.specK>0?c.accept:0));
  r.tokPerGpu = r.tokPerStep*r.dpA/r.N/r.stepT;
  r.mem = memory(m,hw,c,r);
  return r;
}
const opT=(r,k)=>r.ops.filter(o=>o.key===k).reduce((s,o)=>s+o.t,0);
/* TBO 什么时候划算：当前 decode 配置只改每卡并发，TBO 开 / 关各算一步。当前 MoE 后端没有单独的 a2a 时换成 DeepEP、attention 全 DP 来看；
   没有 MoE 或只有一张卡时返回 null。a2aOff：关 TBO 时一步里 dispatch + combine 的时间 */
const TBO_B = Common.freeze([4,8,16,32,64,128,256,512,1024]);
function tboSweep(m, hw, topo, c, g){
  if(!(derive(m).nMoe>0) || !(c.N>1)) return null;
  const mp=c.moePar||ENGINES[c.engine].defaultMoe, ok=["standard","deepep","hybridep"].includes(mp);
  const base=Object.assign({}, c, { phase:"decode" }, ok ? {} : { moePar:"deepep", dpA:c.N });
  const a2a = r => r.comm.filter(x=>x.key==="disp"||x.key==="comb").reduce((s,x)=>s+x.t,0);
  const rows = TBO_B.map(B => { const off=decode(m,hw,topo,Object.assign({},base,{B,tbo:false}),g), on=decode(m,hw,topo,Object.assign({},base,{B,tbo:true}),g);
    return { B, off:off.stepT, on:on.stepT, offTok:off.tokPerGpu, onTok:on.tokPerGpu, a2aOff:a2a(off), exposed:on.exposedTbo, fit:off.mem.Bmax>=off.Breq, Breq:off.Breq }; });
  const win = rows.find(r => r.on < r.off*(1-1e-9));
  // 开 TBO 这一侧引擎根本起不来的原因（例如 SGLang 的 DeepSeek-V4.1），曲线照画，页面另外说明
  const invalid = decode(m,hw,topo,Object.assign({},base,{B:TBO_B[0],tbo:true}),g).warn.filter(w=>w && w.bad);
  return { rows, swapped:!ok, moePar:base.moePar, dpA:base.dpA, win:win ? win.B : null, invalid };
}

/* DCP 扫一遍：当前 decode 配置只改 dcp（取能整除 attention TP 的 2 的幂；GQA 只到 tp / Hkv），每个值给每 token 的 KV、并发上限、
   当前并发下的一步和其中 DCP 的通信，以及放满（并发上限，最多 cap）时的单卡吞吐。用不了 DCP 时 rows 为空，why 是原因
   （noFullAttn / dcpGqa / dcpCsa / dcpNoTp，和 inert、step 的检查一致） */
function dcpSweep(m, hw, topo, c, g, cap){
  const f=m.full, tpA=c.N/Math.max(1,c.dpA);
  const why = !(f.n>0) ? "noFullAttn" : f.kind==="gqa" && c.engine!=="vllm" ? "dcpGqa" : f.kind==="csa" && c.engine==="vllm" ? "dcpCsa" : !(tpA>1) ? "dcpNoTp" : null;
  if(why) return { why, tpA, rows:[] };
  const top = f.kind==="gqa" ? Math.max(1, Math.floor(tpA/f.Hkv)) : tpA, base=Object.assign({}, c, { phase:"decode", specK:0 });
  const ds=[]; for(let d=1; d<=top; d*=2) if(tpA % d===0) ds.push(d);
  const dcpT = r => r.comm.filter(x=>/^dcp/.test(x.key)).reduce((s,x)=>s+x.t,0);
  const rows = ds.map(d => { const cc=Object.assign({}, base, { dcp:d }), r=decode(m,hw,topo,cc,g), Bf=Math.max(1, Math.min(r.mem.Bmax, cap||256)), full=decode(m,hw,topo,Object.assign({}, cc, { B:Bf }),g);
    return { dcp:d, kvTok:r.mem.kvTok, perSeq:r.mem.perSeq, Bmax:r.mem.Bmax, stepT:r.stepT, tpot:r.tpot, dcpT:dcpT(r), fit:r.mem.Bmax>=r.Breq, Bfull:Bf, fullTok:full.tokPerGpu, fullTpot:full.tpot, warn:r.warn }; });
  return { why:null, tpA, top, rows, B:c.B, n:c.n, algo:c.dcpComm||ENGINES[c.engine].dcpComm||"ag_rs" };
}

/* G 张卡上可以怎么切（部署页「同样多的卡怎么切」）。这里只列配置，算的还是 decode / prefill：
   · tp：G / t 个独立的 TP t 实例（t 到 8，NVLink 域到 16 卡以上时到 16），MoE 跟着 TP 切
   · hybrid：attention TP a × DP N / a，MoE 在这 N 张卡上做 EP。SGLang 是 --tp N --attn-dp-size N/a，a2a 后端时 ep_size = tp_size
     （python/sglang/srt/arg_groups/overrides.py）；vLLM 是 TP a × DP N/a 加 --enable-expert-parallel，EP = TP × DP，
     进 MoE 前每个 TP rank 只发 1 / TP 的 token（vllm/config/parallel.py 的 sequence-parallel MoE）。step 里的 Tr = Tg / N 就是这个
   · dep：attention 全 DP（a = 1），EP N
   · dcp：TP t 实例加 DCP；hybrid 在 a ≥ 4 时也加 DCP a（SGLang Kimi K3 的 PD decode 配方就是 DCP8 × attn-DP2 + EP16）。
     DCP 能开多大和 step 里的检查一致：MLA / DSA 到 t，GQA 只有 vLLM 支持、到 t / Hkv，CSA 只有 SGLang 支持
   N 取 8、16、32、64 里不超过 G 的（G 小于 8 时取 G）；没有 MoE 层时只有 tp 和 dcp。moePar 写的是 tp 或 deepep，引擎不支持时由界面换成兼容的后端 */
function layouts(m, c, topo, G){
  const moe=derive(m).nMoe>0, f=m.full, L=[], tpMax=topo.domain>=16 ? 16 : 8;
  const dcpMax = t => !(f.n>0) || (f.kind==="csa" && c.engine!=="sglang") || (f.kind==="gqa" && c.engine!=="vllm") ? 1 : f.kind==="gqa" ? Math.floor(t/f.Hkv) : t;
  const pow2 = x => { let d=1; while(d*2<=x) d*=2; return d; };
  const add = (fam, N, a, moePar, dcp) => L.push({ fam, id:fam+":"+N+":"+a+(dcp>1 ? ":"+dcp : ""), N, a, dcp, inst:Math.floor(G/N), cfg:{ N, dpA:N/a, moePar, dcp } });
  for(const t of [1,2,4,8,16]) if(t<=G && t<=tpMax) add("tp", t, t, "tp", 1);
  for(const t of [4,8,16]) if(t<=G && t<=tpMax && dcpMax(t)>=2) add("dcp", t, t, "tp", pow2(dcpMax(t)));
  if(moe){ const Ns=[8,16,32,64].filter(n=>n<=G); if(!Ns.length && G>1) Ns.push(G);
    for(const N of Ns){ add("dep", N, 1, "deepep", 1);
      for(const a of [2,4,8]) if(a<N && N%a===0){ add("hybrid", N, a, "deepep", 1); if(a>=4 && dcpMax(a)>=2) add("hybridDcp", N, a, "deepep", pow2(dcpMax(a))); } } }
  return L;
}
/* 每卡并发 b 下的一步 decode，拆成 attention、MoE、通信三块：一个 attention rank 跨 tpA 张卡，所以 rank 的并发是 b × tpA。
   rowsPer 是每个被碰到的 expert 的 GEMM 行数，I 是 MoE GEMM 的强度，ridge 是这张卡在 expert 精度下的屋脊点 */
function layoutPoint(m, hw, topo, c, g, b){
  const tpA=c.N/Math.max(1,c.dpA), cc=Object.assign({}, c, { phase:"decode", B:Math.max(1, Math.round(b*tpA)) }), r=decode(m,hw,topo,cc,g);
  const sum = (...gs) => r.ops.filter(o=>gs.includes(o.group)).reduce((s,o)=>s+o.t,0), moe=r.ops.find(o=>o.key==="moe");
  return { b, B:cc.B, tpA, fit:r.mem.Bmax>=r.Breq, cap:r.mem.Bmax/tpA, wGpu:r.mem.weights, kvTok:r.mem.kvTok,
    tAttn:sum("attn","lin"), tMoe:sum("moe"), tOther:sum("misc","lm"), tComm:r.tComm+r.exposedOverlap+r.exposedTbo, stepT:r.stepT, tpot:r.tpot, tok:r.tokPerGpu,
    rowsPer:moe ? moe.rowsPer : 0, I:moe ? moe.I : 0, ridge:moe ? peakOf(hw, eActOf(cc,hw))/(hw.hbm*1e9) : 0, warn:r.warn };
}

/* prefill：按 chunk 迭代一个长 prompt，得到 TTFT；单 chunk 的吞吐 */
function prefill(m, hw, topo, c, g){
  const C=c.C, P=c.P||C, chunks=[], n=Math.ceil(P/C), stride=Math.max(1,Math.ceil((n-1)/255)); let t=0;
  const one=(p,cc,req,w)=>{ const r=step(m,hw,topo,Object.assign({},c,{phase:"prefill",C:cc,p,reqPerChunk:req}),g); chunks.push({p,C:cc,t:r.stepT}); t+=r.stepT*w; };
  // 只有收尾的一块跑 lm_head。块数很多时每 stride 块取中间一块代表（单块时间随前缀平滑变化），总时间仍覆盖整个 prompt
  for(let i=0;i<n-1;i+=stride){ const w=Math.min(stride,n-1-i); one((i+Math.floor((w-1)/2))*C,C,0,w); }
  one((n-1)*C,P-(n-1)*C,c.reqPerChunk??1,1);
  const r = step(m,hw,topo,Object.assign({},c,{phase:"prefill",p:c.p||0}),g);
  r.ttft=t; r.chunks=chunks; r.nChunks=n; r.tokPerGpu = c.C*r.dpA/r.N/r.stepT; r.mem=memory(m,hw,Object.assign({},c,{phase:"decode",B:1}),r);
  return r;
}

/* chunked prefill 和 decode 怎么排。每个 DP rank 上 B 条请求在 decode（上下文 n），进来一条 P 个 token 的 prompt，块大小 / token 预算是 C：
   · 分开跑（SGLang 默认）：调度器有 prefill 就先跑 prefill（managers/scheduler.py 的 "Run prefill first if possible"），一块接一块
     跑完整条 prompt，这期间 decode 不动：卡顿 = 整条 prompt 的 prefill。--prefill-decode-interval k（c.pdInterval）让每块 prefill
     之后先跑 k 步 decode：卡顿缩到一块，TTFT 多出 (块数 − 1)·k 步 decode
   · 混合（vLLM V1 和 TensorRT-LLM 先排 decode、剩下的预算给 prefill；SGLang --enable-mixed-chunk 从块里扣掉 decode 的行）：
     每步 B 行 decode 加 C − B 个 prefill token 一起跑（mixStep）：卡顿 = 一个混合步，TTFT = ⌈P / (C − B)⌉ 个混合步
   混合步的 prefill 块按 prompt 中点那块的前缀算（attention 随前缀线性增长，平均就是中点）；分开跑的块按 prefill 逐块累加。
   稳态（闭环压测，每条请求 P 个输入、c.osl 个输出）：每步完成 B / osl 条请求、换进同样多条新请求，即每步要 demand = B·P/osl 个 prefill token。
   分开跑平均每步多 demand / P 条 prompt 的 prefill；混合时带块的步占 f = demand / (C − B)，f > 1 时预算跟不上新请求 */
/* prefill 和 decode 是否混在一步里：vLLM / TitanRL 总是混；SGLang 看 --enable-mixed-chunk，且没被 inert 关掉 */
const mixedOf = (m, c, g) => !!(ENGINES[c.engine].mixed || (c.engine==="sglang" && c.mixChunk && !inert(m,c,g).mixChunk));
const CHUNKS = Common.freeze([512, 1024, 2048, 4096, 8192, 16384]);
function chunked(m, hw, topo, c, g, Cs){
  const mixed=mixedOf(m,c,g), k=mixed || c.engine!=="sglang" ? 0 : Math.max(0, Math.round(c.pdInterval||0));   // 两个开关只有 SGLang 有（见 inert）
  const base=Object.assign({}, c, { phase:"decode", specK:0, tbo:false });
  const dec=decode(m,hw,topo,base,g), B=dec.Breq, Pl=Math.max(1, c.P||1), demand=B*Pl/Math.max(1,c.osl||1);
  const list=[...new Set((Cs||CHUNKS).concat(c.C>0 ? [c.C] : []))].sort((a,b)=>a-b);
  const rows=list.map(C => {
    const pf=prefill(m,hw,topo,Object.assign({}, c, { phase:"prefill", C:Math.min(C,Pl), P:Pl, p:0, reqPerChunk:1, specK:0, tbo:false }), g), nS=pf.nChunks;
    const sep={ stall: k>0 ? pf.ttft/nS : pf.ttft, ttft: pf.ttft + (k>0 ? (nS-1)*k*dec.stepT : 0), tpot: dec.stepT + demand/Pl*pf.ttft, f: demand/Pl*nS };
    const Cp=C-B; let mix=null;
    if(Cp>=1){ const cc=Math.min(Cp,Pl), mx=mixStep(m,hw,topo,base,g,cc,Math.max(0,Math.floor((Pl-cc)/2))), f=demand/Cp;
      // 整块按中点前缀的混合步算；最后剩下不满一块的 rem 个 token 单独算一步（只带 rem 行 prefill，接在整条 prompt 末尾）
      const nFull=Math.floor(Pl/cc), rem=Pl-nFull*cc, tRem=rem>0 ? mixStep(m,hw,topo,base,g,rem,Pl-rem).stepT : 0;
      mix={ stall:mx.stepT, ttft:nFull*mx.stepT + tRem, tpot:dec.stepT + Math.min(1,f)*(mx.stepT-dec.stepT), f, over:f>1, rows:mx.mix.rows }; }
    return { C, Cp, nChunks:nS, tPre:pf.ttft/nS, ttftAlone:pf.ttft, sep, mix };
  });
  const tok = tp => B*dec.dpA/dec.N/tp;   // 输出 tok/s/GPU
  for(const r of rows){ r.sep.tokPerGpu=tok(r.sep.tpot); if(r.mix) r.mix.tokPerGpu=tok(r.mix.tpot); }
  return { mixed, k, B, n:c.n, P:Pl, demand, tDec:dec.stepT, dpA:dec.dpA, N:dec.N, rows, warn:dec.warn };
}

/* 合部署时 decode 每步顺带的 prefill（闭环：每步完成 out / osl 条请求，换进同样多条 isl 的新请求，摊到 DP rank 上）。
   c 是这一步 decode 的配置（B 是每个 DP rank 的并发），dec 是它的 decode 结果，pre 是一条 isl 的 prefill（块大小 c.C）。
   · 分开跑（SGLang 默认，可加 --prefill-decode-interval）：每步多出 新请求数 × 整条 prompt 的 prefill 时间；decode 最长等完整条 prompt，有间隔时等一块
   · 混合（vLLM V1、SGLang --enable-mixed-chunk）：prefill token 跟着 decode 一起跑，按混合步里每个 prefill token 的边际时间算
     （mixStep：C − B 个 prefill 行的混合步减去其中纯 decode 的部分）；decode 最长等一个满预算的混合步 */
function coloc(m, hw, topo, c, g, dec, pre, isl, osl){
  const mixed=mixedOf(m,c,g), k=mixed || c.engine!=="sglang" ? 0 : Math.max(0, Math.round(c.pdInterval||0));
  const req=dec.tokPerStep/Math.max(1,osl), demand=req*isl;   // 每个 DP rank 每步换进的请求数和 prefill token 数
  if(!mixed) return { mixed, k, demand, preT:req*pre.ttft, stall: k>0 ? pre.ttft/Math.max(1,pre.nChunks) : pre.ttft };
  const C=Math.max(1, c.C||BASE.C), Cp=Math.max(1, Math.min(C-dec.Breq, isl));
  const mx=mixStep(m,hw,topo,c,g,Cp,Math.max(0,Math.floor((isl-Cp)/2))), per=Math.max(0, mx.stepT-mx.mix.dec)/Cp;
  return { mixed, k, demand, Cp, preT:demand*per, stall:mx.stepT, f:demand/Cp };
}

/* PD 分离：一条 isl 的请求的 KV 从 prefill 实例搬到 decode 实例（c 是 decode 实例的配置，pdN 是 prefill 实例的卡数）。
   两个实例都在一个 NVLink 域里走 NVLink，否则走网卡；每个 attention rank 搬自己那份、各用自己的链路，和 Fabric.transfers 同一口径 */
function pdXfer(m, hw, topo, c, g, isl, pdN){
  const kv = decode(m,hw,topo,Object.assign({},c,{phase:"decode",B:1,n:isl,specK:0}),g).mem.perSeq, nvl = pdN + c.N <= topo.domain;
  return { bytes:kv, nvl, t:kv/((nvl?hw.nvl:hw.ib)*1e9*g.etaN) + (nvl?g.alphaNv:g.alphaIb)*1e-6 };
}

/* 闭环压测（SGLang bench_serving 那种：并发 conc 固定，每条 isl 输入、osl 输出）的稳态估计。
   decode：conc 条请求一起 decode，上下文取 isl + osl/2。每条请求从开始到结束的这段时间里，其余 conc − 1 个槽位各换进一条新请求，
   它们的 prefill 插在 decode 之间，所以平均 TPOT = decode 的 TPOT + (conc − 1) × 单条 prefill 时间 / (osl − 1)
   （TPOT 按 bench_serving / genai-bench 的口径：(E2E − TTFT) / (osl − 1)）。
   ttft0 是空载时单条 prompt 的 prefill（不含排队、HTTP、分词这些引擎外的时间）；tokPerGpu 按 cookbook 的口径：(输入 + 输出) / 时间 / 卡数。 */
function serve(m, hw, topo, c, g){
  const isl=c.isl, osl=c.osl, B=Math.max(1,c.conc);
  const d = decode(m,hw,topo,Object.assign({},c,{phase:"decode",B,n:isl+osl/2}),g);
  const pf = prefill(m,hw,topo,Object.assign({},c,{P:isl,C:Math.min(c.C,isl),p:0,reqPerChunk:1}),g);
  const dec = Math.max(1, osl-1), interf = (B-1)*pf.ttft/dec, tpot = d.tpot + interf;
  return { tpot, tpotDecode:d.tpot, interf, ttft0:pf.ttft, stepT:d.stepT, draftT:d.draftT, tokPerStep:d.tokPerStep, N:d.N,
    tokPerGpu: B*(isl+osl)/(pf.ttft+dec*tpot)/d.N, outTokPerS: B/tpot, Breq:d.Breq, graph:d.graph, mem:d.mem, warn:d.warn.concat(pf.warn) };
}

/* 显存：权重 + KV 池 + 线性注意力状态槽 + 预留 */
function memory(m, hw, c, r){
  const d=derive(m), f=m.full, l=m.lin, e=m.moe, N=c.N, tpA=N/Math.max(1,c.dpA);
  const wbE=dtB(c.wDt), wbD=dtB(c.dDt), kvb=dtB(c.kvDt), sb=c.stateDt==="bf16"?2:4, P=ENGINES[c.engine];
  // CSA：wo_a 按 f.oDt（bf16）存，compressor 的 wkv / wgate 在 m > 1 时是 f.cmpDt（fp32），indexer 的 weights_proj 和 index K 的 wk 是 bf16
  const wAttn = f.kind==="csa" ? f.n*((d.pQKV+d.pO-d.pOa)/tpA + d.pAttnRep)*wbD + f.n*d.pOa/tpA*dtB(f.oDt) + d.nIdx*(f.qLora*f.idxH*f.idxD*wbD + m.D*f.idxH*2)
      + d.csa.filter(x=>x.kv).reduce((s,x)=>s + m.D*f.dh*(x.m>1?2:1)*dtB(x.m>1?f.cmpDt:"bf16") + f.dh*f.idxD*2, 0)
    : f.n*(((d.pQKV+d.pO)/tpA) + d.pAttnRep + d.pIdx)*wbD;
  // Engram：wkv 各卡一份；表放在显存时按行切在 N 张卡上（fp8，每 32 个一个 scale），放在主机内存时不占显存
  const wEng = d.pEngW*wbD + (c.engramAt==="host" ? 0 : d.pEngTab*dtB(c.engramDt||"mxfp8")/N);
  const wLin = l.n*d.pLin/tpA*wbD;
  const wExp = e.E ? d.nMoe*d.pRouted*(e.E+(r.moePar==="tp"||N<=1?0:c.redundant||0))/e.E/N*wbE : 0;
  const wMoeRep = d.nMoe*((d.pShared)/tpA + d.pRouter + d.pLat)*wbD;
  const wDense = m.dense.n*d.pDense/tpA*wbD;
  const wEmb = 2*m.V*m.D/tpA*wbD;
  const sm = c.specK>0 && SPEC[c.specMethod] && c.specMethod!=="mtp" ? SPEC[c.specMethod] : null;   // 外挂 drafter：自己的权重和上下文 KV
  const own = sm && c.specMethod==="dspark" && m.drafter ? m.drafter : null;                         // 模型自带的 DSpark（DeepSeek-V4.1）
  const wMtp = (c.specK>0 && m.mtp && !sm) ? (d.total-d.pEmb-d.pHead)/m.L/N*Math.min(wbD,wbE)*1.0 : 0;
  const wDraft = own ? ownDraftW(m, c, tpA, N) : sm ? (sm.layers*12*m.D*m.D + (sm.vocab ? 3*m.D*m.D + sm.vocab*m.D : 5*m.D*m.D + sm.layers*2*DRAFT_KVW*m.D) + (sm.rank ? 2*sm.rank*m.V*tpA : 0))*2/tpA : 0;
  const weights = wAttn+wLin+wExp+wMoeRep+wDense+wEmb+wMtp+wDraft+wEng;
  const kvTok = r.kvFull + (sm && !own ? sm.layers*2*DRAFT_KVW*2/Math.min(tpA,8) : 0);
  // 滑窗层每个请求最多存 W 个 token 的 KV；自带的 DSpark 是滑窗层，同样按 W 存
  const swaPer = r.kvSwa*Math.min(c.n, r.kvW||0) + (own ? own.layers*Math.min(c.n, f.W)*csaKv(m, c.kvDt).swa : 0);
  const slots = c.engine==="sglang" ? 1 + (c.prefixCache?1:0) + (c.specK>0?4:0) : c.engine==="tokenspeed" ? 2 : 1 + (c.specK||0);
  const statePer = l.n>0 ? l.n*(r.Sl + r.Cl)*slots : 0;
  const total = hw.mem*1e9;
  const avail = total*c.memFrac - weights - c.reserveGB*1e9;
  const perSeq = statePer + swaPer + kvTok*c.n;
  const Bmax = Math.max(0, Math.floor(avail/Math.max(perSeq,1)));
  return { weights, wAttn, wLin, wExp, wMoeRep, wDense, wEmb, wMtp, wDraft, wEng, kvTok, swaPer, statePer, slots, perSeq, avail, total, Bmax };
}

/* 默认的推理配置（界面的初始值；预设和架构页都在它上面改） */
const BASE = Common.freeze({ engine:"sglang", phase:"decode", N:1, dpA:1, B:64, n:8192, C:8192, P:32768, p:0, wDt:"bf16", eAct:"", dDt:"bf16", kvDt:"bf16", stateDt:"fp32",
  specK:0, specMethod:"mtp", accept:0, specVer:1, graph:true, pdl:false, isl:1024, osl:1024, conc:16, imb:1.15, redundant:0, dcp:1, dcpComm:"", mixChunk:false, pdInterval:0, prefixCache:true, moePar:"tp", dispFp8:true, tbo:false, hostSyncUs:20, replay:true, engramAt:"gpu", engramRepl:"none", engramLookup:"auto", engramStage:false, engramDt:"mxfp8",
  memFrac:0.85, reserveGB:6, bi:false, biFactor:2.5, implEff:1, reqPerChunk:1 });

/* 最少切几份放得下权重：每卡权重不超过显存的 60%，取 2 的幂（2–64） */
function fitShards(m, hw){ const w=derive(m).total*dtB(m.wDefault); let p=2; while(w/p>0.6*hw.mem*1e9 && p<64) p*=2; return p; }

/* 当前配置下不影响结果的选项：{ 配置项: 原因 key }。界面据此把控件灰掉并说明原因 */
function inert(m, c, g){
  const d=derive(m), P=ENGINES[c.engine], moe=d.nMoe>0, mp=c.moePar||P.defaultMoe, o={}, dec=c.phase==="decode";
  const put=(cond,why,...keys)=>{ if(cond) for(const k of keys) o[k]=o[k]||why; };
  const ep=!["tp","replicated"].includes(mp)&&mp!=="ag"&&mp!=="fused";            // 走单独 a2a 的后端
  put(!moe,"noMoe","moePar","imb","redundant","dispFp8","tbo","hostSyncUs","eAct");
  put(moe&&c.N<=1,"oneCard","moePar","imb","redundant","dispFp8","tbo","hostSyncUs");
  put(mp==="tp","moeTp","imb","redundant");
  put(!ep,"noA2a","tbo");
  put(mp!=="deepep","notDeepep","dispFp8");
  put(mp!=="standard","notStandard","hostSyncUs");
  put(m.full.n<=0,"noFullAttn","dcp","dcpComm");
  put(m.full.kind==="gqa"&&c.engine!=="vllm","dcpGqa","dcp","dcpComm");
  put(c.N/Math.max(1,c.dpA)<=1,"dcpNoTp","dcp","dcpComm");
  put(!((c.dcp||1)>1),"noDcp","dcpComm");
  put(P.mixed,"engineMixed","mixChunk");
  put(c.engine!=="sglang","mixSglang","mixChunk","pdInterval");
  put(m.full.kind==="csa","mixCsa","mixChunk");        // SGLang 的 DeepSeek-V4 系列不支持 mixed prefill / decode（arg_groups/deepseek_v4_hook.py）
  put(c.specK>0,"mixSpec","mixChunk");                  // 投机算法要自己支持 mixed chunk，否则报错或自动关掉（validation_hook.py、speculative_hook.py），这里按关掉算
  put(!!c.mixChunk&&!o.mixChunk,"mixNoInterval","pdInterval");
  put(m.full.kind!=="csa","noReplay","replay");
  put(m.full.kind==="csa"&&!P.replay,"engineNoReplay","replay");
  put(m.full.kind==="csa"&&dec,"decodeNoReplay","replay");
  put(!m.engram,"noEngram","engramAt","engramRepl","engramLookup","engramStage","engramDt");
  put(c.engine!=="sglang","engramSglang","engramRepl","engramLookup","engramStage");
  put(c.engramAt!=="host","engramGpu","engramRepl","engramStage");
  put(c.engramAt==="host"&&c.engramRepl==="per_node","engramPerNode","engramLookup");
  put(c.engramRepl!=="per_node","engramShard","engramStage");
  put(c.N<=1,"oneCard","engramLookup");
  put(dec,"engramDecode","engramStage");
  put(c.engine!=="sglang","slotsFixed","prefixCache");
  put(!(c.specK>0),"noSpec","specMethod","accept","specVer");
  put(c.specMethod!=="dspark","notDspark","specVer");
  put(!!g.lb,"ideal","imb","bi","biFactor","implEff","hostSyncUs");
  put(!c.bi,"noBi","biFactor");
  return o;
}

return { ENGINES, SPEC, BASE, TBO_B, inert, fitShards, step, mixStep, decode, prefill, chunked, CHUNKS, coloc, dcpSweep, layouts, layoutPoint, serve, pdXfer, memory, cover, eActOf, specCapWarn, tboSweep };
})();
