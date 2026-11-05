/* 推理核心：只描述 serving 引擎真正发出的 kernel 序列（paged KV、状态缓存、CUDA graph 补齐、采样、MTP 验证、DP-attention / EP 通信）。
   与训练核心不共享算子表。每条口径对应的仓库路径见 UI 的“口径与来源”。 */
const Infer = (function(){
const { dtB, peakOf, derive, qoW, capKeys, link, agT, arT, a2aT, roof, gK } = Common;

function range(a,b,s){ const r=[]; for(let x=a;x<=b;x+=s) r.push(x); return r; }
/* 引擎画像：来自三个引擎的默认值（research/inference_engines.md Part IV） */
const ENGINES = {
  sglang: { name:"SGLang", mixed:false, lmInGraph:true, samplerInGraph:false, specCap:48, defaultMoe:"tp",
    ladder:(hw,spec)=> spec ? [...range(1,8,1),...range(10,32,2),...range(40,64,4),...range(72,256,8)] : [1,2,4,8,12,...range(16,256,8),...range(272,496,16),512],
    maxGraph:(hw,spec)=> spec?48:512,
    arFusion:(m,c,T)=> (m.family==="qwen"||m.family==="kimi") && c.dpA===1 && T<=2048,
    stateDecode:()=>2, stateVerify:(m,N)=>N+3, prefillGraph:m=>m.family!=="kimi", prefillPad:0.06,
    hMat:(m,hw)=> !(m.lin.kind==="GDN" && hw.sm<100), mlaReexpand:true, sharedOverlap:"graph",
    kernels:{ gqa:Msg.of("engine_infer.kernel.sglang.gqa"), mla:"trtllm_mla", dsa:Msg.of("engine_infer.kernel.sglang.dsa"), gdn:"fused_recurrent_gated_delta_rule_packed_decode", kda:Msg.of("engine_infer.kernel.sglang.kda"), moe:Msg.of("engine_infer.kernel.sglang.moe"), ar:Msg.of("engine_infer.kernel.sglang.ar") } },
  vllm: { name:"vLLM", mixed:true, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"ag",
    ladder:(hw,spec)=> [1,2,4,...range(8,248,8),...range(256, hw.sm>=100?1024:512,16)],
    maxGraph:(hw)=> hw.sm>=100?1024:512,
    arFusion:(m,c,T,D,hw)=> m.family==="qwen" && c.dpA===1 && T*D*2 <= (hw.sm>=100?1e6:5e5),
    stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>true, prefillPad:0.06,
    hMat:m=> m.lin.kind==="GDN", mlaReexpand:true, sharedOverlap:"small",
    kernels:{ gqa:"SM100 FlashInfer trtllm-gen / SM90 FA3", mla:Msg.of("engine_infer.kernel.vllm.mla"), dsa:"FLASHINFER_MLA_SPARSE + DeepGEMM paged MQA logits", gdn:"fused_recurrent_gated_delta_rule_packed_decode + causal_conv1d_update", kda:Msg.of("engine_infer.kernel.vllm.kda"), moe:Msg.of("engine_infer.kernel.vllm.moe"), ar:Msg.of("engine_infer.kernel.vllm.ar") } },
  tokenspeed: { name:"TokenSpeed", mixed:false, lmInGraph:true, samplerInGraph:true, specCap:Infinity, defaultMoe:"replicated",
    ladder:(hw,spec)=> [1,2,4,...range(8, spec?80:160, 8)],
    maxGraph:(hw,spec)=> spec?80:160,
    arFusion:(m,c,T)=> m.family!=="glm" && c.dpA===1 && T<=2048,
    stateDecode:m=> m.lin.kind==="KDA"?3:2, stateVerify:()=>3, prefillGraph:m=>m.full.kind!=="dsa", prefillPad:0.125,
    hMat:()=>false, mlaReexpand:true, dsaAbsorbPrefill:true, sharedOverlap:"graph",
    kernels:{ gqa:"triton_gqa_prologue + trtllm-gen / fa3", mla:Msg.of("engine_infer.kernel.tokenspeed.mla"), dsa:Msg.of("engine_infer.kernel.tokenspeed.dsa"), gdn:"triton_causal_conv1d_update + flashinfer_gdn_decode_step", kda:Msg.of("engine_infer.kernel.tokenspeed.kda"), moe:Msg.of("engine_infer.kernel.tokenspeed.moe"), ar:Msg.of("engine_infer.kernel.tokenspeed.ar") } },
  /* 架构页用的参考画像：只取各引擎都默认开的优化。全注意力 FlashAttention 类 kernel（FA3 / FA4 / FlashInfer / TRT-LLM FMHA）；
     MLA decode absorbed（FlashMLA 类），prefill 走 MHA 并把前缀 latent 重新展开（SGLang MHA_CHUNKED_KV、vLLM _compute_prefill_context）；
     DSA 按 DeepSeek 开源做法：DeepGEMM fp8 MQA logits 做 indexer，FlashMLA sparse 在 latent 上做 MQA，prefill 也 absorbed；
     线性注意力 FLA：decode fused_recurrent（读写一遍状态），prefill chunk（C=64，h 每个 chunk 写回 HBM）；MoE fused grouped GEMM。
     不补齐 CUDA graph（调用方关掉 graph），没有 AR 融合，shared expert 不藏。 */
  ref: { name:Msg.of("engine_infer.engine.ref"), mixed:false, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"tp",
    ladder:()=>[1,2,4,...range(8,1024,8)], maxGraph:()=>1024,
    arFusion:()=>false, stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>false, prefillPad:0,
    hMat:()=>true, mlaReexpand:true, dsaAbsorbPrefill:true, sharedOverlap:"none",
    kernels:{ gqa:"FlashAttention", mla:"FlashMLA", dsa:"FlashMLA sparse + DeepGEMM indexer", gdn:"FLA fused_recurrent", kda:"FLA fused_recurrent (KDA)", moe:"fused MoE grouped GEMM", ar:"—" } },
  titan: { name:"TorchTitan TitanRL", mixed:true, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"standard",
    ladder:(hw)=> [1,2,4,...range(8,248,8),...range(256,512,16)], maxGraph:()=>512,
    arFusion:()=>false, stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>false, prefillPad:0,
    hMat:()=>true, mlaReexpand:false, sharedOverlap:"none",
    kernels:{ gqa:Msg.of("engine_infer.kernel.titan.gqa"), mla:Msg.of("engine_infer.kernel.titan.mla"), dsa:Msg.of("engine_infer.kernel.titan.dsa"), gdn:"attn_gym GDN", kda:"attn_gym KDA", moe:"torch._grouped_mm / Dist-MoE", ar:Msg.of("engine_infer.kernel.titan.ar") } }
};

const MOEPARS = ["tp","replicated","ag","standard","deepep","hybridep","fused"];
const padTo = (ladder, x) => { for(const v of ladder) if(v>=x) return v; return x; };
/* 每个算子查效率表用的 kernel 类别（Common.KEFF）：带权重的线性层是 gemm，attention 核心是 attn，MoE 是 grouped GEMM；
   elementwise、norm、线性注意力的扫描没有公开的单独效率，不查表 */
const KCLS = { moe:"moe", acore:"attn", aidx:"gemm" };
for(const k of ["aproj","aout","aqa","aqb","agp","akvb","auk","auv","aidxp","lproj","lout","shared","dense","lm","latd","latu","router"]) KCLS[k]="gemm";
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
  const specK = dec ? (c.specK||0) : 0, w = specK>0 ? specK+1 : 1;
  const warn=[];
  if(!Number.isInteger(tpA)) warn.push(Msg.of("engine_infer.warn.dpDivides",{dp:dpA,n:N}).invalid());
  // 批大小与 CUDA graph 补齐
  let Breq = dec ? c.B : 1;
  if(dec && specK>0 && Breq>P.specCap){ warn.push(specCapWarn(P)); Breq=P.specCap; }
  const ladder=P.ladder(hw,specK>0), maxG=P.maxGraph(hw,specK>0);
  let graph, rowsG, rowsR;
  if(dec){
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
  const fused = tpA>1 && P.arFusion(m,c,rowsG,D,hw);
  // 只有 PCIe 的机器上超过 2 张卡时，vLLM 和 SGLang 都关掉自定义 all-reduce，退回 NCCL 环（topo.pcieOnly 只有集群页会设）
  const arAlgo = topo.pcieOnly && tpA>2 ? "ring" : rowsG*D*2 <= 512*1024 ? "oneshot" : "twoshot";
  if(tpA>1){
    cm("ar", fused?Msg.of("engine_infer.comm.arFused"):Msg.of("engine_infer.comm.ar"), boundaries, arT(rowsG*D*2,tpA,lkA,g,arAlgo), rowsG*D*2*(tpA-1)/tpA);
    if(epGather) cm("epag",Msg.of("engine_infer.comm.epag"),epGather,agT(rowsG*D*2,tpA,lkA,g),rowsG*D*2*(tpA-1)/tpA);
    op("norm",Msg.of("engine_infer.op.norm"),"misc",fused?epGather:m.L*2,8*rowsG*D,3*rowsG*D*2,pkA,1);
  } else op("norm",Msg.of("engine_infer.op.norm"),"misc",boundaries,8*rowsG*D,3*rowsG*D*2,pkA,1);
  if(m.mhc) op("mhc",Msg.of("engine_infer.op.mhc"),"misc",m.L*2,4*m.mhc*rowsG*D,2*m.mhc*rowsG*D*2,pkA,2);
  // AttnRes：每层两次（attention 前、MLP 前；SGLang kimi_k3.py 的 attn_res / mlp_res_proj），读 bank 里之前各块的快照和当前流，写一行
  if(m.attnRes) op("attnres",Msg.of("engine_infer.op.attnres"),"misc",m.L*2,2*(m.attnRes/2+1)*rowsG*D,(m.attnRes/2+2)*rowsG*D*2,pkA,1);

  /* ---- 全注意力层 ---- */
  let kvTokRank=0, idxTok=0;
  if(f.n>0){
    if(f.kind==="gqa"){
      const qo=qoW(f), kvo=2*f.Hkv*f.dh, Hl=f.Hq/tpA, kvHeads=Math.max(1,f.Hkv/tpA);
      const nSwa=f.swa||0, nGlob=f.n-nSwa;
      kvTokRank = 2*kvHeads*f.dh*kvb;
      op("aproj",f.gate?Msg.of("engine_infer.op.aprojGate"):Msg.of("engine_infer.op.aproj"),"attn",f.n,2*rowsG*D*(qo+kvo)/tpA, D*(qo+kvo)/tpA*wbD + rowsG*(D+(qo+kvo)/tpA)*2, pkD,1);
      op("aprol",Msg.of("engine_infer.op.aprol"),"attn",f.n,10*rowsG*(Hl+2*kvHeads)*f.dh, rowsG*(Hl+2*kvHeads)*f.dh*2*2 + rowsR*kvTokRank, pkA, c.engine==="sglang"&&f.gate?2:1);
      // 滑窗层只读最近 W 个 token 的 KV；全局层读整段
      const core=(layers,keyCap,tag)=>{ const cx=Math.min(ctx,keyCap);
        const mod = keyCap<Infinity ? "swa" : "full";     // 架构页按层类型拆模块用
        if(dec) op("acore",Msg.of("engine_infer.op.acoreDecode",{kernel:P.kernels.gqa,layers:tag}),"attn",layers, 4*Hl*f.dh*cx*Breq*w, Breq*cx*kvTokRank + rowsR*Hl*f.dh*2*2, pkA,1,{scalesWith:"real",mod});
        else op("acore",Msg.of("engine_infer.op.acorePrefill",{layers:tag}),"attn",layers, 4*Hl*f.dh*rowsR*capKeys(ctx,rowsR,keyCap), (cx+rowsR)*kvTokRank + rowsR*Hl*f.dh*2*2, pkA,1,{mod}); };
      core(nGlob, Infinity, nSwa?Msg.of("engine_infer.op.layersGlobal"):"");
      if(nSwa) core(nSwa, f.W, Msg.of("engine_infer.op.layersSwa",{w:f.W}));
      if(f.gate) op("agate",f.gateHead?Msg.of("engine_infer.op.agateHead"):Msg.of("engine_infer.op.agate"),"attn",f.n,2*rowsG*Hl*f.dh,3*rowsG*Hl*f.dh*2,pkA,1);
      op("aout",Msg.of("engine_infer.op.aout"),"attn",f.n,2*rowsG*f.Hq*f.dh*D/tpA, f.Hq*f.dh*D/tpA*wbD + rowsG*(f.Hq*f.dh/tpA+D)*2, pkD,1);
    } else {
      const H=f.H, Hl=H/tpA, dq=f.nope+f.rope, lat=f.kvLora+f.rope, dcp=Math.max(1,c.dcp||1);
      kvTokRank = lat*kvb/dcp;                                    // latent 无头维度，TP 切不开，只能靠 DCP
      op("aqa",Msg.of("engine_infer.op.aqa"),"attn",f.n,2*rowsG*D*(f.qLora+lat), D*(f.qLora+lat)*wbD + rowsG*(D+f.qLora+lat)*2, pkD,1);
      op("anorm",Msg.of("engine_infer.op.anorm"),"attn",f.n,8*rowsG*(f.qLora+f.kvLora), rowsG*(f.qLora+f.kvLora)*2*2 + rowsR*kvTokRank*dcp/dcp, pkA,2);
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
        if(dec) op("acore",tag,"attn",f.n, 2*Hl*(lat+f.kvLora)*nKV*Breq*w, Breq*nKV*kvTokRank*(f.kind==="dsa"?1:1) + rowsR*Hl*(lat+f.kvLora)*2, pkA,2,{scalesWith:"real"});
        else op("acore",tag,"attn",f.n, 2*Hl*(lat+f.kvLora)*nKV*rowsR, (ctx+rowsR)*kvTokRank, pkA,2);
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
  // Blackwell 上 FP4 MoE（TRT-LLM-gen）的 hidden 维按 256 补零（vLLM PR 39510）：补出来的部分照样算、照样读
  const dPad = !lb && (c.wDt==="nvfp4"||c.wDt==="mxfp4") && hw.sm>=100 ? Math.ceil(De/256)*256/De : 1;
  op("moe",Msg.of("engine_infer.op.moe",{kernel:P.kernels.moe}),"moe",nMoe, 2*rowsExp*3*De*e.F*dPad, touched*wSlice*wbE*dPad + rowsExp*(De*2+3*e.F*2/(moePar==="tp"?N:1)), pkE, 3,{touched,cover:cv,dPad});
  if(e.Fs>0){ const Fl=e.Fs/tpA;
    op("shared",e.sharedGate?Msg.of("engine_infer.op.sharedGate"):Msg.of("engine_infer.op.shared"),"moe",nMoe,6*Tloc*D*Fl, 3*D*Fl*wbD + Tloc*(D+3*Fl)*2, pkD,2,
      {overlap: (P.sharedOverlap==="graph"&&graph&&dec) || (P.sharedOverlap==="small"&&Tloc<=256) ? "moe" : false}); }
  if(e.Dlat){
    // 算法和层边界的 AR 一样按消息大小选：大消息（prefill）用 one-shot 要读 N−1 份整消息，会比 two-shot 慢好几倍
    if(tpA>1){ const lb=Tloc*e.Dlat*2; cm("latar",Msg.of("engine_infer.comm.latar",{dim:e.Dlat}),nMoe,arT(lb,tpA,lkA,g,lb<=512*1024?"oneshot":"twoshot"),lb*(tpA-1)/tpA); }
    // SGLang K3 在 TP8、不超过 12 行时把复制的 latent up 换成按列切的 GEMV + multicast all-gather（k3_ar_fusion.gemm_ag_up_proj）
    const latAg = c.engine==="sglang" && m.family==="kimi" && fused && tpA===8 && hw.sm>=100 && Tloc<=12, lw = latAg ? tpA : 1;
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
  const moeT = ops.filter(o=>o.key==="moe").reduce((s,o)=>s+o.t,0);
  let tOps=0, tHidden=0;
  for(const o of ops){ if(o.overlap==="moe"){ tHidden+=o.t; } else tOps+=o.t; }
  const sharedT = ops.filter(o=>o.overlap==="moe").reduce((s,o)=>s+o.t,0);
  let tComm=0, fusedComm=0, tboComm=0;
  for(const x of comm){ if(x.overlap==="moe") fusedComm+=x.t; else if(x.overlap==="tbo") tboComm+=x.t; else tComm+=x.t; }
  const exposedOverlap = Math.max(0, sharedT + fusedComm - moeT);  // 共享 expert 与融合 a2a 都藏在 routed expert 后面
  const tComp = tOps;
  // TBO：一个 micro-batch 的通信和另一个的计算重叠；建模按能藏一半，下限按全部能藏
  const exposedTbo = c.tbo ? Math.max(0,tboComm - tComp*(lb?1:0.5)) : tboComm;
  let gpu = tComp + tComm + exposedOverlap + exposedTbo;
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
  return { ops, comm, tComp, tComm, exposedOverlap, exposedTbo, gpu, gapT, stepT, bound, graph, rowsG, rowsR, Breq, Bp, w, specK, tpA, dpA, N, Tg, kvTokRank, idxTok, Sl, Cl, nKern, cpuLaunch, fused, warn, moePar, P };
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
  if(S==="eagle3"){ const w=k+1, Vd=SPEC.eagle3.vocab;
    const t=R(2*B*w*3*D*D/tp, 3*D*D*2/tp) + layer(B*w) + (k-1)*layer(B) + k*lm(B,Vd);
    return { t, rows:B*w }; }
  const L=SPEC[S].layers, rows=B*(S==="dspark"?k:k+1), ctxRows=B*(k+1);
  const ctx=R(2*ctxRows*5*D*D/tp, 5*D*D*2/tp) + R(2*ctxRows*D*L*2*kvw/tp, D*L*2*kvw*2/tp);   // 目标特征 → 每层上下文 K / V
  let t=ctx + L*layer(rows) + lm(B*k,m.V);
  if(S==="dspark"){ const rk=SPEC.dspark.rank; t+=k*R(2*B*rk*m.V, rk*m.V*2 + B*m.V*4); }
  return { t, rows };
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
  r.tokPerGpu = r.tokPerStep*r.dpA/r.N/r.stepT;
  r.mem = memory(m,hw,c,r);
  return r;
}
const opT=(r,k)=>r.ops.filter(o=>o.key===k).reduce((s,o)=>s+o.t,0);

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
  const wAttn = f.n*(((d.pQKV+d.pO)/tpA) + d.pAttnRep + d.pIdx)*wbD;
  const wLin = l.n*d.pLin/tpA*wbD;
  const wExp = e.E ? d.nMoe*d.pRouted*(e.E+(r.moePar==="tp"||N<=1?0:c.redundant||0))/e.E/N*wbE : 0;
  const wMoeRep = d.nMoe*((d.pShared)/tpA + d.pRouter + d.pLat)*wbD;
  const wDense = m.dense.n*d.pDense/tpA*wbD;
  const wEmb = 2*m.V*m.D/tpA*wbD;
  const sm = c.specK>0 && SPEC[c.specMethod] && c.specMethod!=="mtp" ? SPEC[c.specMethod] : null;   // 外挂 drafter：自己的权重和上下文 KV
  const wMtp = (c.specK>0 && m.mtp && !sm) ? (d.total-d.pEmb-d.pHead)/m.L/N*Math.min(wbD,wbE)*1.0 : 0;
  const wDraft = sm ? (sm.layers*12*m.D*m.D + (sm.vocab ? 3*m.D*m.D + sm.vocab*m.D : 5*m.D*m.D + sm.layers*2*DRAFT_KVW*m.D) + (sm.rank ? 2*sm.rank*m.V*tpA : 0))*2/tpA : 0;
  const weights = wAttn+wLin+wExp+wMoeRep+wDense+wEmb+wMtp+wDraft;
  const nSwa = f.n>0 ? (f.swa||0) : 0;
  const kvTok = (f.n>0 ? (f.n-nSwa)*(r.kvTokRank + (r.idxTok||0)) : 0) + (sm ? sm.layers*2*DRAFT_KVW*2/Math.min(tpA,8) : 0);
  const swaPer = nSwa*r.kvTokRank*Math.min(c.n, f.W||0);          // 滑窗层每个请求最多存 W 个 token 的 KV
  const slots = c.engine==="sglang" ? 1 + (c.prefixCache?1:0) + (c.specK>0?4:0) : c.engine==="tokenspeed" ? 2 : 1 + (c.specK||0);
  const statePer = l.n>0 ? l.n*(r.Sl + r.Cl)*slots : 0;
  const total = hw.mem*1e9;
  const avail = total*c.memFrac - weights - c.reserveGB*1e9;
  const perSeq = statePer + swaPer + kvTok*c.n;
  const Bmax = Math.max(0, Math.floor(avail/Math.max(perSeq,1)));
  return { weights, wAttn, wLin, wExp, wMoeRep, wDense, wEmb, wMtp, wDraft, kvTok, swaPer, statePer, slots, perSeq, avail, total, Bmax };
}

/* 默认的推理配置（界面的初始值；预设和架构页都在它上面改） */
const BASE = Common.freeze({ engine:"sglang", phase:"decode", N:1, dpA:1, B:64, n:8192, C:8192, P:32768, p:0, wDt:"bf16", eAct:"", dDt:"bf16", kvDt:"bf16", stateDt:"fp32",
  specK:0, specMethod:"mtp", accept:0, graph:true, pdl:false, isl:1024, osl:1024, conc:16, imb:1.15, redundant:0, dcp:1, prefixCache:true, moePar:"tp", dispFp8:true, tbo:false, hostSyncUs:20,
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
  put(m.full.n<=0||m.full.kind==="gqa","noLatent","dcp");
  put(c.engine!=="sglang","slotsFixed","prefixCache");
  put(!(c.specK>0),"noSpec","specMethod","accept");
  put(!!g.lb,"ideal","imb","bi","biFactor","implEff","hostSyncUs");
  put(!c.bi,"noBi","biFactor");
  return o;
}

return { ENGINES, SPEC, BASE, inert, fitShards, step, decode, prefill, serve, memory, cover, eActOf, specCapWarn };
})();
