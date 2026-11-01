/* 训练核心：Megatron-Core 与 torchtitan 的逐层 fwd / dgrad / wgrad / 重算，以及各自的并行通信。
   与推理核心不共享算子表：这里没有 KV cache、状态缓存、采样，有的是反向、重算、整段 CE、优化器与梯度同步。 */
const Train = (function(){
const { dtB, peakOf, derive, layerPlan, qoW, swaKeys, link, agT, arT, a2aT, roof } = Common;

/* FLA chunk kernel（C=64）每个 value head、每个 token 的 FLOPs：前向 / 反向（反向重算 w,u 与整条 h 递推） */
function linFlops(kind, K, V){ const C=64;
  return kind==="KDA" ? { f:6*K*V+3.5*C*K+4*C*V+0.5*C*C, b:16*K*V+9*C*K+10*C*V+4*C*C }
                      : { f:6*K*V+7.25*C*K+4*C*V+0.5*C*C, b:16*K*V+18*C*K+10*C*V+4*C*C }; }

/* GEMM 精度配方：f 是前向，b 是 dgrad / wgrad；w2d 表示权重用方块，一份量化同时给两个方向用（Hopper 上仍要另存一份转置，见 Quant.RECIPES） */
const PREC = {
  bf16:   { f:"bf16",  b:"bf16",  name:"BF16" },
  fp8:    { f:"fp8",   b:"fp8",   w2d:true, name:Msg.of("engine_train.prec.fp8") },
  mxfp8:  { f:"mxfp8", b:"mxfp8", name:Msg.of("engine_train.prec.mxfp8") },
  mxfp8hp:{ f:"mxfp8", b:"bf16",  name:Msg.of("engine_train.prec.mxfp8hp") },
  nvfp4:  { f:"nvfp4", b:"nvfp4", w2d:true, name:Msg.of("engine_train.prec.nvfp4") },
  nvfp4hp:{ f:"nvfp4", b:"bf16",  tok:true, name:Msg.of("engine_train.prec.nvfp4hp") } };

/* 每个框架认的枚举值：传别的值不会被悄悄换成默认值，而是报无效配置 */
const ENUMS = {
  megatron:{ dispatcher:["alltoall","deepep","hybridep","ncclep","standard","distmoe"], cpType:["p2p","allgather","a2a"], linCP:["headwise","chunkwise"], recompute:["none","full","selective"] },
  torchtitan:{ dispatcher:Object.keys(Par.EP), cpType:Object.keys(Par.CP), linCP:["state","ulysses"], recompute:["none","default","full"] } };

/* 当前配置下不影响结果的选项：{ 配置项: 原因 key }。界面据此把控件灰掉并说明原因 */
function inert(m, c, g){
  const d=derive(m), mg=c.fw==="megatron", moe=d.nMoe>0, o={};
  const put=(cond,why,...keys)=>{ if(cond) for(const k of keys) o[k]=o[k]||why; };
  put(!moe,"noMoe","ep","etp","imb","dispatcher","epOverlap","hostSyncUs","mokF");
  put(moe&&c.ep<=1,"noEp","imb","dispatcher","epOverlap","sharedOverlap","hostSyncUs","mokF");
  put(!moe||!m.moe.Fs,"noShared","sharedOverlap");
  put(mg?["hybridep","distmoe"].includes(c.dispatcher):c.dispatcher!=="standard"&&c.dispatcher!=="deepep","dispNoSync","hostSyncUs");
  put(!mg&&c.dispatcher!=="mok","dispNoMok","mokF");
  put(c.cp<=1,"noCp","cpType","linCP","cpLB");
  put(m.full.n<=0,"noAttn","cpType","cpLB");
  put(m.lin.n<=0,"noLin","linCP");
  put(c.cp>1&&(mg||c.cpType==="ulysses"),"cpLbNa","cpLB");
  put(c.tp<=1,"noTp","tpOverlap");
  put(!c.offload,"noOffload","cpuAdamGBs","overlapOffload");
  put(!!g.lb,"ideal","etaA","imb","hostSyncUs");
  put(c.pp<=1,"noPp","vpp");
  const dp=c.N/(c.tp*c.cp*c.pp), edp=mg ? c.N/(Math.max(1,c.etp||1)*c.ep*c.pp) : dp*c.cp*c.tp/c.ep, R=Math.max(1,c.hsdp||1);
  put(mg&&dp*c.cp<=1&&edp<=1,"noDp","overlapGrad","overlapParam");
  put(!mg&&dp*c.cp/R<=1&&edp/R<=1,"noFsdp","gradDt","reshard");
  return o;
}

function run(m, hw, topo, c, g){
  const d=derive(m), f=m.full, l=m.lin, e=m.moe, D=m.D, N=c.N, mg=c.fw==="megatron";
  const tp=c.tp, cp=c.cp, pp=c.pp, vpp=Math.max(1,c.vpp||1), ep=c.ep, etp=mg?Math.max(1,c.etp||1):1;
  const dp = N/(tp*cp*pp);
  const edp = mg ? N/(etp*ep*pp) : dp*cp*tp/ep;
  const warn=[];
  if(!Number.isInteger(dp)) warn.push(Msg.of("engine_train.warn.dpDivides",{n:N}).invalid());
  if(mg && d.nMoe>0 && !Number.isInteger(edp)) warn.push(Msg.of("engine_train.warn.edpDivides",{n:N}).invalid());
  if(d.nMoe>0 && !mg && (ep<tp || ep%tp)) warn.push(Msg.of("engine_train.warn.ttEpTp").invalid());
  if(d.nMoe>0 && e.E%ep) warn.push(Msg.of("engine_train.warn.epExperts",{e:e.E}));
  for(const [k,opts] of Object.entries(ENUMS[c.fw]||ENUMS.megatron)) if(!opts.includes(c[k])) warn.push(Msg.of("engine_train.warn.enum",{key:k,val:String(c[k]),fw:c.fw,opts:opts.join(" / ")}).invalid());
  if(!PREC[c.prec]) warn.push(Msg.of("engine_train.warn.enum",{key:"prec",val:String(c.prec),fw:c.fw,opts:Object.keys(PREC).join(" / ")}).invalid());
  if(pp*vpp>m.L) warn.push(Msg.of("engine_train.warn.ppTooMany",{l:m.L,n:pp*vpp}).invalid());
  else if(pp>1 && m.L%(pp*vpp)) warn.push(Msg.of("engine_train.warn.ppLayers",{l:m.L,n:pp*vpp}));
  const tok=c.tok, T=tok*cp, Tc=tok, Tl=tok/tp;    // T：TP×CP 组一个 micro-batch 的 token；Tc：本 CP rank；Tl：SP 下本 rank 驻留
  const PR=PREC[c.prec]||PREC.bf16, fq=PR.f!=="bf16", bq=PR.b!=="bf16";
  if(c.S>T) warn.push(Msg.of("engine_train.warn.docGtBatch",{s:c.S,t:T}));
  const S=c.S, wbF=dtB(PR.f), wbB=dtB(PR.b), pkF=peakOf(hw,PR.f), pkB=peakOf(hw,PR.b), pkA=peakOf(hw,"bf16");
  const pkAt = c.etaA>0 && !g.lb ? pkA*c.etaA/g.etaC : pkA;   // attention 核心单独给效率（0 = 和 GEMM 一样用 η_c；理想下限不打折）
  if(fq && PR.f!=="fp8" && hw.sm<100) warn.push(Msg.of("engine_train.warn.needBlackwell",{prec:PR.name}));
  // TE 不支持优化器 CPU offload 和 --fp8-param-gather 一起开（slime 的 FP8 笔记）：端到端 FP8 要么关 offload，要么参数按 BF16 gather
  if(mg && fq && c.offload) warn.push(Msg.of("engine_train.warn.fp8Offload",{prec:PR.name}));
  // 低精度时每个 GEMM 输入多一次量化：读 bf16，写行方向一份（反向也低精度时再写列方向一份，wgrad 用）
  const qF = fq ? 2 + wbF + (bq?wbB:0) : 0, qB = bq ? 2 + 2*wbB : 0;
  const ops=[], comm=[];
  let tokB=0;                                       // NVFP4 按 token FP32 scale 没融合时，每个 GEMM 前后各多一次逐元素缩放
  // 每个算子三遍：fwd、dgrad（反向对输入）、wgrad（反向对权重）；非 GEMM 的反向全记在 dgrad
  const op=(key,name,group,layers,fwd,dg,wg,extra)=>{ if(layers<=0) return; if(fwd&&fwd.tb) tokB+=fwd.tb*layers;
    const r=(x,pk)=>x? roof(x.fl, x.by, pk||x.pk, hw, g, x.k||1) : {t:0,tc:0,tm:0,bound:"—",I:0};
    const rf=r(fwd), rd=r(dg), rw=r(wg);
    ops.push(Object.assign({key,name,group,layers,
      fwd:{fl:(fwd?fwd.fl:0)*layers, by:(fwd?fwd.by:0)*layers, t:rf.t*layers, bound:rf.bound, I:rf.I},
      dgrad:{fl:(dg?dg.fl:0)*layers, by:(dg?dg.by:0)*layers, t:rd.t*layers, bound:rd.bound, I:rd.I},
      wgrad:{fl:(wg?wg.fl:0)*layers, by:(wg?wg.by:0)*layers, t:rw.t*layers, bound:rw.bound, I:rw.I}},extra||{})); };
  // GEMM：M 行、K→N，权重按 shard 切；三遍 FLOPs 相同，字节不同。wMul：grouped GEMM 一次读完的权重份数（本 rank 的 expert 数）
  const gemm=(M,K,Nn,wShard,wMul=1)=>{ const W=K*Nn/wShard*wMul;
    return [ {fl:2*M*K*Nn/wShard, by:W*wbF + M*(K+Nn/wShard)*2 + qF*M*K, pk:pkF, tb: PR.tok&&!c.tokFused ? 4*M*K + 4*M*Nn/wShard : 0},
             {fl:2*M*K*Nn/wShard, by:W*wbB + M*(K+Nn/wShard)*2 + qB*M*Nn/wShard, pk:pkB},
             {fl:2*M*K*Nn/wShard, by:W*4*2 + M*(K+Nn/wShard)*2, pk:pkB} ]; };      // wgrad 累加进 fp32 main_grad
  const elem=(flPerTok,byPerTok,M,bwdMult)=>[ {fl:flPerTok*M,by:byPerTok*M,pk:pkA}, {fl:flPerTok*M*(bwdMult||2),by:byPerTok*M*(bwdMult||2),pk:pkA}, null ];
  const cm=(key,name,cls,layers,passes,t1,bytes1,extra)=>{ if(layers<=0||t1<=0) return; comm.push(Object.assign({key,name,cls,layers,passes,t:t1*layers,bytes:bytes1*layers},extra||{})); };
  // 逐次调用的版本：xs 是一层里每一次集合通信（Par.xc），总时间和总字节由它们加出来；blk 标明属于哪种层（full / swa / lin / moe / dense / head / stage）
  const xc=Par.xc, cmX=(key,name,cls,layers,blk,xs,extra)=>cm(key,name,cls,layers,1,xs.reduce((s,x)=>s+x.t,0),xs.reduce((s,x)=>s+x.bytes,0),Object.assign({blk,xs},extra||{}));

  const lkTP=link(hw,topo,tp,1), lkCP=link(hw,topo,cp,tp), lkEP=link(hw,topo,ep, mg?etp:1);
  const lkTPx=Par.lk(hw,topo,tp,1), lkCPx=Par.lk(hw,topo,cp,tp);   // 带 kind 的同一条链路：和 FSDP 抢带宽时按它记账
  const lkPP=link(hw,topo,2,tp*cp*dp), lkDP=link(hw,topo,dp*cp,tp);
  const lkEDP=link(hw,topo,edp, mg?etp*ep:ep);
  // 重算策略 → 每个算子是否在反向里再跑一次 fwd
  const rcFull = c.recompute==="full";
  const sel = new Set(c.rcModules||[]);
  const ttDef = !mg && c.recompute==="default";
  const rc = key => rcFull ? true : mg ? (
      (sel.has("core_attn") && key==="acore") || (sel.has("layernorm") && key==="norm") || (sel.has("moe_act") && key==="act") ||
      (sel.has("mla_up_proj") && (key==="aqb"||key==="akvb")) || (sel.has("moe") && ["router","permute","moe1","act","moe2","unperm","latd","latu"].includes(key)) ||
      (sel.has("shared_experts") && key==="shared") || (sel.has("gdn") && ["lproj","lconv","lcore","lnorm","lout"].includes(key)) )
    : ttDef ? (key==="moe1"||key==="act") : false;
  const rcA2A = mg ? (rcFull || sel.has("moe")) : (rcFull && c.dispatcher!=="distmoe");
  const linHead = c.linCP==="headwise" || c.linCP==="ulysses";   // 线性注意力的 CP：按头 a2a，否则按段传状态（FLA cp_context）
  const tpBlocks = []; // 走 TP+SP 的块：每块 fwd AG+RS，bwd 2 AG + 1 RS

  /* ---- 全注意力 ---- */
  let coreFullOp=null, coreSwaOp=null;
  if(f.n>0){
    op("norm",Msg.of("engine_train.op.normAttn"),"misc",f.n,...elem(8*D/tp,3*D*2/tp,Tc,2));
    if(f.kind==="gqa"){
      const qo=qoW(f), kvo=2*f.Hkv*f.dh, nSwa=f.swa||0;
      op("aqkv",f.gate?Msg.of("engine_train.op.qkvGate"):Msg.of("engine_train.op.qkv"),"attn",f.n,...gemm(Tc,D,qo+kvo,tp));
      op("arope",Msg.of("engine_train.op.arope"),"attn",f.n,...elem(10*(f.Hq+2*f.Hkv)*f.dh/tp,(f.Hq+2*f.Hkv)*f.dh*2*2/tp,Tc,2));
      // 每 token 平均看 S/2 个 key（滑窗层看 swaKeys 个）：QKᵀ + PV
      const core=(layers,keys,tag,mod)=>{ const fl=4*(f.Hq/tp)*f.dh*keys*Tc;
        op("acore",mg?Msg.of("engine_train.op.acoreMg",{tag}):Msg.of("engine_train.op.acoreTt",{tag}),"attn",layers,{fl,by:Tc*(2*f.Hq+2*f.Hkv)*f.dh*2/tp,pk:pkAt},{fl:2.5*fl,by:Tc*(4*f.Hq+4*f.Hkv)*f.dh*2/tp,pk:pkAt},null,{mod}); return ops[ops.length-1]; };
      coreFullOp = core(f.n-nSwa, S/2, nSwa?Msg.of("engine_train.op.tagGlobal"):"", "full");
      if(nSwa) coreSwaOp = core(nSwa, swaKeys(S,f.W), Msg.of("engine_train.op.tagSwa",{w:f.W}), "swa");
      if(f.gate) op("agate",f.gateHead?Msg.of("engine_train.op.gateHead"):Msg.of("engine_train.op.gate"),"attn",f.n,...elem(2*f.Hq*f.dh/tp,3*f.Hq*f.dh*2/tp,Tc,2));
      op("aout",Msg.of("engine_train.op.aout"),"attn",f.n,...gemm(Tc,f.Hq*f.dh,D,tp));
    } else {
      const H=f.H, dq=f.nope+f.rope, lat=f.kvLora+f.rope;
      op("aqa",mg?Msg.of("engine_train.op.aqaMg"):Msg.of("engine_train.op.aqaTt"),"attn",f.n,...gemm(mg?Tl:Tc,D,f.qLora+lat,1));
      op("aqb",Msg.of("engine_train.op.aqb"),"attn",f.n,...gemm(Tc,f.qLora,H*dq,tp));
      if(f.kind==="dsa"){
        // Megatron DSA = absorbed MLA：MQA 一个 latent 头；indexer 在每个 TP rank 上对整个 TP 组序列重复算
        op("auk",Msg.of("engine_train.op.auk"),"attn",f.n,...gemm(Tc,f.nope,f.kvLora,1/(H/tp)));
        const kp=f.kpool||1, nk=Math.min(S/2,f.topk);
        op("aidxp",Msg.of("engine_train.op.aidxp"),"attn",f.n,...gemm(Tc,D,(f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH)/D,1));
        const fi=2*f.idxH*f.idxD*(S/2)/kp*Tc;
        op("aidx",Msg.of("engine_train.op.aidx"),"attn",f.n,{fl:fi*2,by:Tc*(S/2)/kp*4*2,pk:pkA},{fl:fi*2,by:Tc*(S/2)/kp*4*2,pk:pkA},null);
        const fl=2*(H/tp)*(lat+f.kvLora)*nk*Tc;
        op("acore",Msg.of("engine_train.op.acoreDsa",{k:f.topk}),"attn",f.n,{fl,by:Tc*f.topk*4 + Tc*(H/tp)*(lat+f.kvLora)*2,pk:pkAt},{fl:2.5*fl,by:2*Tc*(H/tp)*(lat+f.kvLora)*2 + Tc*f.topk*4,pk:pkAt},null); coreFullOp=ops[ops.length-1];
        op("auv",Msg.of("engine_train.op.auv"),"attn",f.n,...gemm(Tc,f.kvLora,f.vh,1/(H/tp)));
      } else {
        op("akvb",Msg.of("engine_train.op.akvb"),"attn",f.n,...gemm(Tc,f.kvLora,H*(f.nope+f.vh),tp));
        const fl=2*(H/tp)*((dq+f.vh)/2)*S*Tc;
        op("acore",Msg.of("engine_train.op.acoreMla",{qk:dq,v:f.vh}),"attn",f.n,{fl,by:Tc*H*(2*dq+2*f.vh)*2/tp,pk:pkAt},{fl:2.5*fl,by:Tc*H*(4*dq+4*f.vh)*2/tp,pk:pkAt},null); coreFullOp=ops[ops.length-1];
      }
      if(f.gate){ op("agp",Msg.of("engine_train.op.agp"),"attn",f.n,...gemm(Tc,D,H*f.vh,tp)); op("agate",Msg.of("engine_train.op.gate"),"attn",f.n,...elem(2*H*f.vh/tp,3*H*f.vh*2/tp,Tc,2)); }
      op("aout",Msg.of("engine_train.op.aout"),"attn",f.n,...gemm(Tc,H*f.vh,D,tp));
    }
    tpBlocks.push({name:Msg.of("engine_train.blk.full"),blk:"attn",layers:f.n,inW: f.kind==="gqa"?D:(f.qLora+f.kvLora)});
  }

  /* ---- 线性注意力 ---- */
  if(l.n>0){
    const Hv=l.Hv/tp, lf=linFlops(l.kind,l.dk,l.dv), byTok=Hv*(l.kind==="KDA"?8200:5500)*(l.dk+l.dv)/256;
    op("norm",Msg.of("engine_train.op.normAttn"),"misc",l.n,...elem(8*D/tp,3*D*2/tp,Tc,2));
    op("lproj",Msg.of("engine_train.op.lproj"),"lin",l.n,...gemm(Tc,D,d.linIn+(d.linFa+d.linFb)/D,tp));   // KDA 的低秩 f 门按等量 FLOPs 并进宽度
    op("lconv",Msg.of("engine_train.op.lconv"),"lin",l.n,...elem(2*l.conv*d.linConvDim/tp+20*Hv*l.dk,(d.linConvDim+2*Hv*l.dk)*2*2/tp,Tc,3));
    op("lcore",Msg.of("engine_train.op.lcore",{kind:l.kind,impl:mg?"FLA":"attn_gym"}),"lin",l.n,{fl:Hv*lf.f*Tc,by:byTok*Tc,pk:pkA,k:5},{fl:Hv*lf.b*Tc,by:2.5*byTok*Tc,pk:pkA,k:8},null);
    if(cp>1 && !linHead){
      // FLA CP：每段除 S_ext 外还要算转移矩阵 M（chunk_cp_fwd_kernel_h）：每个 C 块一次 KᵀW（2K²C）+ 一次 M 链乘（2K³，TF32x3 约 1/3 峰值），再按前面的 rank 合并 S←M·S+S_ext；反向对每个乘积各算两边
      const K=l.dk, C=64, seqs=S<Tc?2:1, flKW=Hv*2*K*K*Tc, flM=Hv*2*K*K*K/C*Tc + (cp-1)*Hv*2*K*K*l.dv, pkM=(flKW+flM)/(flKW/pkA+flM/(pkA/3)), byM=Tc*Hv*K*2*2 + seqs*Hv*K*(K+l.dv)*4*cp;
      op("lcore",Msg.of("engine_train.op.lcoreM",{kind:l.kind}),"lin",l.n,{fl:flKW+flM,by:byM,pk:pkM,k:2},{fl:2*(flKW+flM),by:2*byM,pk:pkM,k:3},null);
    }
    op("lnorm",Msg.of("engine_train.op.lnorm"),"lin",l.n,...elem(10*Hv*l.dv,3*Hv*l.dv*2,Tc,2));
    op("lout",Msg.of("engine_train.op.lout"),"lin",l.n,...gemm(Tc,l.Hv*l.dv,D,tp));
    tpBlocks.push({name:Msg.of("engine_train.blk.lin"),blk:"lin",layers:l.n,inW:D});
  }

  /* ---- MoE ---- */
  const nM=d.nMoe, De=d.De, imb=g.lb||ep<=1 ? 1 : c.imb||1;     // 理想下限：EP 负载完全均衡；没有 EP 就没有组间不均
  const dedup = ["deepep","hybridep","ncclep","distmoe"].includes(c.dispatcher);
  const perRankExp = e.E/ep;
  const kEff = dedup ? ep*Par.pDest(e.E, ep, e.k) : e.k;     // 去重后每 token 发往的不同 rank 数（含本 rank）
  const R = Tl*e.k*imb;                                     // 本 rank 上 expert 处理的行数（EP 组内均匀）
  op("norm",Msg.of("engine_train.op.normMoe"),"misc",nM,...elem(8*D/tp,3*D*2/tp,Tc,2));
  op("router",mg?Msg.of("engine_train.op.routerMg"):Msg.of("engine_train.op.routerTt"),"moe",nM,{fl:2*Tl*D*e.E+10*Tl*e.E,by:D*e.E*4+Tl*(D*2+e.E*4*3),pk:pkA/2},{fl:4*Tl*D*e.E,by:D*e.E*8+Tl*e.E*4*3,pk:pkA/2},null);
  if(e.Dlat) op("latd",Msg.of("engine_train.op.latd",{n:e.Dlat}),"moe",nM,...gemm(Tl,D,e.Dlat,1));
  op("permute",c.dispatcher==="distmoe"?Msg.of("engine_train.op.permuteDist"):Msg.of("engine_train.op.permute"),"moe",nM,{fl:0,by:Tl*De*2+R*De*2,pk:pkA},{fl:0,by:Tl*De*2+R*De*2,pk:pkA},null);
  op("moe1",Msg.of("engine_train.op.moe1"),"moe",nM,...gemm(R,De,2*e.F*etp,etp,perRankExp));
  op("act",Msg.of("engine_train.op.act",{fn:m.family==="kimi"?"SiTU-GLU":"SwiGLU"}),"moe",nM,...elem(6*e.F,(2*e.F+e.F)*2,R,2));
  op("moe2",Msg.of("engine_train.op.moe2"),"moe",nM,...gemm(R,e.F*etp,De,etp,perRankExp));
  op("unperm",Msg.of("engine_train.op.unperm"),"moe",nM,{fl:R*De,by:R*De*2+Tl*De*2,pk:pkA},{fl:0,by:R*De*2+Tl*De*2,pk:pkA},null);
  if(e.Dlat) op("latu",Msg.of("engine_train.op.latu",{n:e.Dlat}),"moe",nM,...gemm(Tl,e.Dlat,D,1));
  if(e.Fs>0){ op("shared",Msg.of("engine_train.op.shared"),"moe",nM,...[0,1,2].map(i=>{ const a=gemm(Tc,D,3*e.Fs,tp)[i]; return a; }), {ovl: mg&&c.sharedOverlap}); tpBlocks.push({name:Msg.of("engine_train.blk.shared"),blk:"moe",layers:nM,inW:D}); }
  if(m.dense.n>0){ op("norm",Msg.of("engine_train.op.normDense"),"misc",m.dense.n,...elem(8*D/tp,3*D*2/tp,Tc,2)); op("dense",Msg.of("engine_train.op.dense"),"moe",m.dense.n,...gemm(Tc,D,3*m.dense.F,tp)); tpBlocks.push({name:Msg.of("engine_train.blk.dense"),blk:"dense",layers:m.dense.n,inW:D}); }

  /* ---- 输出头：整段序列的 logits + CE（推理只需最后一个 token） ---- */
  const chunked = !mg;
  op("lm",Msg.of("engine_train.op.lm"),"lm",1,...gemm(Tc,D,m.V,tp));
  op("ce",(mg?Msg.of("engine_train.op.ceMg"):Msg.of("engine_train.op.ceTt")),"lm",1,{fl:5*Tc*m.V/tp,by:Tc*m.V/tp*(2+4)*2,pk:pkA},{fl:3*Tc*m.V/tp,by:Tc*m.V/tp*(4+2),pk:pkA},null);
  // 词表并行的一对：embedding 前向 RS 到序列分片、反向 AG；lm_head 前向 AG 输入、反向 RS。合起来和一个 Column + Row 块的四次一样
  if(tp>1) tpBlocks.push({name:Msg.of("engine_train.blk.head"),blk:"head",layers:1,inW:D});

  /* ---- 通信（每 micro-batch，每 rank） ---- */
  const tpX = (pass,op,what,msg) => xc(pass,op,what,msg*(tp-1)/tp,agT(msg,tp,lkTP,g));
  if(mg && tp>1){
    // Column 的输入在 SP 下只存本卡那份：前向 AG 出全序列；反向 Row 输出的 RS 变成 AG 梯度，Column 为 wgrad 再 AG 一次输入，输入梯度 RS 回去
    for(const b of tpBlocks){ const msgAG=Tc*b.inW*2, msgRS=Tc*D*2, inN=b.inW===D?Msg.of("engine_train.tp.inX"):Msg.of("engine_train.tp.inLatent");
      const xs=[tpX("fwd","AG",Msg.of("engine_train.tp.fwdAG",{input:inN}),msgAG), tpX("fwd","RS",Msg.of("engine_train.tp.fwdRS"),msgRS)];
      if(rcFull) xs.push(tpX("rc","AG",Msg.of("engine_train.tp.rcAG",{input:inN}),msgAG), tpX("rc","RS",Msg.of("engine_train.tp.rcRS"),msgRS));
      xs.push(tpX("bwd","AG",Msg.of("engine_train.tp.bwdAG"),msgRS), tpX("bwd","AG",Msg.of("engine_train.tp.bwdAGInput",{input:inN}),msgAG), tpX("bwd","RS",Msg.of("engine_train.tp.bwdRS",{input:inN}),msgAG));
      cmX("tp",rcFull?Msg.of("engine_train.tp.nameRc",{blk:b.name}):Msg.of("engine_train.tp.name",{blk:b.name}),"tp",b.layers,b.blk,xs,{ovl:c.tpOverlap?"comp":false,lk:lkTPx}); }
    if(f.kind==="dsa") cmX("tpidx",Msg.of("engine_train.tp.idx"),"tp",f.n,"attn",[tpX("fwd","AG",Msg.of("engine_train.tp.idxX"),Tc*D*2),tpX("fwd","AG",Msg.of("engine_train.tp.idxQ"),Tc*f.qLora*2),tpX("bwd","RS",Msg.of("engine_train.tp.idxGradX"),Tc*D*2),tpX("bwd","RS",Msg.of("engine_train.tp.idxGradQ"),Tc*f.qLora*2)],{lk:lkTPx});
  } else if(!mg && tp>1){
    // DTensor：Column 输入 Shard → Replicate（AG），Row 输出 Partial → Shard（RS）；反向各自对偶，gather 后的输入留给 wgrad
    for(const b of tpBlocks){ const msg=Tc*D*2;
      const xs=[tpX("fwd","AG",Msg.of("engine_train.tpTt.fwdAG"),msg), tpX("fwd","RS",Msg.of("engine_train.tpTt.fwdRS"),msg)];
      if(rcFull) xs.push(tpX("rc","AG",Msg.of("engine_train.tpTt.rcAG"),msg), tpX("rc","RS",Msg.of("engine_train.tp.rcRS"),msg));
      xs.push(tpX("bwd","AG",Msg.of("engine_train.tpTt.bwdAG"),msg), tpX("bwd","RS",Msg.of("engine_train.tpTt.bwdRS"),msg));
      cmX("tp",rcFull?Msg.of("engine_train.tpTt.nameRc",{blk:b.name}):Msg.of("engine_train.tpTt.name",{blk:b.name}),"tp",b.layers,b.blk,xs,{ovl:c.tpOverlap?"comp":false,lk:lkTPx}); }
  }
  // CP：全注意力层走 Par.cpLayer（和“并行”页同一套公式）
  const cpMeth = mg ? ({p2p:"ring",allgather:"allgather",a2a:"ulysses"}[c.cpType]||"ring") : (Par.CP[c.cpType] ? c.cpType : "allgather");
  const cpLB = mg || c.cpLB!==false;                               // Megatron 的 CP 总是按 2C 段首尾配对
  const docs = T/S;                                                // 这一组 CP rank 一个 micro-batch 里的文档数
  const imbCP = (cp===1 || cpMeth==="ulysses" || cpLB || g.lb) ? 1 : 2-1/cp;   // 理想下限：CP 也按均衡
  let cpFull=null, cpSwa=null;
  if(cp>1 && f.n>0){
    if(imbCP>1 && coreFullOp){ coreFullOp.fwd.t*=imbCP; coreFullOp.dgrad.t*=imbCP; coreFullOp.imb=imbCP; }
    const nSwa = f.kind==="gqa" ? (f.swa||0) : 0, nFull = f.n-nSwa;
    const per = (o2, n2) => o2 ? (o2.fwd.t + o2.dgrad.t + (rc("acore")?o2.fwd.t:0))/n2/docs : 0;   // 一层一篇的核心 F+B（含重算）
    const x = { S, C:cp, tp, m, d, hw, topo, g, ac:rcFull };
    cpFull = Par.cpLayer(cpMeth, Object.assign({ tCore:per(coreFullOp,nFull), kind:"full" }, x));
    if(!cpFull.ok) warn.push(cpFull.why);
    const how = { ring:Msg.of("engine_train.cp.how.ring"), allgather:Msg.of("engine_train.cp.how.allgather"), ulysses:Msg.of("engine_train.cp.how.ulysses") }[cpMeth];
    const perDocs = o2 => o2.xs.map(x=>xc(x.pass,x.op,x.what,x.bytes*docs,x.t*docs));     // 一个 micro-batch 有 docs 篇，每篇各走一遍
    cmX("cpkv",f.kind==="dsa"?Msg.of("engine_train.cp.kvDsa",{meth:Par.CP[cpMeth].name,how}):Msg.of("engine_train.cp.kv",{meth:Par.CP[cpMeth].name,how}),"cp",nFull,"full",perDocs(cpFull),{ovl:"fixed",expT:nFull*docs*cpFull.exposed,lk:cpFull.lk,par:"cp"});
    if(nSwa){ cpSwa = Par.cpLayer(cpMeth, Object.assign({ tCore:per(coreSwaOp,nSwa), kind:"swa" }, x));
      cmX("cpswa",cpMeth==="ulysses"?Msg.of("engine_train.cp.swaUlysses"):Msg.of("engine_train.cp.swa",{w:f.W}),"cp",nSwa,"swa",perDocs(cpSwa),{ovl:"fixed",expT:nSwa*docs*cpSwa.exposed,lk:cpSwa.lk,par:"cp"}); }
  }
  if(cp>1 && l.n>0){
    const Hv=l.Hv/tp;
    const rcLin = rcFull||sel.has("gdn");
    if(linHead){ if(Hv%cp) warn.push(Msg.of("engine_train.warn.linHeadCp",{hv:Hv,cp}).invalid());
      const sIn=Tc*d.linIn/tp*2, sOut=Tc*Hv*l.dv*2, aX=(pass,what,s)=>xc(pass,"A2A",what,s*(cp-1)/cp,a2aT(s,cp,lkCP,g));
      const xs=[aX("fwd",Msg.of("engine_train.cpLin.fwdIn"),sIn), aX("fwd",Msg.of("engine_train.cpLin.fwdOut"),sOut)];
      if(rcLin) xs.push(aX("rc",Msg.of("engine_train.cpLin.rcIn"),sIn), aX("rc",Msg.of("engine_train.cpLin.rcOut"),sOut));
      xs.push(aX("bwd",Msg.of("engine_train.cpLin.bwdOut"),sOut), aX("bwd",Msg.of("engine_train.cpLin.bwdIn"),sIn));
      cmX("cplin",rcLin?Msg.of("engine_train.cpLin.headRc",{kind:l.kind}):Msg.of("engine_train.cpLin.head",{kind:l.kind}),"cp",l.n,"lin",xs,{lk:lkCPx}); }
    else {
      // 本卡有不止一段序列时（S < Tc）FLA 只交换首尾两段的 [S_ext | M]：(2, HV, K, V+K)；否则一段。字节和序列长度无关
      const seqs=S<Tc?2:1, msg=cp*seqs*Hv*l.dk*(l.dv+l.dk)*4, gX=(pass,what)=>xc(pass,"AG",what,msg*(cp-1)/cp,agT(msg,cp,lkCP,g));
      // 短卷积的 halo：每段开头要前一段的最后 conv−1 个 token
      const halo=(l.conv-1)*d.linConvDim/tp*2, hX=(pass,what)=>xc(pass,"P2P",what,halo,halo/(lkCP.bw*g.etaN)+(lkCP.intra?g.alphaNv:g.alphaIb)*1e-6);
      const xs=[hX("fwd",Msg.of("engine_train.cpLin.haloFwd")), gX("fwd",Msg.of("engine_train.cpLin.chunkFwd"))];
      if(rcLin) xs.push(hX("rc",Msg.of("engine_train.cpLin.haloRc")), gX("rc",Msg.of("engine_train.cpLin.chunkRc")));
      xs.push(gX("bwd",Msg.of("engine_train.cpLin.chunkBwd")), hX("bwd",Msg.of("engine_train.cpLin.haloBwd")));
      cmX("cplin",rcLin?Msg.of("engine_train.cpLin.chunkNameRc",{kind:l.kind}):Msg.of("engine_train.cpLin.chunk",{kind:l.kind}),"cp",l.n,"lin",xs,{lk:lkCPx}); }
  }
  // EP
  let a2aPerLayer=0, epL=null;
  if(ep>1 && !mg){
    const be = Par.EP[c.dispatcher] ? c.dispatcher : "standard";
    let tEF=0, tEB=0;
    for(const o of ops) if(["moe1","act","moe2"].includes(o.key)){ tEF+=o.fwd.t/nM; tEB+=(o.dgrad.t+o.wgrad.t+(rc(o.key)?o.fwd.t:0))/nM; }
    epL = Par.epLayer(be, { n:Tl, K:e.k, Z:e.E, E:ep, De, stride:1, hw, topo, g, ac:rcFull, prec:PR.f, imb, tExpF:tEF, tExpB:tEB, syncUs:g.lb?0:c.hostSyncUs, mokF:c.mokF });
    if(!epL.ok) warn.push(epL.why);
    a2aPerLayer = epL.tw + epL.tAlpha;
    const ibShare = epL.wire>0 ? Math.min(1, epL.ib/epL.wire) : 0;
    cmX("a2a",Par.EP[be].fused?Msg.of("engine_train.ep.a2aFused",{be:Par.EP[be].name,n:epL.exch}):Msg.of("engine_train.ep.a2a",{be:Par.EP[be].name,n:epL.exch}),"ep",nM,"moe",epL.xs,{exch:epL.exch,ovl:"fixed",expT:nM*(epL.extra-epL.tSync),lk:epL.lk,ibShare,par:"ep",fused:!!Par.EP[be].fused});
    // 计数交换：每卡把 E 个 int64 计数 all-to-all 出去（torchtitan AllToAllTokenDispatcher._token_count_exchange），重算时再来一次
    if(epL.syncs){ const one=xc("fwd","sync",be==="deepep"?Msg.of("engine_train.ep.syncWhatDeepep"):Msg.of("engine_train.ep.syncWhat"),e.E*8*(ep-1)/ep,epL.tSync/epL.syncs);
      cmX("cnt",be==="deepep"?Msg.of("engine_train.ep.syncDeepep",{n:epL.syncs}):Msg.of("engine_train.ep.sync",{n:epL.syncs}),"ep",nM,"moe",epL.syncs>1?[one,Object.assign({},one,{pass:"rc",what:Msg.of("engine_train.ep.syncRc")})]:[one],{ovl:"fixed",expT:nM*epL.tSync,par:"ep"}); }
  } else if(ep>1){
    const nA = 4 + (rcA2A?2:0);
    const sent = Tl*kEff*De*2;
    const t1 = a2aT(sent,ep,lkEP,g)*(c.dispatcher==="ncclep"?0.55:1), b1 = sent*(ep-1)/ep;
    a2aPerLayer=nA*t1;
    const name = {alltoall:Msg.of("engine_train.ep.disp.alltoall"),deepep:Msg.of("engine_train.ep.disp.deepep"),hybridep:Msg.of("engine_train.ep.disp.hybridep"),ncclep:Msg.of("engine_train.ep.disp.ncclep"),standard:Msg.of("engine_train.ep.disp.standard"),distmoe:Msg.of("engine_train.ep.disp.distmoe")}[c.dispatcher]||c.dispatcher;
    const xs=[xc("fwd","A2A",Msg.of("engine_train.ep.dispatch"),b1,t1), xc("fwd","A2A",Msg.of("engine_train.ep.combine"),b1,t1)];
    if(rcA2A) xs.push(xc("rc","A2A",Msg.of("engine_train.ep.rcDispatch"),b1,t1), xc("rc","A2A",Msg.of("engine_train.ep.rcCombine"),b1,t1));
    xs.push(xc("bwd","A2A",Msg.of("engine_train.ep.bwdCombine"),b1,t1), xc("bwd","A2A",Msg.of("engine_train.ep.bwdDispatch"),b1,t1));
    cmX("a2a",Msg.of("engine_train.ep.a2a",{be:name,n:nA}),"ep",nM,"moe",xs,{exch:nA,ovl: c.dispatcher==="distmoe"?"moe" : (mg&&c.epOverlap?"comp":false),fused:c.dispatcher==="distmoe"});
    if(!["hybridep","distmoe"].includes(c.dispatcher)){ const one=xc("fwd","sync",Msg.of("engine_train.ep.cntWhat"),ep*e.E*8/(1+(rcA2A?1:0)),(lkEP.intra?g.alphaNv:g.alphaIb)*1e-6+(g.lb?0:c.hostSyncUs)*1e-6);
      cmX("cnt",Msg.of("engine_train.ep.cnt"),"ep",nM,"moe",rcA2A?[one,Object.assign({},one,{pass:"rc",what:Msg.of("engine_train.ep.cntRc")})]:[one]); }
  }
  if(mg && etp>1){ const msg=R*etp*De*2, eX=(pass,op,what,m2)=>xc(pass,op,what,m2*(etp-1)/etp,agT(m2,etp,lkTP,g));
    cmX("etp",Msg.of("engine_train.etp.name"),"ep",nM,"moe",[eX("fwd","AG",Msg.of("engine_train.etp.fwdAG"),msg),eX("fwd","RS",Msg.of("engine_train.etp.fwdRS"),2*msg),eX("bwd","AG",Msg.of("engine_train.etp.bwdAG"),2*msg),eX("bwd","RS",Msg.of("engine_train.etp.bwdRS"),msg)],{dim:"etp"}); }
  // PP p2p
  if(pp>1){ const msg=Tl*D*2, t1=(msg/(lkPP.bw*g.etaN)+(lkPP.intra?g.alphaNv:g.alphaIb)*1e-6)*vpp, v=vpp>1?Msg.of("engine_train.pp.vstage",{n:vpp}):"";
    cmX("pp",Msg.of("engine_train.pp.name"),"pp",1,"stage",[xc("fwd","P2P",Msg.of("engine_train.pp.fwd",{v}),msg*vpp,t1),xc("bwd","P2P",Msg.of("engine_train.pp.bwd",{v}),msg*vpp,t1)],{ovl:"comp",perStage:true}); }

  if(tokB>0) op("tok",Msg.of("engine_train.op.tok"),"misc",1,{fl:tokB/4,by:tokB,pk:pkA},null,null);

  /* ---- 汇总一个 micro-batch（整模型，按 rank） ---- */
  let tF=0,tD=0,tW=0,tR=0, flF=0,flB=0, moeT=0;
  for(const o of ops){ tF+=o.fwd.t; tD+=o.dgrad.t; tW+=o.wgrad.t; flF+=o.fwd.fl; flB+=o.dgrad.fl+o.wgrad.fl;
    if(rc(o.key)) { tR+=o.fwd.t; o.rc=true; }
    if(["moe1","moe2","act"].includes(o.key)) moeT+=o.fwd.t+o.dgrad.t+o.wgrad.t; }
  const attnT = ops.filter(o=>o.key==="acore").reduce((s,o)=>s+o.fwd.t+o.dgrad.t,0);
  const compMB = tF+tD+tW+tR;
  let exposed=0, hidden=0;
  for(const x of comm){ if(x.perStage) continue;
    const ex = x.ovl==="fixed" ? Math.min(x.t, x.expT) : x.ovl==="moe" ? Math.max(0,x.t-moeT) : x.ovl==="attn" ? Math.max(0,x.t-attnT)
             : x.ovl==="comp" ? x.t*0.15 : x.t;                                    // comp：userbuffers / combined 1F1B，经验上约 85% 可藏
    x.exp=ex; exposed+=ex; hidden+=x.t-ex; }
  if(mg && c.sharedOverlap){ const sh=ops.find(o=>o.key==="shared"); if(sh){ const save=Math.min(sh.fwd.t+sh.dgrad.t+sh.wgrad.t, a2aPerLayer*nM*0.5); exposed=Math.max(0,exposed-save); } }

  // FSDP2（torchtitan）：每层每个 micro-batch AG 参数（fwd，reshard 时 bwd 再一次）+ RS 梯度；HSDP 时分片组缩到 1/R，副本之间每步 all-reduce
  const Pdense = d.pDenseAll, Pexp = d.pExpertsAll;
  let fsdpMB=0, fsdp=null;
  if(!mg){
    const R = Math.max(1, c.hsdp||1), fs = dp*cp/R, fe = edp/R;
    if(!Number.isInteger(fs) || (nM>0 && !Number.isInteger(fe))) warn.push(Msg.of("engine_train.warn.hsdp",{r:R,dpcp:dp*cp,edp}).invalid());
    const reshard = c.reshard ?? (pp===1), aG = reshard?2:1, gB = c.gradDt==="bf16"?2:4;
    const lkD = Par.lk(hw,topo,fs,tp), lkE = Par.lk(hw,topo,fe,ep), lkR = Par.lk(hw,topo,R,fs*tp);
    const pRep = nM ? (d.pShared+d.pRouter+d.pLat)/tp : 0;           // MoE 层里随 dense 一起切的部分（shared、router、latent 投影）
    const pFull = f.n ? d.pAttn/tp + pRep : 0, pLin = l.n ? d.pLin/tp + pRep : 0, pDen = d.pDense/tp, pEH = (d.pEmb+d.pHead)/tp, pExpL = nM ? e.E*d.pExpert/ep : 0;
    // torchtitan apply_fsdp_to_decoder：每个 transformer 块一个 FSDP 单元，attention + MLP 的非 expert 部分在 dense 分片组（d），
    // routed expert 在 expert 分片组（e）。pFull / pLin 是 MoE 块的 d（界面的说明用）；dense FFN 块没有 shared / router
    const blocks = layerPlan(m).map(x => ({ d: (x.attn==="lin" ? d.pLin : d.pAttn)/tp + (x.mlp==="moe" ? pRep : pDen), e: x.mlp==="moe" ? pExpL : 0 }));
    // [参数, 块数, 每块 gather 几次]，大小相同的块合成一项。tok_embeddings 反向不 gather（set_unshard_in_backward(False)），
    // [norm, lm_head] 只在 policy = always 时 reshard，所以两者每个 micro-batch 只在前向 gather 一次
    const same = new Map(); for(const b of blocks) same.set(b.d, (same.get(b.d)||0)+1);
    const lay = [...same].map(([P,n2])=>[P,n2,aG]).concat([[d.pEmb/tp,1,1],[d.pHead/tp,1,1]]);
    const tDense = lay.reduce((a,[P,n2,ag])=>a+n2*Par.fsdpT(P,fs,lkD,g,ag,gB),0)/pp, bDense = lay.reduce((a,[P,n2,ag])=>a+n2*Par.fsdpBytes(P,fs,ag,gB),0)/pp;
    const tExp = nM*Par.fsdpT(pExpL,fe,lkE,g,aG,gB)/pp, bExp = nM*Par.fsdpBytes(pExpL,fe,aG,gB)/pp;
    // 每个 micro-batch 的逐次清单（每卡线上字节）：dim = dp（dense 分片组）/ edp（expert 分片组），calls = 这一项发几次
    const fsdpX = (P,F,l,n2,ag,dim) => n2>0 ? Par.fsdpDet(P,F,l,g,ag,gB).map(x=>Object.assign(x,{dim,calls:n2/pp,bytes:x.bytes*n2/pp,t:x.t*n2/pp})) : [];
    const xs = lay.flatMap(([P,n2,ag])=>fsdpX(P,fs,lkD,n2,ag,"dp")).concat(fsdpX(pExpL,fe,lkE,nM,aG,"edp"));
    const firstAG = Par.wireT(blocks[0].d*(fs-1)/fs*2, lkD, g.etaAg, 1, g) + (fe>1 && blocks[0].e ? Par.wireT(blocks[0].e*(fe-1)/fe*2, lkE, g.etaAg, 1, g) : 0);
    fsdpMB = tDense + tExp;
    fsdp = { R, fs, fe, aG, gB, reshard, lkD, lkE, lkR, pFull, pLin, pDen, pEH, pExpL, blocks, tDense, tExp, bDense, bExp, firstAG, xs };
    comm.push({key:"fsdp",name:Msg.of("engine_train.fsdp.name",{passes:Msg.of(reshard?"engine_train.fsdp.passesBoth":"engine_train.fsdp.passesFwd"),dt:gB===4?"FP32":"BF16",fs,fe,hsdp:R>1?Msg.of("engine_train.fsdp.hsdp",{r:R}):""}),cls:"dp",layers:1,passes:1,t:fsdpMB,bytes:bDense+bExp,ovl:"prefetch",perStage:true,par:"fsdp"});
  }

  // PP：层按 stage 平分，lm_head 和 CE 都在最后一段。下限按能把层搬开算（不低于头本身）
  const tHead = ops.filter(o=>o.group==="lm").reduce((s,o)=>s+o.fwd.t+o.dgrad.t+o.wgrad.t+(rc(o.key)?o.fwd.t:0),0);
  const stageComp = pp<=1 ? compMB : g.lb ? Math.max(compMB/pp,tHead) : tHead+(compMB-tHead)/pp;
  const tMBstage = stageComp + exposed/pp;
  // FSDP 在后台流上跑，但和 CP / EP 抢同一条链路：一个 micro-batch 至少要等最忙那条链路跑完；第一层的 gather 藏不住
  let fsdpExposedMB = 0;
  if(fsdp){
    const lane = { nvl:0, ib:0 }, put = (l2,t) => { if(t>0) lane[l2.intra?"nvl":"ib"] += t; };
    put(fsdp.lkD, fsdp.tDense); put(fsdp.lkE, fsdp.tExp);
    for(const x of comm) if(!x.perStage && x.lk){ if(x.ibShare!=null){ lane.ib += x.t*x.ibShare/pp; lane.nvl += x.t*(1-x.ibShare)/pp; } else put(x.lk, x.t/pp); }
    fsdp.lane = lane;
    fsdpExposedMB = Math.max(0, Math.max(lane.nvl, lane.ib) - tMBstage) + fsdp.firstAG;
  }
  const ppComm = comm.filter(x=>x.key==="pp").reduce((s,x)=>s+x.t,0)*0.1;
  const tMB = tMBstage + fsdpExposedMB + ppComm;

  /* ---- 步级：micro-batch 数、PP bubble、DP 同步、优化器 ---- */
  const microsPerStep = Math.max(1, Math.ceil(c.Gtok/(dp*T)));
  const bubble = pp>1 ? (pp-1)/vpp : 0;
  const tPipe = tMB*(microsPerStep + bubble);
  // 每 rank 的本地参数
  const PlD = Pdense/(tp*pp), PlE = Pexp/(ep*etp*pp);
  let tDP=0, tDPexp=0, dpNote="";
  const stepX=[];                                                    // 每个 step 一次的集合通信（dim：dp / edp / rep）
  if(mg){
    const dd=dp*cp, sx=(dim,op,what,msg,n,l2)=>{ if(n>1&&msg>0) stepX.push(Object.assign(xc("step",op,what,msg*(n-1)/n,agT(msg,n,l2,g)),{dim,n,lk:Par.lk(hw,topo,n,dim==="dp"?tp:etp*ep)})); };
    sx("dp","RS",Msg.of("engine_train.dp.rsDense"),PlD*4,dd,lkDP); sx("edp","RS",Msg.of("engine_train.dp.rsExpert"),PlE*4,edp,lkEDP);
    sx("dp","AG",Msg.of("engine_train.dp.agDense"),PlD*2,dd,lkDP); sx("edp","AG",Msg.of("engine_train.dp.agExpert"),PlE*2,edp,lkEDP);
    const rs = agT(PlD*4,dd,lkDP,g) + agT(PlE*4,edp,lkEDP,g), ag = agT(PlD*2,dd,lkDP,g)+agT(PlE*2,edp,lkEDP,g);
    tDP = rs+ag;
    const lastBwd = (tD+tW)/pp, firstFwd=tF/pp;
    const hid = g.lb ? 1 : 0.9;                                       // 能藏到最后一段反向 / 第一段前向里的比例（下限按全部）
    tDPexp = (c.overlapGrad? Math.max(0,rs-lastBwd*hid) : rs) + (c.overlapParam? Math.max(0,ag-firstFwd*hid) : ag);
    dpNote=Msg.of("engine_train.dp.noteMg",{dd,edp});
  }
  const shardD = mg ? PlD/(dp*cp) : Pdense/(tp*pp)/fsdp.fs, shardE = mg ? PlE/edp : Pexp/(ep*pp)/fsdp.fe;
  const Pshard = shardD + shardE;
  let tOpt, optNote;
  if(c.offload){
    const host=hw.host*1e9;
    const xfer = Pshard*(4+2)/host;                                   // grad D2H + param H2D
    const cpu = Pshard*16*2/(c.cpuAdamGBs*1e9);                       // CPU Adam：读写 master + m + v
    tOpt = c.overlapOffload ? Math.max(xfer,cpu) : xfer+cpu;
    optNote=c.overlapOffload?Msg.of("engine_train.opt.cpuOverlap"):Msg.of("engine_train.opt.cpuSerial");
  } else { tOpt = Pshard*30/(hw.hbm*1e9*g.etaM) + Pshard*4/(hw.hbm*1e9*g.etaM); optNote=Msg.of("engine_train.opt.gpu"); }
  const tSmallAR = 4*((lkDP.intra?g.alphaNv:g.alphaIb)*1e-6)*2;       // grad norm、aux-free bias、loss finite 等标量 all-reduce
  let tAR = 0;
  if(fsdp && fsdp.R>1){ tAR = Par.wireT((g.lb ? 1 : 2*(fsdp.R-1)/fsdp.R)*Pshard*fsdp.gB, fsdp.lkR, g.etaAg, 1, g); tDP = tAR; tDPexp = Math.max(0, tAR - (tD+tW)/pp*(g.lb?1:0.9));
    stepX.push(Object.assign(xc("step","AR",Msg.of("engine_train.dp.arRep",{dt:fsdp.gB===4?"FP32":"BF16"}),2*(fsdp.R-1)/fsdp.R*Pshard*fsdp.gB,tAR),{dim:"rep",n:fsdp.R,lk:fsdp.lkR}));
    dpNote = Msg.of("engine_train.dp.noteHsdp",{r:fsdp.R,link:fsdp.lkR.kind}); }
  const stepT = tPipe + tDPexp + tOpt + tSmallAR;
  const tokens = microsPerStep*dp*T;
  const mfu = (flF+flB)/pp*microsPerStep/(stepT*pkF);

  /* ---- 显存 ---- */
  const actTok = {};      // 每 token 每层保存的激活字节（本 rank）
  const sp=tp;
  const attnAct = f.n>0 ? (f.kind==="gqa" ? (2*D*2 + (2*f.Hq+2*f.Hkv)*f.dh*2 + 4*f.Hq + (f.gate?2*f.Hq*f.dh*2:0))/sp
                 : (2*D*2 + (f.qLora+f.kvLora)*2 + (sel.has("mla_up_proj")?0:f.H*((f.nope+f.rope)*3+f.nope+f.vh)*2) + f.H*f.vh*2 + 4*f.H + (f.kind==="dsa"?f.topk*4*sp + (f.idxH*f.idxD+f.idxD)*2*sp:0))/sp) : 0;
  const linAct = l.n>0 ? (2*D*2 + d.linIn*2 + d.linConvDim*2 + l.Hv*(l.kind==="KDA"?1300:650)*(l.dk+l.dv)/256 + l.Hv*l.dv*4*2 + l.Hv*l.dv*2)/sp : 0;
  const rowAct = e.k*imb*(De*2 + 2*e.F*2*(ttDef?0:1) + e.F*2*(sel.has("moe_act")?0:1) + 4);
  const moeAct = (2*D*2 + e.E*4*3)/sp + (sel.has("moe")?0:rowAct)/1 + (e.Dlat?(D+e.Dlat)*2:0) + (e.Fs>0&&!sel.has("shared_experts")?3*e.Fs*2/tp:0);
  const denseAct = 2*D*2+3*m.dense.F*2/tp;
  let perTokLayers;
  if(rcFull){ perTokLayers = (m.L/pp)*D*2/sp; }
  else perTokLayers = (f.n*attnAct + l.n*(sel.has("gdn")?D*2/sp:linAct) + d.nMoe*moeAct + m.dense.n*denseAct)/pp;
  const oneLayerWS = Math.max(attnAct,linAct) + Math.max(d.nMoe>0?moeAct:0, m.dense.n>0?denseAct:0);   // 重算时一层的工作集（MoE 层或 dense 层里大的那个）
  const inflight = pp>1 ? Math.min(microsPerStep,pp)*(vpp>1?1+(pp-1)/(pp*vpp):1) : 1;
  const act = perTokLayers*Tc*inflight + (rcFull?oneLayerWS*Tc:0);
  const logits = Tc*m.V/tp*(mg?6:6/8);
  const a2aBuf = nM<=0 ? 0 : epL ? epL.buf : Tl*e.k*imb*De*2*(ep>1?2:1);   // 没有 EP 只有 permute 那一份
  const cpBuf = !cpFull ? 0 : Math.max((f.n-(cpSwa?f.swa:0))>0?cpFull.buf:0, cpSwa?cpSwa.buf:0);
  let states, paramsGPU, gathered=0;
  if(mg){
    paramsGPU = (PlD+PlE)*(2+4);                                      // bf16 参数 + fp32 main_grad
    states = c.offload ? 0 : shardD*12 + shardE*12;
  } else {
    paramsGPU = 0;
    states = c.offload ? 0 : Pshard*16;
    const gD = fsdp.fs>1 ? Pdense/tp/pp : 0, gE = fsdp.fe>1 ? Pexp/ep/pp : 0;   // 分片组只有 1 张卡时本来就是整份，不用 gather
    gathered = 2*(gD+gE)/(m.L/pp)*2*2;                                 // 当前 + 预取 block 的 bf16 参数与梯度
    // reshard 关掉（reshard_after_forward=False）时前向 gather 的 BF16 参数一直留到反向：本 stage 所有层的整份参数（对照 PopSoda2002/FSDP_memory_estimator 的 SHARD_GRAD_OP）
    if(!fsdp.reshard) gathered += (gD+gE)*2;
  }
  const work = c.workspaceGB*1e9;
  // 低精度权重缓存：前向一份；反向也低精度且权重是 1D 块时再从 BF16 量化一份列方向（2D 块共用一份）
  const qW = fq ? (mg ? PlD+PlE : Pdense/(tp*pp)+Pexp/(ep*pp))*(wbF + (bq&&(!PR.w2d||hw.sm<100)?wbB:0)) : 0;
  const used = paramsGPU + states + gathered + act + logits + a2aBuf + cpBuf + work + qW;
  const mem = { paramsGPU, states, gathered, act, logits, a2aBuf, cpBuf, work, qW, used, total:hw.mem*1e9, perTokLayers, inflight };

  return { ops, comm, tF, tD, tW, tR, compMB, exposed, hidden, tMB, tMBstage, fsdpMB, fsdpExposedMB, microsPerStep, bubble, tPipe, tDP, tDPexp, dpNote, tOpt, optNote, tSmallAR, stepT,
    tokens, tokPerGpu: tokens/(N*stepT), mfu, dp, edp, tp, cp, pp, vpp, ep, etp, T, Tc, Tl, R, kEff, Pshard, PlD, PlE, mem, warn, flF, flB, moeT, a2aPerLayer, nM, stepX,
    lk:{ tp:lkTP, cp:lkCP, ep:lkEP, pp:lkPP },
    par:{ cpMeth, cpLB, imbCP, docs, cpFull, cpSwa, coreFullOp, ep:epL, fsdp, tAR } };
}

/* 默认的训练配置（界面的初始值；预设和架构页都在它上面改）。卡数 N 由调用方按拓扑给 */
const BASE = Common.freeze({ fw:"megatron", tp:1, cp:1, pp:1, vpp:1, ep:8, etp:1, dispatcher:"alltoall", cpType:"p2p", linCP:"chunkwise", tok:16384, S:8192, Gtok:2097152,
  prec:"bf16", recompute:"full", rcModules:[], offload:true, overlapOffload:true, cpuAdamGBs:30, overlapGrad:true, overlapParam:true,
  epOverlap:false, sharedOverlap:false, tpOverlap:false, reshard:true, imb:1.15, hostSyncUs:20, workspaceGB:6,
  cpLB:true, hsdp:1, gradDt:"fp32", mokF:0.25, etaA:0 });

/* torchtitan 的默认选项：BASE 是 Megatron 的枚举，切到 torchtitan 时用这份盖上去，否则 BASE 的值在 torchtitan 下没有定义 */
const FW_DEF = Common.freeze({ megatron:{}, torchtitan:{ dispatcher:"standard", cpType:"allgather", linCP:"state", recompute:"default", offload:false } });

return { run, linFlops, PREC, BASE, FW_DEF, inert };
})();
