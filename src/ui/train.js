/* ================= 后训练 · 一个 step ================= */
const DISP = { megatron:[["alltoall",t("train.opt.disp.megatron.alltoall")],["deepep",t("train.opt.disp.megatron.deepep")],["hybridep",t("train.opt.disp.megatron.hybridep")],["ncclep",t("train.opt.disp.megatron.ncclep")]],
  torchtitan:[["standard",t("train.opt.disp.torchtitan.standard")],["ag",t("train.opt.disp.torchtitan.ag")],["deepep",t("train.opt.disp.torchtitan.deepep")],["distmoe",t("train.opt.disp.torchtitan.distmoe")],["mok",t("train.opt.disp.torchtitan.mok")]] };
const CPT = { megatron:[["p2p",t("train.opt.cpType.megatron.p2p")],["allgather",t("train.opt.cpType.megatron.allgather")],["a2a",t("train.opt.cpType.megatron.a2a")]], torchtitan:[["allgather",t("train.opt.cpType.torchtitan.allgather")],["ring",t("train.opt.cpType.torchtitan.ring")],["ulysses",t("train.opt.cpType.torchtitan.ulysses")]] };
const LCP = { megatron:[["headwise",t("train.opt.linCP.megatron.headwise")],["chunkwise",t("train.opt.linCP.megatron.chunkwise")]], torchtitan:[["state",t("train.opt.linCP.torchtitan.state")],["ulysses",t("train.opt.linCP.torchtitan.ulysses")]] };
const RCO = { megatron:[["full",t("train.opt.recompute.megatron.full")],["selective",t("train.opt.recompute.megatron.selective")],["none",t("train.opt.recompute.none")]], torchtitan:[["default",t("train.opt.recompute.torchtitan.default")],["full",t("train.opt.recompute.torchtitan.full")],["none",t("train.opt.recompute.none")]] };
const RCMODS = [["core_attn","core_attn"],["layernorm","layernorm"],["moe_act","moe_act"],["mla_up_proj","mla_up_proj"],["moe",t("train.opt.rcMod.moe")],["shared_experts","shared_experts"],["gdn","gdn"]];
function trnRun(over){ const {m,hw,topo,g}=ctx(); return Train.run(m,hw,topo,Object.assign({N:topo.gpus},S.trn,over||{}),g); }
function syncTrnRack(){
  const c=S.trn;
  qa("[data-fw]").forEach(el=>el.hidden=el.dataset.fw!==c.fw);
  for(const [id,L,k] of [["t-disp",DISP,"dispatcher"],["t-cptype",CPT,"cpType"],["t-lincp",LCP,"linCP"],["t-rc",RCO,"recompute"]]){
    if(!L[c.fw].some(x=>x[0]===c[k])) c[k]=L[c.fw][0][0]; fillOpts($(id),L[c.fw],c[k]); }
  $("t-rcmods").innerHTML=c.recompute==="selective"?'<p class="hint">--recompute-modules</p>'+RCMODS.map(x=>'<label class="check"><input type="checkbox" data-rcmod="'+x[0]+'"'+(c.rcModules.includes(x[0])?" checked":"")+'><span>'+x[1]+'</span></label>').join(""):"";
  presetSel($("t-preset"),$("t-preset-src"),trnPresets(c.fw,S.modelKey,S.topo),S.trnPreset);
  const {m,topo,g}=ctx(); markInert("trn",Train.inert(m,Object.assign({N:topo.gpus},c),g));
}
function renderTrain(){
  const {m,d,hw,topo}=ctx(), c=S.trn, mg=c.fw==="megatron";
  syncTrnRack();
  const r=trnRun(), M=r.mem;
  const warn=[...r.warn]; if(M.used>M.total) warn.push(t("train.warn.memOver",{used:fB(M.used),cap:hw.mem})); if(r.dp<1) warn.push(t("train.warn.tooManyGpus",{n:topo.gpus}));
  setWarn(warn);
  $("t-lede").innerHTML=t("train.lede.main",{model:esc(m.short),hw:hw.name,topo:esc(topo.name),fw:mg?"Megatron-Core":"torchtitan",tp:r.tp,sp:mg&&r.tp>1?t("train.lede.sp"):"",cp:r.cp,pp:r.pp,
    vpp:r.vpp>1?t("train.lede.vpp",{vpp:r.vpp}):"",dp:fI(r.dp),ep:r.ep,etp:mg?" × ETP "+r.etp:"",edp:fI(r.edp),tok:fInt(c.tok),tokens:fInt(r.tokens),micro:r.microsPerStep});
  kpis($("t-kpis"),[
    [t("train.kpi.step"),fT(r.stepT),t("train.kpi.stepSub",{pipe:fT(r.tPipe),opt:fT(r.tOpt)})],
    [t("train.kpi.thr"),fN(r.tokPerGpu)+" tok/s/GPU",t("train.kpi.thrSub",{mfu:(r.mfu*100).toFixed(1)})],
    [t("train.kpi.micro"),fT(r.tMB),t("train.kpi.microSub",{comp:fT(r.compMB/r.pp),exp:fT(r.exposed/r.pp)})],
    [t("train.kpi.bubble"),r.pp>1?pct(r.bubble,r.microsPerStep):t("train.kpi.bubbleNone"),r.pp>1?t("train.kpi.bubbleSub",{n:fI(r.bubble)}):""],
    [t("train.kpi.mem"),fB(M.used),t("train.kpi.memSub",{cap:hw.mem}),M.used>M.total],
    [t("train.kpi.params"),fN(r.PlD+r.PlE),t("train.kpi.paramsFoot",{d:fN(r.PlD),e:fN(r.PlE)})]]);
  // micro-batch 分解
  const pp=r.pp, ex=r.comm.filter(x=>!x.perStage), exSum=sum(ex,x=>x.exp||0), scale=exSum>0?r.exposed/exSum:0, byCls={};
  ex.forEach(x=>byCls[x.cls]=(byCls[x.cls]||0)+(x.exp||0)*scale);
  const clsName={tp:t("train.cls.tp"),cp:t("train.cls.cp"),ep:t("train.cls.ep"),dp:t("train.cls.dp"),pp:t("train.cls.pp")};
  stackBar($("t-bar"),$("t-barlist"),[
    {label:t("train.bar.fwd"),v:r.tF/pp,color:css("c-attn")},{label:t("train.bar.dgrad"),v:r.tD/pp,color:css("c-moe")},{label:t("train.bar.wgrad"),v:r.tW/pp,color:css("c-lin")},{label:t("train.bar.recompute"),v:r.tR/pp,color:css("c-cp")},
    ...Object.keys(byCls).map(k=>({label:clsName[k],v:byCls[k]/pp,color:CLSC[k],tag:pill("comm")})),
    {label:t("train.bar.fsdp"),v:r.fsdpExposedMB,color:css("c-fsdp"),tag:pill("comm")},
    {label:t("train.bar.pp"),v:sum(r.comm.filter(x=>x.key==="pp"),x=>x.t)*0.1,color:css("c-dp"),tag:pill("comm")}],r.tMB,fT);
  /* a2a 比 expert 计算还长时点一句；换后端值不值看 EP 那一节的对比，这里不替它下结论 */
  const moeL=d.nMoe?r.moeT/d.nMoe:0, epSlow=r.ep>1&&r.a2aPerLayer>moeL, barNote=$("t-bar-note");
  barNote.hidden=!epSlow; barNote.innerHTML=epSlow?t("train.bar.epNote",{ep:r.ep,a2a:V(fT(r.a2aPerLayer)),gemm:V(fT(moeL))}):"";
  $("t-step-lede").innerHTML=t("train.stepBar.lede",{n:r.microsPerStep,dp:esc(r.dpNote||t("train.stepBar.ledeFsdp")),opt:esc(r.optNote)});
  stackBar($("t-sbar"),$("t-sbarlist"),[
    {label:t("train.stepBar.micro",{n:r.microsPerStep}),v:r.tMB*r.microsPerStep,color:css("c-attn")},{label:t("train.stepBar.bubble"),v:r.tMB*r.bubble,color:css("c-free")},
    {label:t("train.stepBar.dp"),v:r.tDPexp,color:css("c-fsdp"),tag:pill("comm")},{label:t("train.stepBar.opt",{note:r.optNote}),v:r.tOpt,color:css("c-moe")},
    {label:t("train.stepBar.smallAR"),v:r.tSmallAR,color:css("c-ar"),tag:pill("comm")}],r.stepT,fT);
  // 逐算子
  table($("t-ops"),[t("train.ops.head.op"),t("train.ops.head.layers"),t("train.ops.head.fwd"),"dgrad","wgrad",t("train.ops.head.rc"),t("train.ops.head.bound"),t("train.ops.head.I"),t("train.ops.head.flops")],
    r.ops.map(o=>[sw(GROUPC[o.group])+esc(o.name),o.layers,fT(o.fwd.t),fT(o.dgrad.t),o.wgrad.t?fT(o.wgrad.t):"—",o.rc?"✓":"",pill(o.fwd.bound),fI(o.fwd.I),fF(o.fwd.fl+o.dgrad.fl+o.wgrad.fl)])
      .concat([{cls:"hl",cells:["<b>"+t("train.ops.total")+"</b>","",fT(r.tF),fT(r.tD),fT(r.tW),fT(r.tR),"","",fF(r.flF+r.flB)]}]),[1,2,3,4,7,8]);
  // 扫 token
  const toks=logspace(1024,65536,25).map(v=>Math.round(v/512)*512), ra=toks.map(t=>trnRun({tok:t}));
  plot($("t-sw-thr"),{x0:1024,x1:65536,y0:0,y1:Math.max(...ra.map(q=>q.tokPerGpu))*1.15,logx:true,fx:fxPow,fy:v=>fN(v),xl:t("train.sweep.xl"),yl:"tok/s/GPU",
    series:[{pts:toks.map((t,i)=>[t,ra[i].tokPerGpu]),color:css("c-attn"),w:2.4}],vlines:[{x:c.tok,color:css("ink"),label:t("train.sweep.cur")}],
    rects:toks.map((t,i)=>ra[i].mem.used>ra[i].mem.total?{x0:t*0.97,x1:t*1.15}:null).filter(Boolean)});
  plot($("t-sw-mem"),{x0:1024,x1:65536,y0:0,y1:Math.max(hw.mem*1.2,...ra.map(q=>q.mem.used/1e9*1.1)),logx:true,fx:fxPow,fy:v=>v+" GB",xl:t("train.sweep.xl"),yl:t("train.sweep.memYl"),
    series:[{pts:toks.map((t,i)=>[t,ra[i].mem.used/1e9]),color:css("c-act"),w:2.4},{pts:toks.map((t,i)=>[t,ra[i].mem.act/1e9]),color:css("c-act"),w:1.2,dash:"5 4",label:t("train.sweep.act")},{pts:[[1024,hw.mem],[65536,hw.mem]],color:css("warn"),w:1.2,dash:"3 3",label:"HBM "+hw.mem+" GB"}],
    vlines:[{x:c.tok,color:css("ink"),label:t("train.sweep.cur")}]});
  $("t-sw-legend").innerHTML='<span style="color:var(--c-attn)"><i></i>'+t("train.sweep.legendThr")+'</span><span style="color:var(--c-act)"><i></i>'+t("train.sweep.legendMem")+'</span><span>'+t("train.sweep.legendOver")+'</span>';
  // 显存
  stackBar($("t-mem"),$("t-memlist"),[
    {label:mg?t("train.mem.params"):"—",v:M.paramsGPU,color:css("c-w")},{label:t("train.mem.states"),v:M.states,color:css("c-e")},{label:t("train.mem.gathered"),v:M.gathered,color:css("c-fsdp")},
    {label:t("train.mem.act",{how:c.recompute==="full"?t("train.mem.actFull"):t("train.mem.actSaved"),inflight:r.pp>1?t("train.mem.actInflight",{n:fI(M.inflight)}):""}),v:M.act,color:css("c-act")},
    {label:mg?t("train.mem.logitsFp32"):t("train.mem.logitsChunked"),v:M.logits,color:css("c-lm")},{label:t("train.mem.a2a"),v:M.a2aBuf,color:css("c-a2a")},{label:t("train.mem.cpKV"),v:M.cpBuf,color:css("c-cp")},
    {label:t("train.mem.qW",{prec:esc(Train.PREC[c.prec].name)}),v:M.qW,color:css("c-moe")},
    {label:t("train.mem.work"),v:M.work,color:css("c-free")},{label:t("train.mem.free"),v:Math.max(0,M.total-M.used),color:css("soft")}],Math.max(M.total,M.used),fB);
  // 两个框架
  const cur=(k,name)=>c.fw===k?t("train.diff.current",{fw:name}):name;
  table($("t-diff"),["",cur("megatron","Megatron-Core"),cur("torchtitan","torchtitan")],[
    [t("train.diff.shard.k"),t("train.diff.shard.mg"),t("train.diff.shard.tt")],
    [t("train.diff.dpComm.k"),t("train.diff.dpComm.mg"),t("train.diff.dpComm.tt")],
    ["TP",t("train.diff.tp.mg"),t("train.diff.tp.tt")],
    ["CP",t("train.diff.cp.mg"),t("train.diff.cp.tt")],
    ["EP",t("train.diff.ep.mg"),t("train.diff.ep.tt")],
    [t("train.diff.rc.k"),t("train.diff.rc.mg"),t("train.diff.rc.tt")],
    [t("train.diff.lmHead.k"),t("train.diff.lmHead.mg"),t("train.diff.lmHead.tt")],
    [t("train.diff.linKernel.k"),"FLA chunk_gated_delta_rule / chunk_kda",t("train.diff.linKernel.tt")],
    [t("train.diff.opt.k"),t("train.diff.opt.mg"),t("train.diff.opt.tt")],
    [t("train.diff.mfu.k"),t("train.diff.mfu.mg"),t("train.diff.mfu.tt")]]);
}

/* 按算子的明细留在这里；每层的机制在模块页。点过去时带上序列长度和精度（模块页看一条长 S 的序列），"分三遍"的链接直接打开屋顶线的那个视图 */
$("t-to-mods").addEventListener("click",e=>{ const a=e.target.closest("a"); if(!a) return;
  if(a.getAttribute("href")==="#train/par/roof"){ roofView="pass"; return; }
  Object.assign(S.arch,{phase:"train",S:S.trn.S,prec:S.trn.prec}); syncInputs(); });
