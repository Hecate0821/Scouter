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
}
function renderTrain(){
  const {m,d,hw,topo,g}=ctx(), c=S.trn, mg=c.fw==="megatron";
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
  renderMesh(r); renderLayerGantt(r);
  // micro-batch 分解
  const pp=r.pp, ex=r.comm.filter(x=>!x.perStage), exSum=sum(ex,x=>x.exp||0), scale=exSum>0?r.exposed/exSum:0, byCls={};
  ex.forEach(x=>byCls[x.cls]=(byCls[x.cls]||0)+(x.exp||0)*scale);
  const clsName={tp:t("train.cls.tp"),cp:t("train.cls.cp"),ep:t("train.cls.ep"),dp:t("train.cls.dp"),pp:t("train.cls.pp")};
  stackBar($("t-bar"),$("t-barlist"),[
    {label:t("train.bar.fwd"),v:r.tF/pp,color:css("c-attn")},{label:t("train.bar.dgrad"),v:r.tD/pp,color:css("c-moe")},{label:t("train.bar.wgrad"),v:r.tW/pp,color:css("c-lin")},{label:t("train.bar.recompute"),v:r.tR/pp,color:css("c-cp")},
    ...Object.keys(byCls).map(k=>({label:clsName[k],v:byCls[k]/pp,color:CLSC[k],tag:pill("comm")})),
    {label:t("train.bar.fsdp"),v:r.fsdpExposedMB,color:css("c-fsdp"),tag:pill("comm")},
    {label:t("train.bar.pp"),v:sum(r.comm.filter(x=>x.key==="pp"),x=>x.t)*0.1,color:css("c-dp"),tag:pill("comm")}],r.tMB,fT);
  $("t-step-lede").innerHTML=t("train.stepBar.lede",{n:r.microsPerStep,dp:esc(r.dpNote||t("train.stepBar.ledeFsdp")),opt:esc(r.optNote)});
  stackBar($("t-sbar"),$("t-sbarlist"),[
    {label:t("train.stepBar.micro",{n:r.microsPerStep}),v:r.tMB*r.microsPerStep,color:css("c-attn")},{label:t("train.stepBar.bubble"),v:r.tMB*r.bubble,color:css("c-free")},
    {label:t("train.stepBar.dp"),v:r.tDPexp,color:css("c-fsdp"),tag:pill("comm")},{label:t("train.stepBar.opt",{note:r.optNote}),v:r.tOpt,color:css("c-moe")},
    {label:t("train.stepBar.smallAR"),v:r.tSmallAR,color:css("c-ar"),tag:pill("comm")}],r.stepT,fT);
  // 逐算子
  table($("t-ops"),[t("train.ops.head.op"),t("train.ops.head.layers"),t("train.ops.head.fwd"),"dgrad","wgrad",t("train.ops.head.rc"),t("train.ops.head.bound"),t("train.ops.head.I"),t("train.ops.head.flops")],
    r.ops.map(o=>[sw(GROUPC[o.group])+esc(o.name),o.layers,fT(o.fwd.t),fT(o.dgrad.t),o.wgrad.t?fT(o.wgrad.t):"—",o.rc?"✓":"",pill(o.fwd.bound),fI(o.fwd.I),fF(o.fwd.fl+o.dgrad.fl+o.wgrad.fl)])
      .concat([{cls:"hl",cells:["<b>"+t("train.ops.total")+"</b>","",fT(r.tF),fT(r.tD),fT(r.tW),fT(r.tR),"","",fF(r.flF+r.flB)]}]),[1,2,3,4,7,8]);
  // roofline
  const PR=Train.PREC[c.prec]||Train.PREC.bf16, pks=[{v:peakOf(hw,"bf16"),n:"BF16"}];
  for(const d of [PR.f,PR.b]) if(d!=="bf16" && !pks.some(p=>p.v===peakOf(hw,d))) pks.push({v:peakOf(hw,d),n:Common.peakKey(d).toUpperCase()});
  const pts=[], topT=new Set([...r.ops].sort((a,b)=>b.fwd.t-a.fwd.t).slice(0,8));
  for(const o of r.ops){ const col=GROUPC[o.group];
    if(o.fwd.fl>0) pts.push({x:o.fwd.I,y:o.fwd.fl/o.fwd.t,color:col,label:topT.has(o)?shortName(o.name):null});
    if(o.dgrad.fl>0) pts.push({x:o.dgrad.I,y:o.dgrad.fl/o.dgrad.t,color:col,shape:"sq",r:4});
    if(o.wgrad.fl>0) pts.push({x:o.wgrad.I,y:o.wgrad.fl/o.wgrad.t,color:col,shape:"tri",r:4}); }
  plot($("t-roof"),{h:380,x0:0.1,x1:20000,y0:1e10,y1:Math.max(...pks.map(p=>p.v))*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:t("train.roof.xl"),yl:t("train.roof.yl"),
    series:roofSeries(hw,g,pks,0.1,20000),points:pts,vlines:pks.map((p,i)=>({x:Common.ridge(p.v,hw,g),color:css("muted"),label:t("train.roof.knee",{peak:p.n,x:fI(Common.ridge(p.v,hw,g))}),dy:i*14}))});
  $("t-roof-legend").innerHTML='<span>'+t("train.roof.legendShapes")+'</span>'+Object.entries({attn:t("train.roof.group.attn"),lin:t("train.roof.group.lin"),moe:t("train.roof.group.moe"),lm:t("train.roof.group.lm"),misc:t("train.roof.group.misc")}).map(([k,v])=>'<span><span class="sw" style="background:'+GROUPC[k]+'"></span>'+v+'</span>').join("");
  // 通信表
  const parOf=x=>x.par||({cp:"cp",ep:"ep"})[x.cls]||(x.key==="fsdp"?"fsdp":"");
  const ovlName={moe:t("train.comm.ovl.moe"),attn:t("train.comm.ovl.attn"),comp:t("train.comm.ovl.comp"),prefetch:t("train.comm.ovl.prefetch")};
  table($("t-comm"),[t("train.comm.head.what"),t("train.comm.head.dim"),t("train.comm.head.bytes"),t("train.comm.head.time"),t("train.comm.head.ovl"),t("train.comm.head.exp"),""],
    r.comm.map(x=>[sw(CLSC[x.cls])+esc(x.name),x.cls.toUpperCase(),fB(x.bytes),fT(x.t),
      ovlName[x.ovl]||t("train.comm.ovl.serial"),
      x.key==="fsdp"?fT(r.fsdpExposedMB):x.key==="pp"?fT(x.t*0.1):fT((x.exp||0)*scale), parOf(x)?'<a href="#train/par/'+parOf(x)+'">'+t("train.comm.how")+'</a>':""]),[2,3,5]);
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
  // 解析
  const ops=[...r.ops].sort((a,b)=>(b.fwd.t+b.dgrad.t+b.wgrad.t)-(a.fwd.t+a.dgrad.t+a.wgrad.t));
  const A=[];
  A.push(t("train.analysis.heavy",{ops:ops.slice(0,3).map(o=>esc(shortName(o.name))+" "+V(fT(o.fwd.t+o.dgrad.t+o.wgrad.t))).join(t("train.sep.list")),
    bwd:V(fI((r.tD+r.tW)/Math.max(r.tF,1e-12))+"×"),rc:r.tR?t("train.analysis.heavyRc",{pct:V(pct(r.tR,r.tF))}):""}));
  const commAll=sum(r.comm.filter(x=>!x.perStage),x=>x.t);
  A.push(t("train.analysis.comm",{total:V(fT(commAll/pp)),exp:V(fT(r.exposed/pp)),top:Object.keys(byCls).sort((a,b)=>byCls[b]-byCls[a]).slice(0,2).map(k=>clsName[k]+" "+V(fT(byCls[k]/pp))).join(t("train.sep.list"))}));
  if(r.ep>1) A.push(t("train.analysis.ep",{ep:r.ep,send:["deepep","hybridep","ncclep","distmoe"].includes(c.dispatcher)?t("train.analysis.epDedup",{k:V(fI(r.kEff)),topk:m.moe.k}):t("train.analysis.epCopies",{k:m.moe.k}),
    a2a:V(fT(r.a2aPerLayer)),gemm:V(fT(r.moeT/d.nMoe)),tail:r.a2aPerLayer>r.moeT/d.nMoe?t("train.analysis.epSlow"):""}));
  if(r.pp>1) A.push(t("train.analysis.pp",{pp:r.pp,vpp:r.vpp>1?t("train.analysis.ppVpp",{vpp:r.vpp}):"",bubble:V(pct(r.bubble,r.microsPerStep+r.bubble)),n:r.microsPerStep}));
  A.push(mg?t("train.analysis.dpMg",{time:V(fT(r.tDP)),exp:V(fT(r.tDPexp))}):t("train.analysis.dpTt",{time:V(fT(r.fsdpMB)),exp:V(fT(r.fsdpExposedMB))}));
  A.push(t("train.analysis.opt",{time:V(fT(r.tOpt)),note:esc(r.optNote),shard:fN(r.Pshard),offload:c.offload?t("train.analysis.optOffload",{saved:fB(r.Pshard*12)}):""}));
  if(M.used>M.total) A.push('<span style="color:var(--warn)">'+t("train.analysis.memOver",{over:fB(M.used-M.total),
    top:[[t("train.analysis.memTop.act"),M.act],[t("train.analysis.memTop.params"),M.paramsGPU],[t("train.analysis.memTop.opt"),M.states],[t("train.analysis.memTop.logits"),M.logits]].sort((a,b)=>b[1]-a[1]).slice(0,2).map(x=>x[0]+" "+fB(x[1])).join(t("train.sep.enum"))})+'</span>');
  if(m.lin.n>0) A.push(t("train.analysis.lin",{kind:m.lin.kind,fwd:V(fT(sum(r.ops.filter(o=>o.key==="lcore"),o=>o.fwd.t))),bwd:V(fT(sum(r.ops.filter(o=>o.key==="lcore"),o=>o.dgrad.t)))}));
  $("t-analysis").innerHTML=ul(A);
}
