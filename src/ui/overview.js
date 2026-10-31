/* ================= 总览：推理一步和训练一步并排（照搬两边的当前配置） ================= */
const OGRP = { attn:t("overview.grp.attn"), lin:t("overview.grp.lin"), moe:t("overview.grp.moe"), lm:t("overview.grp.lm"), misc:t("overview.grp.misc") };
function renderOverview(){
  const {m,hw,topo}=ctx(), ci=S.inf, ct=S.trn;
  syncInfRack(); syncTrnRack();      // 让预设和引擎相关的取值先归一（参数架本身在这一页不显示）
  const ri=infRun({phase:"decode"},"decode"), rt=trnRun(), e=m.moe;
  // 推理：按类别归并一次 decode
  const gi={}, gt={}; Object.keys(OGRP).forEach(k=>{ gi[k]=0; gt[k]=0; });
  for(const o of ri.ops) if(o.overlap!=="moe") gi[o.group]+=o.t;
  const tbo=sum(ri.comm.filter(x=>x.overlap==="tbo"),x=>x.t);
  const commI=sum(ri.comm.filter(x=>!x.overlap),x=>x.t)+Math.max(0,tbo-ri.tComp*0.5)+(ri.exposedOverlap||0);
  const specI=(ri.draftT||0)+(ri.commitT||0), otherI=Math.max(0,ri.stepT-sum(Object.values(gi))-commI-specI);
  // 训练：每个 PP stage 的一个 micro-batch
  for(const o of rt.ops) gt[o.group]+=(o.fwd.t+o.dgrad.t+o.wgrad.t)/rt.pp;
  const recT=rt.tR/rt.pp, commT=Math.max(0,rt.tMB-sum(Object.values(gt))-recT);
  const ratio=rt.tokPerGpu/ri.tokPerGpu;     // 生成一个 token 的卡秒 ÷ 训练一个 token 的卡秒
  $("o-lede").innerHTML=t("overview.lede",{model:esc(m.short),hw:hw.name,topo:esc(topo.name)});
  kpis($("o-kpis"),[
    [t("overview.kpi.tpot"),fT(ri.tpot),t("overview.kpi.tpotSub",{engine:ENGN[ci.engine],n:ri.Breq})],
    [t("overview.kpi.infThr"),fN(ri.tokPerGpu)+" tok/s/GPU",t("overview.kpi.infThrFoot")],
    [t("overview.kpi.trnStep"),fT(rt.stepT),t("overview.kpi.trnStepSub",{fw:ct.fw==="megatron"?"Megatron-Core":"torchtitan",tokens:fN(rt.tokens)})],
    [t("overview.kpi.trnThr"),fN(rt.tokPerGpu)+" tok/s/GPU","MFU "+(rt.mfu*100).toFixed(1)+"%"],
    [t("overview.kpi.ratio"),fX(ratio),t("overview.kpi.ratioSub")]]);
  const lamI=e.E?ri.Tg*e.k/e.E:0, lamT=e.E?ct.tok*rt.ep*e.k/e.E:0;
  const memI=sum(ri.ops.filter(o=>o.bound==="memory"),o=>o.t)/ri.stepT;
  const passes=rt.ops.flatMap(o=>[o.fwd,o.dgrad,o.wgrad]), memT=sum(passes.filter(p=>p.bound==="memory"),p=>p.t)/Math.max(1e-30,sum(passes,p=>p.t));
  const tOf=o=>o.fwd.t+o.dgrad.t+o.wgrad.t, slowI=[...ri.ops].sort((a,b)=>b.t-a.t)[0], slowT=[...rt.ops].sort((a,b)=>tOf(b)-tOf(a))[0], nm=s=>esc(shortName(s));
  const usedI=ri.mem.weights+Math.min(ri.Breq,Math.max(0,ri.mem.Bmax))*ri.mem.perSeq+ci.reserveGB*1e9;
  table($("o-cmp"),["",t("overview.cmp.head.inf"),t("overview.cmp.head.trn")],[
    [t("overview.cmp.par.k"),t("overview.cmp.par.inf",{engine:esc(ENGN[ci.engine]),n:ci.N,tp:+ri.tpA.toFixed(2),dp:ri.dpA,moe:esc(MOENAME[ri.moePar])}),
      t("overview.cmp.par.trn",{fw:ct.fw==="megatron"?"Megatron-Core":"torchtitan",tp:rt.tp,cp:rt.cp,pp:rt.pp,dp:+rt.dp.toFixed(2),ep:rt.ep})],
    [t("overview.cmp.prec.k"),t("overview.cmp.prec.inf",{w:ci.wDt.toUpperCase(),d:ci.dDt.toUpperCase(),kv:ci.kvDt.toUpperCase()}),esc((Train.PREC[ct.prec]||Train.PREC.bf16).name)],
    [t("overview.cmp.rows.k"),ri.graph?t("overview.cmp.rows.infGraph",{rows:fInt(ri.rowsG),n:ri.Breq,w:ri.w}):t("overview.cmp.rows.inf",{rows:fInt(ri.rowsG),n:ri.Breq,w:ri.w}),t("overview.cmp.rows.trn",{rows:fInt(ct.tok)})],
    ...(e.E?[[t("overview.cmp.perExpert.k"),fI(lamI),fI(lamT)]]:[]),
    [t("overview.cmp.memBound.k"),pct(memI,1),pct(memT,1)],
    [t("overview.cmp.slowest.k"),nm(slowI.name)+" "+fT(slowI.t),nm(slowT.name)+" "+fT(tOf(slowT))],
    [t("overview.cmp.comm.k"),t("overview.cmp.comm.inf",{pct:pct(commI,ri.stepT)}),t("overview.cmp.comm.trn",{pct:pct(commT,rt.tMB)})],
    [t("overview.cmp.mem.k"),fB(usedI)+" / "+hw.mem+" GB",fB(rt.mem.used)+" / "+hw.mem+" GB"],
    [t("overview.cmp.thr.k"),fN(ri.tokPerGpu)+" tok/s/GPU",fN(rt.tokPerGpu)+" tok/s/GPU"]]);
  stackBar($("o-ibar"),$("o-ibarlist"),[...Object.keys(OGRP).map(k=>({label:OGRP[k],v:gi[k],color:GROUPC[k]})),
    {label:t("overview.bar.comm"),v:commI,color:css("c-a2a"),tag:pill("comm")},{label:t("overview.bar.spec"),v:specI,color:css("c-cp")},
    {label:t("overview.bar.other"),v:otherI,color:css("c-free"),tag:pill("launch")}],ri.stepT,fT);
  stackBar($("o-tbar"),$("o-tbarlist"),[...Object.keys(OGRP).map(k=>({label:k==="lm"?t("overview.grp.lmTrain"):OGRP[k],v:gt[k],color:GROUPC[k]})),
    {label:t("overview.bar.recompute"),v:recT,color:css("c-cp")},{label:t("overview.bar.comm"),v:commT,color:css("c-a2a"),tag:pill("comm")}],rt.tMB,fT);
  const rows=ri.rowsG>0?ct.tok/ri.rowsG:0;
  $("o-notes").innerHTML=ul([
    t("overview.notes.rows",{ratio:V(fI(rows)+"×"),rowsI:fInt(ri.rowsG),rowsT:fInt(ct.tok),perExpert:e.E?t("overview.notes.rowsPerExpert",{inf:fI(lamI),trn:fI(lamT)}):"",memI:pct(memI,1),memT:pct(memT,1)}),
    t("overview.notes.par"),
    t("overview.notes.rl",{ratio:V(fX(ratio)),tail:ratio>1?t("overview.notes.rlRollout"):t("overview.notes.rlTrain")})]);
}
