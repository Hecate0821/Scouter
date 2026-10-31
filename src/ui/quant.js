/* ================= 低精度：推理、训练各一节；格式放附录，RL 组合放 RL 页 ================= */
const QC = { bf16:css("c-lm"), fp8:css("c-attn"), mxfp8:css("c-lin"), w4a16:css("c-ar"), w4a8:css("c-cp"), mxfp4:css("c-moe"), nvfp4:css("c-a2a") };
const QE2E = { bf16:{wDt:"bf16",eAct:"",dDt:"bf16"}, fp8:{wDt:"fp8",eAct:"",dDt:"fp8"}, mxfp8:{wDt:"mxfp8",eAct:"",dDt:"mxfp8"},
  w4a16:{wDt:"mxfp4",eAct:"bf16",dDt:"bf16"}, w4a8:{wDt:"mxfp4",eAct:"mxfp8",dDt:"mxfp8"}, mxfp4:{wDt:"mxfp4",eAct:"mxfp4",dDt:"bf16"}, nvfp4:{wDt:"nvfp4",eAct:"nvfp4",dDt:"bf16"} };
const QTRN = ["bf16","fp8","mxfp8","nvfp4"];
const DTN = { bf16:"BF16", fp8:t("common.dt.fp8Block"), mxfp8:"MXFP8", mxfp4:"MXFP4", nvfp4:"NVFP4" };
const qOpt = (over) => Object.assign({ fuse:S.q.fuse, align:1, floor:false, BM:S.q.BM, deqRate:S.q.deqRate }, over||{});   // 单个 GEMM 按纯 roofline，不计 kernel 最短时间
const qTopo = h => /^GB/.test(h) ? "nvl64" : (S.topo==="nvl64" ? "hgx8" : S.topo);
function qShape(){ const m=S.model, e=m.moe, De=e.Dlat||m.D, F=e.F||m.dense.F;   // dense 模型：gate_up / down 取 dense FFN
  const sh={ up:[2*F,De], down:[De,F], shared:[2*(e.Fs||m.dense.F||e.F),m.D] }[S.q.shape]; if(sh){ S.q.N=sh[0]; S.q.K=sh[1]; } }
const fX = x => (x>=10?x.toFixed(1):x.toFixed(2))+"×";
const bits = d => (Common.dtB(d)*8).toFixed(d==="fp8"?2:2);

function qInferE2E(hw, key){ const {m,g}=ctx(), topo=TOPO[qTopo(S.q.hw)];
  const c=Object.assign({},S.inf,{phase:"decode",qSep:S.q.fuse==="separate"},QE2E[key]);
  if(!m.moe.E){ c.dDt=c.wDt; c.dAct=c.eAct; }     // dense 模型没有 expert：方案作用于全部线性层
  return Infer.decode(m,hw,topo,c,g); }
function qTrainE2E(hw, prec){ const {m,g}=ctx(), topo=TOPO[qTopo(S.q.hw)];
  return Train.run(m,hw,topo,Object.assign({N:topo.gpus},S.trn,{prec,tokFused:S.q.tokFused}),g); }

/* 每条实测在这一页的公式下算出来的对应数字 */
function qCheck(key, hw){
  const g=S.glob, Q=Quant, o=qOpt();
  if(key==="peaks"){ const B=HW.B200, B3=HW.B300; return t("quant.check.peaks", {bf16:B.bf16/1000, fp8:B.fp8/1000, fp4:B.fp4/1000, fp4B300:B3.fp4/1000}); }
  if(key==="castBW"){ const B=HW.B200; return t("quant.check.castBW", {eta:g.etaM, bw:(B.hbm*g.etaM/1000).toFixed(1), t:fT(Common.memT(16384*16384*(2+1+1/32),B,g))}); }
  if(key==="w4a8"){ const sh=[[4096,6144],[6144,2048]], r=[];
    for(const [N,K] of sh) for(const M of [256,4096]){ const a=Q.gemm(HW.B200,g,"w4a16",M,N,K,o), b=Q.gemm(HW.B200,g,"w4a8",M,N,K,o); r.push(a.t/b.t); }
    const qw={moe:{E:256,k:8,F:512,Dlat:0},D:2048}, sep=qOpt({fuse:"separate",align:"auto"});
    const e2=[64,1024].map(T=>Q.moeLayer(HW.B200,g,qw,"w4a16",T,1,sep).t/Q.moeLayer(HW.B200,g,qw,"w4a8",T,1,sep).t);
    const lo=fX(Math.min(...r)), hi=fX(Math.max(...r));
    return t("quant.check.w4a8", {range:lo===hi?lo:lo+"–"+hi, t64:fX(e2[0]), t1024:fX(e2[1])}); }
  if(key==="fuse"){ const h=HW.GB300, {m}=ctx(), topo=TOPO.nvl64, base=Object.assign({},S.inf,{phase:"decode",N:4,dpA:1,moePar:"tp",wDt:"nvfp4",eAct:"",dDt:"bf16",specK:0});
    const r=[1,16,128].map(B=>{ const a=Infer.decode(m,h,topo,Object.assign({},base,{B,qSep:true}),g), b=Infer.decode(m,h,topo,Object.assign({},base,{B,qSep:false}),g); return (a.stepT/b.stepT-1)*100; });
    return t("quant.check.fuse", {r:r.map(x=>x.toFixed(1)+"%").join(" / ")}); }
  if(key==="mem"){ const M=2048,K=2048,N=8192, d=Q.trainGemm(HW.B200,g,"mxfp8",M,N,K,{nMicro:1}), h=Q.trainGemm(HW.B200,g,"mxfp8hp",M,N,K,{nMicro:1}),
      dn=Q.trainGemm(HW.B200,g,"nvfp4",M,N,K,{nMicro:1}), hn=Q.trainGemm(HW.B200,g,"nvfp4hp",M,N,K,{nMicro:1});
    return t("quant.check.mem", {mx:fB(d.extra), mxHp:fB(h.extra), pct:((h.extra/d.extra-1)*100).toFixed(0), nv:fB(dn.extra), nvHp:fB(hn.extra)}); }
  if(key==="trainRecipe"){ const {m}=ctx(), e=m.moe, De=e.Dlat||m.D, F=e.F||m.dense.F, M=8192, tg=k=>Q.trainGemm(HW.B200,g,k,M,2*F,De,{nMicro:4,tokFused:false}).t, b=tg("bf16");
    return t("quant.check.trainRecipe", {part:t(e.E?"quant.check.partExpert":"quant.check.partDense"), mx:fX(b/tg("mxfp8")), hpU:fX(b/tg("nvfp4hp")), hpF:fX(b/Q.trainGemm(HW.B200,g,"nvfp4hp",M,2*F,De,{nMicro:4,tokFused:true}).t)}); }
  if(key==="moeTrain"){ const D=5120, F=8192, M=16384, tg=k=>Q.trainGemm(HW.B200,g,k,M,2*F,D,{nMicro:1}).t+Q.trainGemm(HW.B200,g,k,M,D,F,{nMicro:1}).t;
    return t("quant.check.moeTrain", {mx:fX(tg("bf16")/tg("mxfp8"))}); }
  return "";
}

/* 各节共用：填两个方案选择器、硬件说明 */
function qHead(){
  const q=S.q, hw=HW[q.hw], Q=Quant, SC=Q.SCHEMES;
  fillOpts($("q-scheme"), Q.ORDER.map(k=>[k,t("quant.head.schemeOpt", {short:SC[k].short, name:SC[k].name})]), q.scheme);
  if(!Q.RORDER.includes(q.recipe)) q.recipe="mxfp8";
  fillOpts($("q-recipe"), Q.RORDER.map(k=>[k,Q.RECIPES[k].name]), q.recipe);
  const fb=q.hw!==S.hw; $("q-hw-note").hidden=!fb;
  if(fb) $("q-hw-note").textContent = t("quant.head.hwFallback", {cur:HW[S.hw].name, name:hw.name});
  return { q, hw, g:S.glob, Q, SC };
}
const qMs = (hw,g,N,K,o) => { const Ms={}; Quant.ORDER.forEach(k=>Ms[k]=Quant.crossover(hw,g,k,N,K,o)); return Ms; };
const qLegend = cur => Quant.ORDER.map(k=>'<span style="color:'+QC[k]+'"><i'+(k===cur||k==="bf16"?'':' class="d"')+'></i>'+esc(Quant.SCHEMES[k].short)+'</span>').join("");

/* 推理：一个 GEMM、一层 MoE、放回整个 decode 一步 */
function renderQuantInfer(){
  const {m,d}=ctx(), {q,hw,g,Q,SC}=qHead(), o=qOpt();
  const N=q.N, K=q.K, M=Math.max(1,q.M), cur=q.scheme, G={};
  Q.ORDER.forEach(k=>G[k]=Q.gemm(hw,g,k,M,N,K,o));
  const Ms=qMs(hw,g,N,K,o), x=G[cur], b=G.bf16;
  $("q-lede").innerHTML=t("quant.infer.lede.gemm", {M:fInt(M), N:fInt(N), K:fInt(K), scheme:esc(SC[cur].name), hw:esc(hw.name),
    bound:x.bound==="memory" ? t("quant.infer.lede.memory", {pct:pct(x.wB,x.bytes)}) : x.bound==="compute" ? t("quant.infer.lede.compute", {pk:SC[cur].pk.toUpperCase()}) : t("quant.infer.lede.launch")});
  kpis($("q-kpis"),[
    [t("quant.infer.kpi.time"),fT(x.t),x.tQ>0?t("quant.infer.kpi.timeQ", {t:fT(x.tQ)}):x.tDeq>0?t("quant.infer.kpi.timeDeq", {t:fT(x.tDeq)}):""],
    [t("quant.infer.kpi.vsBf16"),fX(b.t/x.t),"BF16 "+fT(b.t),b.t<x.t],
    [t("quant.infer.kpi.bound"),t("common.pill."+x.bound),"I = "+fI(x.I)+" FLOP/B"],
    [t("quant.infer.kpi.knee"),isFinite(Ms[cur])?fInt(Ms[cur]):"∞","BF16 "+fInt(Ms.bf16)],
    [t("quant.infer.kpi.wBytes"),fB(x.wB),t("quant.infer.kpi.wScale", {b:fB(x.wScale)})],
    [t("quant.infer.kpi.aBytes"),fB(x.aB+x.qB),x.qB?t("quant.infer.kpi.aQ", {b:fB(x.qB)}):""]]);
  // 扫 M
  const xs=logspace(1,65536,120), series=[], vl=[{x:M,color:css("ink"),label:t("quant.infer.sweep.curM", {M:fInt(M)})}];
  for(const k of Q.ORDER){ series.push({pts:xs.map(v=>[v, Q.gemm(hw,g,"bf16",v,N,K,o).t/Q.gemm(hw,g,k,v,N,K,o).t]),color:QC[k],w:k===cur?3:1.5,dash:k===cur||k==="bf16"?null:"6 4",label:k===cur?SC[k].short:null}); }
  if(isFinite(Ms[cur])) vl.push({x:Ms[cur],color:QC[cur],label:"M* "+fInt(Ms[cur]),dy:14});
  plot($("q-sweep"),{x0:1,x1:65536,y0:0.25,y1:hw.fp4/hw.bf16*1.6,logx:true,logy:true,fx:fxPow,fy:v=>+v.toPrecision(2)+"×",xl:t("quant.infer.sweep.x"),yl:t("quant.infer.sweep.y"),series,vlines:vl});
  const pks=[{v:peakOf(hw,"bf16"),n:"BF16"},{v:peakOf(hw,"fp8"),n:"FP8"},{v:peakOf(hw,"fp4"),n:"FP4"}];
  plot($("q-roof"),{x0:1,x1:30000,y0:1e11,y1:pks[2].v*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:t("quant.infer.roof.x"),yl:t("quant.infer.roof.y"),
    series:roofSeries(hw,g,pks,1,30000),points:Q.ORDER.map(k=>({x:G[k].I,y:G[k].flReal/G[k].t,color:QC[k],r:k===cur?6:4,label:SC[k].short}))});
  $("q-sweep-legend").innerHTML=qLegend(cur)+'<span>'+t("quant.infer.sweep.roofs")+'</span>';
  table($("q-gemm"),[t("quant.infer.gemm.head.scheme"),t("quant.infer.gemm.head.bits"),t("quant.infer.gemm.head.w"),t("quant.infer.gemm.head.wScale"),t("quant.infer.gemm.head.a"),t("quant.infer.gemm.head.compute"),t("quant.infer.gemm.head.memory"),t("quant.infer.gemm.head.qKernel"),t("quant.infer.gemm.head.deq"),t("quant.infer.gemm.head.total"),t("quant.infer.gemm.head.bound"),t("quant.infer.gemm.head.vsBf16"),"M*"],
    Q.ORDER.map(k=>{ const r=G[k]; return {cls:k===cur?"hl":"",cells:[sw(QC[k])+esc(SC[k].short),bits(SC[k].w)+" / "+bits(SC[k].a),fB(r.wB),fB(r.wScale),fB(r.aB),fT(r.tc),fT(r.tm),r.tQ?fT(r.tQ):"—",r.tDeq?fT(r.tDeq):"—",fT(r.t),pill(r.bound),fX(b.t/r.t),isFinite(Ms[k])?fInt(Ms[k]):"∞"]}; }),[2,3,4,5,6,7,8,9,11,12]);
  // MoE
  const e=m.moe, De=e.Dlat||m.D, T=Math.max(1,q.T), ep=Math.max(1,q.ep), oM=qOpt({align:q.align,floor:true}), L={};
  const hasMoE=e.E>0; let lam=0, Lc={a:"—",rows:0,rowsPad:0,bound:"memory"};
  ["q-moe","q-moe-legend","q-moe-tbl","q-align"].forEach(id=>{ const el=$(id); (el.closest(".fig,.tbl")||el).hidden=!hasMoE; });
  if(!hasMoE) $("q-moe-lede").innerHTML=t("quant.infer.moe.dense", {model:esc(m.short)});
  else {
  Q.ORDER.forEach(k=>L[k]=Q.moeLayer(hw,g,m,k,T,ep,oM));
  lam=T*e.k/e.E; Lc=L[cur];
  $("q-moe-lede").innerHTML=t("quant.infer.moe.lede", {model:esc(m.short), E:e.E, k:e.k, De, F:e.F, T:fInt(T), lam:V(fI(lam)),
    regime:lam<1 ? t("quant.infer.moe.sparse") : lam<64 ? t("quant.infer.moe.belowKnee") : t("quant.infer.moe.aboveKnee")});
  const Ts=logspace(1,262144,90), ms=[];
  for(const k of Q.ORDER) ms.push({pts:Ts.map(t=>[t, Q.moeLayer(hw,g,m,"bf16",t,ep,oM).t/Q.moeLayer(hw,g,m,k,t,ep,oM).t]),color:QC[k],w:k===cur?3:1.5,dash:k===cur||k==="bf16"?null:"6 4",label:k===cur?SC[k].short:null});
  const tAt = l => l*e.E/e.k;
  plot($("q-moe"),{x0:1,x1:262144,y0:0.25,y1:hw.fp4/hw.bf16*1.6,logx:true,logy:true,fx:fxPow,fy:v=>+v.toPrecision(2)+"×",xl:t("quant.infer.moe.x"),yl:t("quant.infer.moe.y"),series:ms,
    vlines:[{x:T,color:css("ink"),label:t("quant.infer.moe.curT")},{x:tAt(1),color:css("muted"),label:"λ = 1",dy:14},{x:tAt(Ms.bf16),color:css("muted"),label:t("quant.infer.moe.lamKnee"),dy:28}]});
  $("q-moe-legend").innerHTML=qLegend(cur);
  table($("q-moe-tbl"),[t("quant.infer.moe.head.scheme", {ep, n:fI(e.E/ep)}),t("quant.infer.moe.head.lam"),t("quant.infer.moe.head.align"),t("quant.infer.moe.head.rowsPad"),t("quant.infer.moe.head.waste"),t("quant.infer.moe.head.touched"),t("quant.infer.moe.head.wRead"),t("quant.infer.moe.head.time"),t("quant.infer.moe.head.bound"),t("quant.infer.moe.head.vsBf16")],
    Q.ORDER.map(k=>{ const r=L[k]; return {cls:k===cur?"hl":"",cells:[sw(QC[k])+esc(SC[k].short),fI(r.lam),r.a,fN(r.rowsPad),pct(r.rowsPad-r.rows,r.rowsPad),fI(r.touched)+" / "+fI(r.El),fB(r.wB),fT(r.t),pill(r.bound),fX(L.bf16.t/r.t)]}; }),[1,2,3,4,5,6,7,9]);
  const AL=[[1,t("quant.infer.align.a1")],[8,t("quant.infer.align.a8")],[16,t("quant.infer.align.a16")],[32,t("quant.infer.align.a32")],[64,t("quant.infer.align.a64")],[128,t("quant.infer.align.a128")]];
  table($("q-align"),[t("quant.infer.align.head.to"),t("quant.infer.align.head.rows", {lam:fI(lam)}),t("quant.infer.align.head.waste"),t("quant.infer.align.head.where")],
    AL.map(([a,w])=>{ const p=Q.poisson(lam,a,q.BM); return {cls:String(a)===String(Lc.a)?"hl":"",cells:[a,fN(e.E*p.pad),pct(Math.max(0,p.pad-lam),p.pad),esc(w)]}; }),[0,1,2]);
  }
  // 放回整个 decode 一步
  const IE={}; Q.ORDER.forEach(k=>IE[k]=qInferE2E(hw,k));
  $("q-e2e-inf-lede").innerHTML=t("quant.infer.e2e.lede", {engine:esc(ENGN[S.inf.engine]||S.inf.engine), N:S.inf.N, B:S.inf.B, ctx:fInt(S.inf.n), hw:esc(hw.name),
    tail:m.moe.E ? t("quant.infer.e2e.ledeMoe") : t("quant.infer.e2e.ledeDense")});
  const ib=IE.bf16;
  table($("q-e2e-inf"),[t("quant.infer.e2e.head.scheme"),t("quant.infer.e2e.head.weights"),t("quant.infer.e2e.head.wPerGpu"),t("quant.infer.e2e.head.maxB"),t("quant.infer.e2e.head.step"),t("quant.infer.e2e.head.tokGpu"),t("quant.infer.e2e.head.vsBf16")],
    Q.ORDER.map(k=>{ const r=IE[k], c=QE2E[k], fit=r.mem.avail>0, over=fit&&S.inf.B>r.mem.Bmax;
      return {cls:k===cur?"hl":"",cells:[sw(QC[k])+esc(SC[k].short),DTN[c.wDt]+(c.eAct?t("quant.infer.e2e.act", {dt:DTN[c.eAct]}):"")+" / "+DTN[c.dDt],fB(r.mem.weights),fit?fInt(r.mem.Bmax)+(over?' <span class="tag">'+t("quant.infer.e2e.overB")+'</span>':""):'<span class="tag">'+t("quant.infer.e2e.noFit")+'</span>',fT(r.stepT),fN(r.tokPerGpu),fX(ib.stepT/r.stepT)]}; }),[2,3,4,5,6]);
  // 解析
  const A=[], ec=d.pExpertsAll/d.total;
  const big16=Q.gemm(hw,g,"bf16",16384,N,K,o), big4=Q.gemm(hw,g,"w4a16",16384,N,K,o);
  A.push(t("quant.infer.an.knee", {bf16:V(fInt(Ms.bf16)), w4a16:V(fInt(Ms.w4a16)), w8a8:V(fInt(Ms.mxfp8)), w4a4:V(fInt(Ms.nvfp4)), x:V(fX(big16.t/big4.t)), BM:q.BM}));
  if(!hasMoE) A.push(t("quant.infer.an.dense", {model:esc(m.short), total:V(fN(d.total))}));
  else A.push(t("quant.infer.an.moe", {model:esc(m.short), share:V(pct(d.pExpertsAll,d.total)), why:ec>0.8 ? t("quant.infer.an.moeMost") : t("quant.infer.an.moeSome"),
    active:V(pct(d.nMoe*m.moe.k*d.pExpert,d.active)), lam:V(fI(lam)), a:Lc.a, waste:V(pct(Lc.rowsPad-Lc.rows,Lc.rowsPad)), tail:Lc.bound==="memory" ? t("quant.infer.an.padMemory") : t("quant.infer.an.padCompute")}));
  const qs=Q.gemm(hw,g,cur,M,N,K,qOpt({fuse:"separate"})), qf=Q.gemm(hw,g,cur,M,N,K,qOpt({fuse:"fused"}));
  if(SC[cur].a!=="bf16") A.push(t("quant.infer.an.actQuant", {t:V(fT(qs.tQ)), pct:pct(qs.tQ,qf.t)}));
  const bestI=Q.ORDER.filter(k=>k!=="bf16"&&IE[k].mem.avail>0).sort((a,b)=>IE[a].stepT-IE[b].stepT)[0];
  if(bestI) A.push(t("quant.infer.an.e2e", {best:esc(SC[bestI].short), x:V(fX(ib.stepT/IE[bestI].stepT)), bBf16:V(fInt(ib.mem.Bmax)), bBest:V(fInt(IE[bestI].mem.Bmax))}));
  A.push(t("quant.infer.an.hw"));
  A.push(t("quant.infer.an.rl"));
  $("q-analysis-inf").innerHTML=ul(A);
}

/* 后训练：三遍 GEMM、放回整个 step */
function renderQuantTrain(){
  const {q,hw,g,Q}=qHead(), N=q.N, K=q.K;
  const Mt=Math.max(16,q.Mt), TR={}, to={nMicro:q.nMicro,tokFused:q.tokFused};
  Q.RORDER.forEach(k=>TR[k]=Q.trainGemm(hw,g,k,Mt,N,K,to));
  const ideal=k=>Q.idealSpeedup(hw,Q.RECIPES[k].f,Q.RECIPES[k].b);
  table($("q-train"),[t("quant.train.head.recipe"),"fprop","dgrad","wgrad",t("quant.train.head.qKernels"),t("quant.train.head.total"),t("quant.train.head.vsBf16"),t("quant.train.head.peakOnly"),t("quant.train.head.saved"),t("quant.train.head.who")],
    Q.RORDER.map(k=>{ const r=TR[k]; return {cls:k===q.recipe?"hl":"",cells:[esc(r.R.name),fT(r.fprop),fT(r.dgrad),fT(r.wgrad),r.tq?fT(r.tq):"—",fT(r.t),fX(TR.bf16.t/r.t),fX(ideal(k)),fB(r.extra),esc(r.R.who)]}; }),[1,2,3,4,5,6,7,8]);
  const tr=TR[q.recipe];
  stackBar($("q-tbar"),$("q-tbarlist"),[{label:t("quant.train.bar.fprop", {dt:DTN[tr.R.f]}),v:tr.fprop,color:css("c-attn")},{label:t("quant.train.bar.dgrad", {dt:DTN[tr.R.b]}),v:tr.dgrad,color:css("c-moe")},{label:t("quant.train.bar.wgrad", {dt:DTN[tr.R.b]}),v:tr.wgrad,color:css("c-lin")},
    ...tr.q.map((z,i)=>({label:z[0],v:z[1],color:[css("c-cp"),css("c-a2a"),css("c-ar"),css("c-dp"),css("muted")][i%5]}))],tr.t,fT);
  // 放回整个 step
  const TE={}; QTRN.forEach(k=>TE[k]=qTrainE2E(hw,k));
  $("q-e2e-trn-lede").innerHTML=t("quant.train.e2e.lede", {fw:S.trn.fw==="megatron"?"Megatron-Core":"torchtitan", tp:S.trn.tp, cp:S.trn.cp, ep:S.trn.ep, pp:S.trn.pp, tok:fInt(S.trn.tok), hw:esc(hw.name)});
  const tb=TE.bf16;
  table($("q-e2e-trn"),[t("quant.train.e2e.head.prec"),t("quant.train.e2e.head.step"),t("quant.train.e2e.head.tokGpu"),t("quant.train.e2e.head.vsBf16"),t("quant.train.e2e.head.qW"),t("quant.train.e2e.head.mem")],
    QTRN.map(k=>{ const r=TE[k]; return {cls:k===q.recipe?"hl":"",cells:[esc(Train.PREC[k].name),fT(r.stepT),fN(r.tokPerGpu),fX(tb.stepT/r.stepT),fB(r.mem.qW),fB(r.mem.used)+(r.mem.used>r.mem.total?' <span class="tag">'+t("quant.train.e2e.over")+'</span>':"")]}; }),[1,2,3,4,5]);
  // 解析
  const A=[], best=Q.RORDER.filter(k=>k!=="bf16").sort((a,b)=>TR[a].t-TR[b].t)[0], mx=TR.mxfp8, tt=TR.mxfp8tt, wq=r=>(r.q.find(z=>z[2]==="w")||[0,0])[1];
  A.push(t("quant.train.an.gemm", {M:fInt(Mt), best:esc(Q.RECIPES[best].name), x:V(fX(TR.bf16.t/TR[best].t)), ideal:V(fX(ideal(best))),
    cacheTe:fB(mx.wCache), cacheTt:fB(tt.wCache), nMicro:q.nMicro, dt:fT(wq(mx)-wq(tt))}));
  const bestT=QTRN.filter(k=>k!=="bf16").sort((a,b)=>TE[a].stepT-TE[b].stepT)[0];
  A.push(t("quant.train.an.e2e", {best:esc(Train.PREC[bestT].name), x:V(fX(tb.stepT/TE[bestT].stepT))}));
  A.push(t("quant.train.an.rl"));
  $("q-analysis-trn").innerHTML=ul(A);
}

/* 附录 · 量化基础：格式表 */
function renderFormats(){
  const {m,d}=ctx();
  const binE2=Math.log2(6/0.5), binE4=Math.log2(448/Math.pow(2,-9)), wAll=m.moe.E?d.pExpertsAll:d.total-d.pEmb-d.pHead;
  table($("q-formats"),[t("quant.formats.table.head.fmt"),t("quant.formats.table.head.elem"),t("quant.formats.table.head.block"),t("quant.formats.table.head.blockScale"),t("quant.formats.table.head.second"),
    t("quant.formats.table.head.bits"),t("quant.formats.table.head.range"),t("quant.formats.table.head.cost"),m.moe.E?t("quant.formats.table.head.wMoe"):t("quant.formats.table.head.wDense")],[
    [t("quant.formats.table.tensorFp8.name"),"E4M3",t("quant.formats.table.tensorFp8.block"),"FP32","—","8",t("quant.formats.table.bins", {n:binE4.toFixed(1)}),t("quant.formats.table.tensorFp8.cost"),fB(wAll*1)],
    [t("common.dt.blockFp8"),"E4M3",t("quant.formats.table.blockFp8.block"),t("quant.formats.table.blockFp8.scale"),"—",t("quant.formats.table.blockFp8.bits", {w:bits("fp8"), a:bits("fp8a")}),t("quant.formats.table.bins", {n:binE4.toFixed(1)}),t("quant.formats.table.blockFp8.cost"),fB(wAll*Common.dtB("fp8"))],
    ["MXFP8","E4M3",t("quant.formats.table.mxfp8.block"),"E8M0","—",bits("mxfp8"),t("quant.formats.table.bins", {n:binE4.toFixed(1)}),t("quant.formats.table.mxfp8.cost"),fB(wAll*Common.dtB("mxfp8"))],
    ["MXFP4","E2M1","1×32","E8M0","—",bits("mxfp4"),t("quant.formats.table.bins", {n:binE2.toFixed(1)}),t("quant.formats.table.mxfp4.cost"),fB(wAll*Common.dtB("mxfp4"))],
    ["NVFP4","E2M1",t("quant.formats.table.nvfp4.block"),"E4M3",t("quant.formats.table.nvfp4.second"),bits("nvfp4"),t("quant.formats.table.bins", {n:binE2.toFixed(1)}),t("quant.formats.table.nvfp4.cost"),fB(wAll*Common.dtB("nvfp4"))]],[5,8]);
}

/* RL：低精度 rollout 和训练组合成一步（Amdahl） */
function qRL(){
  const {q,hw,Q,SC}=qHead(), f=Math.min(1,Math.max(0,q.roll)), sp=(sr,st)=>1/(f/sr+(1-f)/st);
  const IE={}, TE={}; ["bf16","mxfp8","nvfp4","w4a16"].forEach(k=>IE[k]=qInferE2E(hw,k)); ["bf16","mxfp8","mxfp8hp","nvfp4","nvfp4hp"].forEach(k=>TE[k]=qTrainE2E(hw,k));
  const RLC=[[t("quant.rl.combo.bf16"),"bf16","bf16",t("quant.rl.match.yes")],[t("quant.rl.combo.mxfp8"),"mxfp8","mxfp8",t("quant.rl.match.yes")],[t("quant.rl.combo.mxfp8hp"),"mxfp8","mxfp8hp",t("quant.rl.match.yes")],
    [t("quant.rl.combo.nvfp4hp"),"nvfp4","nvfp4hp",t("quant.rl.match.nvfp4hp")],[t("quant.rl.combo.nvfp4te"),"nvfp4","nvfp4",t("quant.rl.match.nvfp4te")],
    [t("quant.rl.combo.w4a16"),"w4a16","bf16",t("quant.rl.match.w4a16")]];
  table($("q-rl"),[t("quant.rl.head.combo"),t("quant.rl.head.rollout"),t("quant.rl.head.train"),t("quant.rl.head.step", {f}),t("quant.rl.head.match")],
    RLC.map(([n,a,t,c])=>{ const sr=IE.bf16.stepT/IE[a].stepT, st=TE.bf16.stepT/TE[t].stepT; return [esc(n),fX(sr),fX(st),fX(sp(sr,st)),esc(c)]; }),[1,2,3]);
  return { hw, f, IE, TE, SC };
}
