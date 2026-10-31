/* ================= 推理 · 投机解码：同一个 Infer.decode，只换 draft 方法、k 和 α ================= */
const SPEC_ORDER = ["mtp","eagle3","dflash","dspark"];
const SPECC = { mtp:"s1", eagle3:"s2", dflash:"s3", dspark:"s4" };
const SPEC_K = { mtp:3, eagle3:4, dflash:7, dspark:7 };          // 各方法常见的 draft 长度
const tauOf = (a,k) => a>=1 ? k+1 : (1-Math.pow(a,k+1))/(1-a);    // 每步平均产出（含 bonus），i.i.d. 接受
function alphaOf(tau,k){ if(k<=0||tau<=1) return 0; if(tau>=k+1) return 1;
  let lo=0, hi=1; for(let i=0;i<60;i++){ const mid=(lo+hi)/2; if(tauOf(mid,k)<tau) lo=mid; else hi=mid; } return (lo+hi)/2; }
/* 左侧没开投机时按 MTP、k = 3、α = 0.8 演示 */
function specNow(){ const c=S.inf, on=(c.specK||0)>0, k=on?c.specK:3;
  return { on, k, method:on?(c.specMethod||"mtp"):"mtp", a:on?alphaOf(1+c.accept,k):0.8 }; }
const specRun = (method,k,a,over) => infRun(Object.assign({phase:"decode",specMethod:method,specK:k,accept:k>0?tauOf(a,k)-1:0},over||{}),"decode");
const SPEC_ENG = [
  ["MTP",[["sglang","v0.5.11",t("spec.eng.mtp.specV2Default")],["sglang","v0.5.13",t("spec.eng.mtp.v1Deprecated")]],[["vllm","v0.14.0",t("spec.eng.mtp.asyncSched")],["vllm","v0.30.0",t("spec.eng.mtp.pp")]],t("spec.eng.trt.yes")],
  ["EAGLE-3",[["sglang","v0.5.5",t("spec.eng.eagle3.deepseek")],["sglang","v0.5.7",t("spec.eng.eagle3.specBundle")],["sglang","v0.5.12",t("spec.eng.eagle3.swaMla")]],[["vllm","v0.12.0",t("spec.eng.eagle3.multiStepGraph")],["vllm","v0.17.0","MRV2 + CUDA graph"]],t("spec.eng.trt.yes")],
  ["DFlash / DFlash2",[["sglang","v0.5.11",t("spec.eng.dflash.sglangAdded")],["lmsys","2026-10-16-beam",t("spec.eng.dflash.beam")]],[["vllm","v0.23.0","causal DFlash"],["vllm","v0.28.0","DFlash2"],["vllm","v0.31.0","DFlash + async scheduling"]],"—"],
  ["DSpark",[["sglang","v0.5.16",t("spec.eng.dspark.sglangAdded")],["sglang","v0.5.20",t("spec.eng.dspark.pdCp")]],[["vllm","v0.25.0",t("spec.eng.added")],["vllm","v0.28.0",t("spec.eng.dspark.confSched")]],"—"],
  [t("spec.eng.ngram.name"),[],[["vllm","v0.18.0",t("spec.eng.ngram.gpu")],["vllm","v0.32.0",t("spec.eng.ngram.mrv2")]],t("spec.eng.trt.ngram")],
  [t("spec.eng.draftModel.name"),[],[["vllm","v0.15.0",t("spec.eng.added")],["vllm","v0.31.0",t("spec.eng.draftModel.mrv2")]],t("spec.eng.trt.draftModel")],
  [t("spec.eng.adaptive.name"),[["sglang","v0.5.12",t("spec.eng.adaptive.sglang")]],[["vllm","v0.24.0","Dynamic SD"]],"—"]];
const TRTLLM_SPEC = "https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html";

function renderSpec(){
  const {m,hw}=ctx(), c=S.inf, sn=specNow(), {k,a,method}=sn, P=Infer.ENGINES[c.engine], capped=isFinite(P.specCap), nm=Infer.SPEC[method].name;
  const r=specRun(method,k,a), base=specRun(method,0,a,{B:r.Breq}), tau=tauOf(a,k), verify=r.stepT-r.draftT-r.commitT;
  const capW=Infer.specCapWarn(P), warn=r.warn.filter(w=>Msg.id(w)!==Msg.id(capW));      // 并发上限在正文里交代
  if(r.mem.avail<=0) warn.push(t("spec.warn.noFit",{w:fB(r.mem.weights)}));
  else if(r.Breq>r.mem.Bmax) warn.push(t("spec.warn.bOver",{b:r.Breq,bmax:r.mem.Bmax,n:fInt(c.n)}));
  setWarn(warn);
  $("s-lede").innerHTML=(sn.on?"":t("spec.lede.demo"))+
    t("spec.lede.setup",{model:esc(m.short),hw:hw.name,engine:ENGN[c.engine],b:r.Breq,capNote:r.Breq<c.B?t("spec.lede.capNote",{set:c.B,engine:ENGN[c.engine],cap:P.specCap}):"",ctx:fInt(c.n)})+
    t("spec.lede.method",{method:nm,k,a:V(a.toFixed(2)),tau:V(tau.toFixed(2))})+
    t("spec.lede.step",{w:k+1,base:fT(base.stepT)});
  const su=base.tpot/r.tpot, st=r.tokPerGpu/base.tokPerGpu;
  kpis($("s-kpis"),[
    ["TPOT",fT(r.tpot),t("spec.kpi.tpotFoot",{off:fT(base.tpot)})],
    [t("spec.kpi.speed"),fX(su),t("spec.kpi.speedFoot"),su<1],
    [t("spec.kpi.thr"),fN(r.tokPerGpu)+" tok/s/GPU",t("spec.kpi.thrFoot",{x:fX(st)}),st<1],
    [t("spec.kpi.verify"),fT(verify),t("spec.kpi.verifyFoot",{x:fX(verify/base.stepT)})],
    [t("spec.kpi.fit"),t("spec.kpi.fitVal",{n:fInt(r.mem.Bmax)}),r.mem.wDraft>0?t("spec.kpi.fitFootDraft",{n:fInt(base.mem.Bmax),w:fB(r.mem.wDraft)}):r.mem.wMtp>0?t("spec.kpi.fitFootMtp",{n:fInt(base.mem.Bmax),w:fB(r.mem.wMtp)}):t("spec.kpi.fitFoot",{n:fInt(base.mem.Bmax)}),r.mem.Bmax<r.Breq]]);
  stackBar($("s-bar"),$("s-barlist"),[
    {label:t("spec.bar.verify",{w:k+1}),v:verify,color:css("c-attn")},
    {label:draftLabel({specMethod:method},k),v:r.draftT,color:css("c-cp")},
    {label:t("spec.bar.commit"),v:r.commitT,color:css("c-lin")}],r.stepT,fT);
  // τ 随 k
  const ks=[...Array(16).keys()], AS=[0.6,0.7,0.8,0.9];
  plot($("s-tau"),{h:320,x0:0,x1:15,y0:1,y1:Math.max(10,tau*1.1),fx:v=>String(v),fy:v=>String(+v.toPrecision(3)),xl:t("spec.axis.k"),yl:t("spec.tau.yl"),
    series:AS.map((x,i)=>({pts:ks.map(kk=>[kk,tauOf(x,kk)]),color:css("s"+(i+1)),w:Math.abs(x-a)<0.05?2.8:1.6,label:"α = "+x})),
    points:[{x:k,y:tau,color:css("ink"),r:6,label:t("spec.tau.cur",{a:a.toFixed(2)})}]});
  $("s-tau-legend").innerHTML=AS.map((x,i)=>'<span style="color:'+css("s"+(i+1))+'"><i></i>α = '+x+'</span>').join("")+
    '<span>'+t("spec.tau.legendCap",{n:(tauOf(0.9,15)-tauOf(0.9,7)).toFixed(1)})+'</span>';
  // 同一个 α，四种方法：左图扫 k，右图扫并发
  const one=[{name:t("spec.vs.off"),short:t("spec.vs.offShort"),col:"muted"}];
  const serK=SPEC_ORDER.map(mt=>({name:Infer.SPEC[mt].name,short:Infer.SPEC[mt].name,col:SPECC[mt],pts:[1,2,3,4,5,6,7,8,10,12,15].map(kk=>[kk,base.tpot/specRun(mt,kk,a).tpot])}));
  lineChart("s-vsk",{series:serK.concat([Object.assign({pts:[[1,1],[15,1]]},one[0])]),xLab:t("spec.axis.k"),yLab:t("spec.vs.ylSpeed"),xFmt:v=>String(v),yFmt:v=>v.toFixed(2)+"×",yMin:0,vlines:[{x:k,label:t("spec.vs.curK")}]});
  const Bs=[1,2,4,8,16,24,32,48,64,96,128,192,256,384,512], b0={};
  const b0At=B=>b0[B]||(b0[B]=specRun(method,0,a,{B}));
  const serB=SPEC_ORDER.map(mt=>{ const kk=SPEC_K[mt], pts=[];
    for(const B of Bs){ const x=specRun(mt,kk,a,{B}); if(x.Breq<B||B>x.mem.Bmax) break; pts.push([B,x.tokPerGpu/b0At(B).tokPerGpu]); }
    return {name:t("spec.vs.nameK",{name:Infer.SPEC[mt].name,k:kk}),short:Infer.SPEC[mt].name,col:SPECC[mt],pts}; });
  const xMax=Math.max(2,...serB.map(s=>s.pts.length?s.pts[s.pts.length-1][0]:1));
  lineChart("s-vsb",{series:serB.concat([Object.assign({pts:[[1,1],[xMax,1]]},one[0])]),xLog:true,xBase:2,xLab:t("spec.vs.xlB"),yLab:t("spec.vs.ylThr"),xFmt:v=>fInt(v),yFmt:v=>v.toFixed(2)+"×",yMin:0,vlines:[{x:r.Breq,label:t("spec.vs.curB")}]});
  // 右图的走势按当前方法那条线描述：dense 一路降；MoE 小并发时多激活 expert，先降后升
  const pts=serB[SPEC_ORDER.indexOf(method)].pts, at=p=>t("spec.trend.at",{b:p[0],x:fX(p[1])}), lo=pts[0], hi=pts[pts.length-1], best=pts.length?pts.reduce((b,p)=>p[1]>b[1]?p:b):null;
  let iMin=-1, iMax=-1;          // 第一个谷，和谷后的最高点（高出 5% 才算回升）
  for(let i=1;i<pts.length-1&&iMin<0;i++) if(pts[i][1]<=pts[i-1][1]&&pts[i][1]<=pts[i+1][1]){
    const j=pts.reduce((b,p,jj)=>jj>i&&p[1]>pts[b][1]?jj:b,i); if(pts[j][1]>pts[i][1]*1.05){ iMin=i; iMax=j; } }
  const dip=!!(m.moe&&m.moe.E) && iMin>0;
  const trend=!pts.length?"":dip
    ? t("spec.trend.moe",{method:nm,lo:at(lo),min:at(pts[iMin]),max:at(pts[iMax]),tail:iMax===pts.length-1?t("spec.trend.moeRising"):hi[1]<pts[iMax][1]*0.95?t("spec.trend.moeFalling",{hi:at(hi)}):t("spec.trend.moeFlat")})
    : t("spec.trend.dense",{drop:pts.length>1&&hi[1]<lo[1]?t("spec.trend.denseDrop",{method:nm,lo:at(lo),hi:at(hi)}):""});
  $("s-cost-lede").innerHTML=t("spec.cost.lede",{a:V(a.toFixed(2)),b:r.Breq})+t("spec.cost.real")+trend+
    (capped?t("spec.cost.capped",{engine:ENGN[c.engine],cap:P.specCap}):t("spec.cost.uncapped"));
  legendInto("s-vs-legend",SPEC_ORDER.map(mt=>({name:Infer.SPEC[mt].name,col:SPECC[mt],line:true})).concat([{name:t("spec.vs.offShort"),col:"muted",line:true}]));
  table($("s-ledger"),[t("spec.ledger.method",{a:a.toFixed(2),b:r.Breq}),"k",t("spec.ledger.width"),t("spec.ledger.draft"),t("spec.ledger.verify"),t("spec.ledger.commit"),t("spec.ledger.step"),"τ","TPOT",t("spec.ledger.speed"),"tok/s/GPU",t("spec.ledger.draftW"),t("spec.ledger.bmax")],
    SPEC_ORDER.map(mt=>{ const kk=SPEC_K[mt], x=specRun(mt,kk,a);
      return {cls:mt===method?"hl":"",cells:[sw(css(SPECC[mt]))+Infer.SPEC[mt].name,kk,kk+1,fT(x.draftT),fT(x.stepT-x.draftT-x.commitT),x.commitT?fT(x.commitT):"—",fT(x.stepT),tauOf(a,kk).toFixed(2),fT(x.tpot),fX(base.tpot/x.tpot),fN(x.tokPerGpu),fB(x.mem.wDraft||x.mem.wMtp||0),fInt(x.mem.Bmax)]}; }),[1,2,3,4,5,6,7,8,9,10,11,12]);
  // 引擎支持、公开数字、RL
  const cell=list=>list.length?list.map(([p,v,note])=>t("spec.eng.cell",{link:relLink(p,v),note:esc(note)})).join("<br>"):"—";
  table($("s-eng"),[t("spec.eng.head.method"),"SGLang","vLLM",t("spec.eng.head.trt",{doc:aOut(TRTLLM_SPEC,t("spec.eng.head.doc"))})],SPEC_ENG.map(([n,sg,vl,tr])=>[esc(n),cell(sg),cell(vl),esc(tr)]));
  table($("s-pub"),[t("spec.pub.head.date"),t("spec.pub.head.src"),t("spec.pub.head.method"),t("spec.pub.head.model"),t("spec.pub.head.setting"),t("spec.pub.head.value")],DB.set("spec_public").slice().sort((x,y)=>y.date.localeCompare(x.date))
    .map(p=>['<span class="v">'+esc(p.date)+'</span>','<span class="nw">'+aOut(p.url,p.src)+'</span>',esc(p.method),esc(p.model)+(p.hw&&p.hw!=="—"?"<br>"+'<span class="tag">'+esc(p.hw)+'</span>':""),esc(p.setting),esc(p.value)]));
  // RL：长尾看右图左端；批量阶段和不开投机、跑满并发比
  const tailNote=!lo?"":t("spec.rl.tail",{method:nm,at:at(lo),extra:lo===best?t("spec.rl.tailBest"):dip?(lo[1]>1.02?t("spec.rl.tailMoeGain"):t("spec.rl.tailMoe")):""});
  const Bf=Math.min(c.B,Math.floor(base.mem.Bmax)), full=Bf>r.Breq?specRun(method,0,a,{B:Bf}):null, rf=full?r.tokPerGpu/full.tokPerGpu:0;
  const batch=full
    ? t("spec.rl.batch",{engine:ENGN[c.engine],b:r.Breq,bf:Bf,thr:fN(r.tokPerGpu),thrOff:fN(full.tokPerGpu),x:fX(rf),verdict:rf<0.97?t("spec.rl.batchLoss"):rf<1.05?t("spec.rl.batchEven"):""})
    : t("spec.rl.batchGeneric");
  $("s-rl").innerHTML=ul([
    t("spec.rl.longTail",{tail:tailNote,swe:relLink("lmsys","2026-10-22-miles-mi355x-agentic-rl",t("spec.rl.sweLink")),batch,nemotron:relLink("lmsys","2026-08-11-nemotron-3-5-lightning","Nemotron 3.5 Lightning"),sglang:relLink("sglang","v0.5.12"),vllm:relLink("vllm","v0.24.0")}),
    t("spec.rl.followPolicy",{miles:relLink("lmsys","2025-11-19-miles","Miles"),slime:relLink("slime","v0.2.0"),milesSync:relLink("miles","v0.1.3"),sglangUpdate:relLink("sglang","v0.5.6"),sglangSession:relLink("sglang","v0.5.21"),vllmUpdate:relLink("vllm","v0.26.0")}),
    t("spec.rl.kernelShape"),
    t("spec.rl.memory")]);
}
