/* ================= 推理 · 一步 decode / prefill（参数架三节共用） ================= */
function infRun(over, phase){ const {m,hw,topo,g}=ctx(); const c=Object.assign({},S.inf,over||{}); return (phase||c.phase)==="decode" ? Infer.decode(m,hw,topo,c,g) : Infer.prefill(m,hw,topo,c,g); }
const infEngines = () => ["sglang","vllm","tokenspeed"].concat(S.inf.engine==="titan"?["titan"]:[]);
const asEng = e => ({engine:e, moePar:compatMoe(e,S.inf.moePar)});
/* TBO 什么时候划算（部署与并行）：计算核心 Infer.tboSweep 给开 / 关两条线，这里只画 */
function renderInferTbo(){
  const {m,hw,topo,g}=ctx(), c=S.inf, s=Infer.tboSweep(m,hw,topo,c,g);
  $("i-tbo-fig").hidden=!s; $("i-tbo-legend").hidden=!s;
  if(!s){ $("i-tbo-lede").innerHTML=t("infer.deploy.tbo.none"); $("i-tbo-read").innerHTML=""; return; }
  $("i-tbo-lede").innerHTML=t("infer.deploy.tbo.lede",{eng:ENGN[c.engine],model:esc(m.short),hw:hw.name,n:c.N,moe:esc(MOENAME[s.moePar]||s.moePar),dp:s.dpA})+(s.swapped?" "+t("infer.deploy.tbo.swapped"):"")
    +(s.invalid.length?" "+t("infer.deploy.tbo.invalid",{why:esc(s.invalid.map(String).join(" / "))}):"");
  const pt=(r,on)=>[r.B,on?r.onTok:r.offTok,{hollow:!r.fit,note:r.fit?"":t("infer.deploy.tbo.noFit")}];
  const ser=[{name:t("infer.deploy.tbo.off"),col:"s1",pts:s.rows.map(r=>pt(r,false))},{name:t("infer.deploy.tbo.on"),col:"s2",pts:s.rows.map(r=>pt(r,true))}];
  const cur=s.rows.some(r=>r.B===c.B)?[{x:c.B,label:t("infer.deploy.tbo.now")}]:[];
  lineChart("i-tbo",{series:ser,xLog:true,xBase:2,xFmt:String,yFmt:fN,yMin:0,xLab:t("infer.deploy.tbo.x"),yLab:t("infer.deploy.tbo.y"),vlines:cur});
  legendInto("i-tbo-legend",ser.map(x=>({name:x.name,col:x.col,line:true})));
  const at=s.rows.find(r=>r.B===c.B), big=s.rows[s.rows.length-1], pc=(a,b)=>(a/b-1>=0?"+":"")+Math.round((a/b-1)*100)+"%";
  $("i-tbo-read").innerHTML=ul([
    g.lb ? t("infer.deploy.tbo.readIdeal")
      : s.win ? t("infer.deploy.tbo.readWin",{b:s.win,bmax:big.B,on:fN(big.onTok),off:fN(big.offTok)})
      : t("infer.deploy.tbo.readNever",{a2a:fT(big.a2aOff),pct:Math.round(big.a2aOff/big.off*100)+"%",b:big.B}),
    at ? t("infer.deploy.tbo.readCur",{b:c.B,off:fT(at.off),on:fT(at.on),d:V(pc(at.on,at.off))}) : "",
    g.lb ? "" : t("infer.deploy.tbo.readWhy",{eta:Math.round(g.etaTbo*100)+"%"}) ]);
}
/* 推理参数架：随引擎和阶段显示 / 隐藏，填 MoE 并行和预设（一步、部署、投机三节共用） */
function syncInfRack(){
  const c=S.inf;
  const onSpec=location.hash==="#infer/spec";                        // 投机解码一节只算 decode，阶段选项和 prefill 的参数在这里没有意义
  qa("[data-phase]").forEach(el=>el.hidden=el.dataset.phase!==(onSpec?"decode":c.phase));
  qa("[data-eng]").forEach(el=>el.hidden=el.dataset.eng!==c.engine);
  if(!MOEPAR[c.engine].some(x=>x[0]===c.moePar)) c.moePar=Infer.ENGINES[c.engine].defaultMoe;
  fillOpts($("i-moepar"),MOEPAR[c.engine],c.moePar);
  presetSel($("i-preset"),$("i-preset-src"),infPresets(c.engine,S.modelKey,S.topo),S.infPreset);
  { const {m,g}=ctx(); markInert("inf",Object.assign(Infer.inert(m,c,g),onSpec?{phase:"specDecode"}:{})); }
  $("i-engine-note").textContent={
    sglang:t("infer.engineNote.sglang"),
    vllm:t("infer.engineNote.vllm"),
    tokenspeed:t("infer.engineNote.tokenspeed"),
    titan:t("infer.engineNote.titan")}[c.engine];
}
const specName = c => (Infer.SPEC[c.specMethod]||Infer.SPEC.mtp).name;
const draftLabel = (c,k) => c.specMethod==="dflash"||c.specMethod==="dspark" ? t("infer.draft.block",{name:specName(c),k})
  : c.specMethod==="eagle3" ? t("infer.draft.eagle3",{name:specName(c),k}) : t("infer.draft.mtp",{name:specName(c),k});
function renderInfer(){
  const {m,d,hw,topo,g}=ctx(), c=S.inf, dec=c.phase==="decode", E=Infer.ENGINES[c.engine];
  syncInfRack();
  const r=infRun(); const inst=Math.floor(topo.gpus/c.N);
  const warn=[...r.warn];
  if(topo.gpus%c.N) warn.push(t("infer.warn.notDivide",{n:c.N,gpus:topo.gpus}));
  if(c.N%c.dpA) warn.push(t("infer.warn.dpDivide"));
  if(c.N>topo.gpus) warn.push(t("infer.warn.tooBig"));
  if(c.moePar==="fused" && c.engine==="titan" && !blackwell()) warn.push(t("infer.warn.distMoe"));
  if(r.mem.avail<=0) warn.push(t("infer.warn.noFit",{w:fB(r.mem.weights),mem:hw.mem,frac:c.memFrac}));
  else if(dec && r.Breq>r.mem.Bmax) warn.push(t("infer.warn.bOver",{b:r.Breq,bmax:r.mem.Bmax,n:fInt(c.n)}));
  if(dec && !r.graph && c.graph){ const gp={engine:ENGN[c.engine],max:E.maxGraph(hw,r.specK>0)};
    warn.push(c.engine==="vllm"||c.engine==="titan"?t("infer.warn.graphCapTok",gp):t("infer.warn.graphCapReq",gp)); }
  setWarn(warn);
  $("i-lede").innerHTML=t("infer.lede.main",{model:esc(m.short),hw:hw.name,topo:esc(topo.name),engine:ENGN[c.engine],n:c.N,tp:fI(r.tpA),dp:r.dpA,moe:MOENAME[r.moePar],inst})+
    (dec? t("infer.lede.decode",{b:r.Breq,ctx:fInt(c.n),spec:r.specK?t("infer.lede.decodeSpec",{method:specName(c),k:r.specK,accept:c.accept}):""}) : t("infer.lede.prefill",{p:fInt(c.P),c:fInt(c.C)}));
  const wPer=r.mem.weights;
  if(dec) kpis($("i-kpis"),[
    ["TPOT",fT(r.tpot),r.specK?t("infer.kpi.tpotFootSpec",{step:fT(r.stepT),tok:(1+c.accept).toFixed(1)}):t("infer.kpi.tpotFoot",{step:fT(r.stepT)})],
    [t("infer.kpi.thr"),fN(r.tokPerGpu)+" tok/s/GPU",t("infer.kpi.thrFoot",{v:fN(r.tokPerGpu*topo.gpus)})],
    [t("infer.kpi.graph"),r.graph?t("infer.kpi.graphPad",{rows:r.rowsG}):t("infer.kpi.graphEager"),r.graph?t("infer.kpi.graphFoot",{rows:r.rowsR,pct:pct(r.rowsG-r.rowsR,r.rowsR)}):t("infer.kpi.eagerFoot",{n:r.nKern,us:g.launchUs}),!r.graph&&c.graph],
    [t("infer.kpi.fit"),t("infer.kpi.fitVal",{n:fInt(r.mem.Bmax)}),t("infer.kpi.fitFoot",{n:fInt(c.n)}),r.mem.Bmax<r.Breq],
    [t("infer.kpi.wPer"),fB(wPer),t("infer.kpi.wPerFoot",{b:fB(r.mem.wExp)}),r.mem.avail<=0],
    [t("infer.kpi.perSeq"),fB(r.mem.perSeq),r.mem.statePer>0?t("infer.kpi.perSeqFootState",{kv:fB(r.mem.kvTok*c.n),state:fB(r.mem.statePer),slots:r.mem.slots}):t("infer.kpi.perSeqFoot",{kv:fB(r.mem.kvTok*c.n)})]]);
  else kpis($("i-kpis"),[
    ["TTFT",fT(r.ttft),t("infer.kpi.ttftFoot",{n:r.nChunks})],
    [t("infer.kpi.firstChunk"),fT(r.chunks[0]?r.chunks[0].t:r.stepT),t("infer.kpi.firstChunkFoot",{last:fT(r.chunks.length?r.chunks[r.chunks.length-1].t:0)})],
    [t("infer.kpi.prefillThr"),fN(r.tokPerGpu)+" tok/s/GPU",t("infer.kpi.prefillThrFoot")],
    [t("infer.kpi.prefillGraph"),r.graph?t("infer.kpi.prefillGraphOn"):t("infer.kpi.prefillGraphOff"),E.prefillGraph(m)?"":t("infer.kpi.prefillGraphNone")],
    [t("infer.kpi.wPer"),fB(wPer),"",r.mem.avail<=0]]);
  // 时间分解
  const items=[];
  for(const o of r.ops) if(o.overlap!=="moe") items.push({label:o.name,v:o.t,color:GROUPC[o.group],tag:pill(o.bound)});
  for(const x of r.comm) if(!x.overlap) items.push({label:x.name,v:x.t,color:COMMC[x.key]||css("net"),tag:pill("comm")});
  if(r.exposedTbo>0) items.push({label:t("infer.bar.tbo"),v:r.exposedTbo,color:css("c-a2a"),tag:pill("comm")});
  if(r.exposedOverlap>0) items.push({label:t("infer.bar.exposed"),v:r.exposedOverlap,color:css("c-a2a"),tag:pill("comm")});
  const base=r.stepT-(r.draftT||0)-(r.commitT||0), dpSync=(c.engine==="vllm"||c.engine==="titan")&&r.dpA>1?g.dpSyncUs*1e-6:0;
  if(r.graph) items.push({label:t("infer.bar.graph"),v:g.graphLaunchUs*1e-6,color:css("c-free"),tag:pill("launch")});
  const cpuExtra=base-dpSync-r.gpu-(r.graph?g.graphLaunchUs*1e-6:0);
  if(cpuExtra>1e-9) items.push({label:t("infer.bar.cpu",{n:r.nKern}),v:cpuExtra,color:css("c-free"),tag:pill("launch")});
  if(dpSync) items.push({label:t("infer.bar.dpSync"),v:dpSync,color:css("c-free"),tag:pill("launch")});
  if(r.draftT) items.push({label:draftLabel(c,r.specK),v:r.draftT,color:css("c-cp")});
  if(r.commitT) items.push({label:t("infer.bar.commit"),v:r.commitT,color:css("c-lin")});
  stackBar($("i-bar"),$("i-barlist"),items,r.stepT,fT);
  // 换引擎
  const engs=infEngines(), runs={}; engs.forEach(e=>runs[e]=infRun(asEng(e)));
  if(!dec){
    let ymax=0; const series=engs.map(e=>{ const pts=runs[e].chunks.map(ch=>[ch.p+ch.C/2,ch.t*1e3]); pts.forEach(p=>ymax=Math.max(ymax,p[1])); return {pts,color:ENGC[e],w:e===c.engine?2.8:1.6}; });
    plot($("i-chunks"),{x0:0,x1:c.P,y0:0,y1:Math.max(ymax,1e-3)*1.1,fx:v=>fxPow(v),fy:v=>+v.toPrecision(3)+"",xl:t("infer.chunks.xl"),yl:t("infer.chunks.yl"),series});
    $("i-chunks-legend").innerHTML=engs.map(e=>'<span style="color:'+ENGC[e]+'"><i></i>'+t("infer.chunks.legend",{engine:ENGN[e],ttft:fT(runs[e].ttft)})+'</span>').join("");
  }
  // 引擎差别表
  const yes=b=>b?t("infer.diff.yes"):t("infer.diff.no"), P=e=>Infer.ENGINES[e];
  const R=[[t("infer.diff.mixed"),e=>yes(P(e).mixed)],
    [t("infer.diff.maxGraph"),e=>e==="vllm"||e==="titan"?t("infer.diff.maxGraphTok",{n:P(e).maxGraph(hw,(c.specK||0)>0)}):t("infer.diff.maxGraphReq",{n:P(e).maxGraph(hw,(c.specK||0)>0)})],
    dec?[t("infer.diff.graphPad"),e=>runs[e].graph?t("infer.diff.graphPadVal",{real:runs[e].rowsR,pad:runs[e].rowsG}):t("infer.diff.eager")]:[t("infer.diff.prefillGraph"),e=>yes(runs[e].graph)],
    [t("infer.diff.lmInGraph"),e=>yes(P(e).lmInGraph)+" / "+yes(P(e).samplerInGraph)],
    [t("infer.diff.arFusion"),e=>runs[e].tpA>1?yes(runs[e].fused):t("infer.diff.noAr")],
    [t("infer.diff.moe"),e=>MOENAME[P(e).defaultMoe]+" / "+MOENAME[runs[e].moePar]],
    [t("infer.diff.specCap"),e=>isFinite(P(e).specCap)?P(e).specCap:t("infer.diff.unlimited")],
    ...(m.lin.n>0?[[t("infer.diff.stateTraffic"),e=>dec&&runs[e].specK>0?P(e).stateVerify(m,runs[e].w)+"·S_l":P(e).stateDecode(m)+"·S_l"],
      [t("infer.diff.hMat"),e=>yes(P(e).hMat(m,hw))]]:[]),
    ...(m.full.kind==="csa"?[[t("infer.diff.replay"),e=>P(e).replay?t("infer.diff.replayAt."+P(e).replay):t("infer.diff.noReplay")]]
      :m.full.kind!=="gqa"?[[t("infer.diff.mlaReexp"),e=>m.full.kind==="dsa"&&e==="tokenspeed"?t("infer.diff.noAbsorbed"):yes(P(e).mlaReexpand)]]:[]),
    [t("infer.diff.sharedOverlap"),e=>({graph:t("infer.diff.shared.graph"),small:t("infer.diff.shared.small"),none:t("infer.diff.shared.none")})[P(e).sharedOverlap]],
    [t("infer.diff.nKern"),e=>fInt(runs[e].nKern)],
    ...(dec?[[t("infer.diff.step"),e=>fT(runs[e].stepT)],["TPOT",e=>fT(runs[e].tpot)],["tok/s/GPU",e=>fN(runs[e].tokPerGpu)],[t("infer.diff.bmax"),e=>fInt(runs[e].mem.Bmax)]]
      :[["TTFT",e=>fT(runs[e].ttft)],[t("infer.diff.prefillThr"),e=>fN(runs[e].tokPerGpu)]])];
  table($("i-diff"),[""].concat(engs.map(e=>e===c.engine?t("infer.diff.headCur",{engine:ENGN[e]}):ENGN[e])),R.map(row=>[row[0]].concat(engs.map(e=>row[1](e)))));
  // 显存
  const M=r.mem, tot=M.total, Bm = dec ? r.Breq*r.dpA/r.dpA : 1;
  const kvU=Math.min(Bm,Math.max(0,M.Bmax))*M.kvTok*c.n, stU=Math.min(Bm,Math.max(0,M.Bmax))*M.statePer;
  stackBar($("i-mem"),$("i-memlist"),[
    {label:t("infer.mem.wAttn"),v:M.wAttn,color:css("c-attn")},{label:t("infer.mem.wLin"),v:M.wLin,color:css("c-lin")},{label:t("infer.mem.wExp"),v:M.wExp,color:css("c-moe")},
    {label:t("infer.mem.wMoeRep"),v:M.wMoeRep,color:css("c-e")},{label:t("infer.mem.wDense"),v:M.wDense,color:css("c-e")},{label:t("infer.mem.wEmb"),v:M.wEmb,color:css("c-lm")},{label:t("infer.mem.wMtp"),v:M.wMtp,color:css("c-cp")},{label:specName(c)+" drafter",v:M.wDraft||0,color:css("c-cp")},
    {label:t("infer.mem.kv",{b:Bm,n:fInt(c.n)}),v:kvU,color:css("c-kv")},{label:t("infer.mem.state",{b:Bm,slots:M.slots}),v:stU,color:css("c-st")},
    {label:t("infer.mem.reserve"),v:c.reserveGB*1e9,color:css("c-act")},{label:t("infer.mem.free"),v:Math.max(0,M.avail-kvU-stU),color:css("soft")},{label:t("infer.mem.outside"),v:tot*(1-c.memFrac),color:css("c-free")}],tot,fB);
}
/* 每个 kernel 的明细在模块页：点过去时带上这里的负载。模块页的权重 / expert 精度对应这里的 dDt / wDt，prefill 的 n 是 prompt 长度 */
$("i-to-mods").addEventListener("click",e=>{ if(!e.target.closest("a")) return; const c=S.inf;
  Object.assign(S.arch,{phase:c.phase,B:c.B,n:c.phase==="decode"?c.n:c.P,C:c.C,wDt:c.dDt,eDt:c.wDt,kvDt:c.kvDt,stateDt:c.stateDt}); syncInputs(); });
