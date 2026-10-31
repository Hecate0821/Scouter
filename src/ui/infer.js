/* ================= 推理 · 一步 decode / prefill（参数架三节共用） ================= */
function infRun(over, phase){ const {m,hw,topo,g}=ctx(); const c=Object.assign({},S.inf,over||{}); return (phase||c.phase)==="decode" ? Infer.decode(m,hw,topo,c,g) : Infer.prefill(m,hw,topo,c,g); }
const infEngines = () => ["sglang","vllm","tokenspeed"].concat(S.inf.engine==="titan"?["titan"]:[]);
const asEng = e => ({engine:e, moePar:compatMoe(e,S.inf.moePar)});
/* 推理参数架：随引擎和阶段显示 / 隐藏，填 MoE 并行和预设（一步、部署、投机三节共用） */
function syncInfRack(){
  const c=S.inf;
  qa("[data-phase]").forEach(el=>el.hidden=el.dataset.phase!==c.phase);
  qa("[data-eng]").forEach(el=>el.hidden=el.dataset.eng!==c.engine);
  if(!MOEPAR[c.engine].some(x=>x[0]===c.moePar)) c.moePar=Infer.ENGINES[c.engine].defaultMoe;
  fillOpts($("i-moepar"),MOEPAR[c.engine],c.moePar);
  presetSel($("i-preset"),$("i-preset-src"),infPresets(c.engine,S.modelKey,S.topo),S.infPreset);
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
    ["CUDA graph",r.graph?t("infer.kpi.graphPad",{rows:r.rowsG}):t("infer.kpi.graphEager"),r.graph?t("infer.kpi.graphFoot",{rows:r.rowsR,pct:pct(r.rowsG-r.rowsR,r.rowsR)}):t("infer.kpi.eagerFoot",{n:r.nKern,us:g.launchUs}),!r.graph&&c.graph],
    [t("infer.kpi.fit"),t("infer.kpi.fitVal",{n:fInt(r.mem.Bmax)}),t("infer.kpi.fitFoot",{n:fInt(c.n)}),r.mem.Bmax<r.Breq],
    [t("infer.kpi.wPer"),fB(wPer),"expert "+fB(r.mem.wExp),r.mem.avail<=0],
    [t("infer.kpi.perSeq"),fB(r.mem.perSeq),r.mem.statePer>0?t("infer.kpi.perSeqFootState",{kv:fB(r.mem.kvTok*c.n),state:fB(r.mem.statePer),slots:r.mem.slots}):t("infer.kpi.perSeqFoot",{kv:fB(r.mem.kvTok*c.n)})]]);
  else kpis($("i-kpis"),[
    ["TTFT",fT(r.ttft),t("infer.kpi.ttftFoot",{n:r.chunks.length})],
    [t("infer.kpi.firstChunk"),fT(r.chunks[0]?r.chunks[0].t:r.stepT),t("infer.kpi.firstChunkFoot",{last:fT(r.chunks.length?r.chunks[r.chunks.length-1].t:0)})],
    [t("infer.kpi.prefillThr"),fN(r.tokPerGpu)+" tok/s/GPU",t("infer.kpi.prefillThrFoot")],
    ["prefill graph",r.graph?t("infer.kpi.prefillGraphOn"):t("infer.kpi.prefillGraphOff"),E.prefillGraph(m)?"":t("infer.kpi.prefillGraphNone")],
    [t("infer.kpi.wPer"),fB(wPer),"",r.mem.avail<=0]]);
  // 时间分解
  const items=[];
  for(const o of r.ops) if(o.overlap!=="moe") items.push({label:o.name,v:o.t,color:GROUPC[o.group],tag:pill(o.bound)});
  for(const x of r.comm) if(!x.overlap) items.push({label:x.name,v:x.t,color:COMMC[x.key]||css("net"),tag:pill("comm")});
  const tboC=sum(r.comm.filter(x=>x.overlap==="tbo"),x=>x.t);
  if(tboC>0) items.push({label:t("infer.bar.tbo"),v:Math.max(0,tboC-r.tComp*0.5),color:css("c-a2a"),tag:pill("comm")});
  if(r.exposedOverlap>0) items.push({label:t("infer.bar.exposed"),v:r.exposedOverlap,color:css("c-a2a"),tag:pill("comm")});
  const base=r.stepT-(r.draftT||0)-(r.commitT||0), dpSync=(c.engine==="vllm"||c.engine==="titan")&&r.dpA>1?g.dpSyncUs*1e-6:0;
  if(r.graph) items.push({label:"graph replay",v:g.graphLaunchUs*1e-6,color:css("c-free"),tag:pill("launch")});
  const cpuExtra=base-dpSync-r.gpu-(r.graph?g.graphLaunchUs*1e-6:0);
  if(cpuExtra>1e-9) items.push({label:t("infer.bar.cpu",{n:r.nKern}),v:cpuExtra,color:css("c-free"),tag:pill("launch")});
  if(dpSync) items.push({label:t("infer.bar.dpSync"),v:dpSync,color:css("c-free"),tag:pill("launch")});
  if(r.draftT) items.push({label:draftLabel(c,r.specK),v:r.draftT,color:css("c-cp")});
  if(r.commitT) items.push({label:t("infer.bar.commit"),v:r.commitT,color:css("c-lin")});
  stackBar($("i-bar"),$("i-barlist"),items,r.stepT,fT);
  // kernel 表
  $("i-ops-lede").innerHTML=t("infer.ops.lede",{engine:ENGN[c.engine]})+(dec?t("infer.ops.ledeRows",{pad:r.rowsG,real:r.Breq,global:fInt(r.Tg)}):"");
  const rowsOf=o=> !dec ? "—" : o.scalesWith==="real" ? t("infer.ops.rows.real",{n:r.Breq}) : (o.key==="router"&&r.moePar==="replicated") ? t("infer.ops.rows.global",{n:fInt(r.Tg)}) : o.key==="moe" ? t("infer.ops.rows.expert") : (o.key==="lm"||o.key==="samp") ? (E.lmInGraph&&r.graph&&o.key==="lm"||E.samplerInGraph&&r.graph&&o.key==="samp" ? t("infer.ops.rows.pad",{n:r.rowsG}) : t("infer.ops.rows.actual",{n:r.rowsR})) : t("infer.ops.rows.pad",{n:r.rowsG});
  const note=o=>[o.overlap==="moe"?t("infer.ops.note.hidden"):"", o.stateMult?t("infer.ops.note.state",{n:o.stateMult}):"", o.touched!=null?t("infer.ops.note.touched",{n:fI(o.touched),pct:pct(o.cover,1)}):""].filter(Boolean).join(t("infer.ops.note.sep"));
  table($("i-ops"),[t("infer.ops.head.kernel"),t("infer.ops.head.layers"),t("infer.ops.head.rows"),"FLOPs",t("infer.ops.head.bytes"),"I",t("infer.ops.head.bound"),t("infer.ops.head.time"),t("infer.ops.head.note")],
    r.ops.map(o=>[sw(GROUPC[o.group])+esc(o.name),o.layers,rowsOf(o),fF(o.flops),fB(o.bytes),fI(o.I),pill(o.bound),fT(o.t),esc(note(o))]),[1,3,4,5,7]);
  table($("i-comm"),[t("infer.comm.head.comm"),t("infer.comm.head.calls"),t("infer.comm.head.bytes"),t("infer.comm.head.time"),t("infer.comm.head.overlap")],
    r.comm.length? r.comm.map(x=>[sw(COMMC[x.key]||css("net"))+esc(x.name),x.layers,fB(x.bytes),fT(x.t),x.overlap==="moe"?t("infer.comm.ov.moe"):x.overlap==="tbo"?t("infer.comm.ov.tbo"):t("infer.comm.ov.none")]) : [[t("infer.comm.none"),"","","",""]],[1,2,3]);
  // roofline
  const pks=[{v:peakOf(hw,"bf16"),n:"BF16"}]; if(dtB(c.wDt)<2) pks.push({v:peakOf(hw,c.wDt),n:c.wDt.toUpperCase()}); else if(dtB(c.dDt)<2) pks.push({v:peakOf(hw,c.dDt),n:"FP8"});
  const top=new Set([...r.ops].sort((a,b)=>b.t-a.t).slice(0,10));
  plot($("i-roof"),{h:380,x0:0.05,x1:20000,y0:1e9,y1:Math.max(...pks.map(p=>p.v))*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:t("infer.roof.xl"),yl:t("infer.roof.yl"),
    series:roofSeries(hw,g,pks,0.05,20000),points:r.ops.filter(o=>o.flops>0).map(o=>({x:o.I,y:o.flops/o.t,color:GROUPC[o.group],label:top.has(o)?o.name.replace(/（.*?）/g,""):null})),
    vlines:pks.map((p,i)=>({x:p.v*g.etaC/(hw.hbm*1e9*g.etaM),color:css("muted"),label:t("infer.roof.knee",{name:p.n,i:fI(p.v*g.etaC/(hw.hbm*1e9*g.etaM))}),dy:i*14}))});
  $("i-roof-legend").innerHTML='<span><i></i>'+t("infer.roof.legendBf16")+'</span>'+(pks[1]?'<span><i class="d"></i>'+t("infer.roof.legendPeak",{name:pks[1].n})+'</span>':'')+
    Object.entries({attn:t("infer.roof.group.attn"),lin:t("infer.roof.group.lin"),moe:t("infer.roof.group.moe"),lm:t("infer.roof.group.lm"),misc:t("infer.roof.group.misc")}).map(([k,v])=>'<span><span class="sw" style="background:'+GROUPC[k]+'"></span>'+v+'</span>').join("")+'<span>'+t("infer.roof.legendFloor")+'</span>';
  // 换引擎
  const engs=infEngines(), runs={}; engs.forEach(e=>runs[e]=infRun(asEng(e)));
  if(dec){
    const bmx=Math.max(...engs.map(e=>runs[e].mem.Bmax)), x1=Math.max(16,Math.min(1024,Math.ceil(Math.max(bmx,r.Breq)*1.15))), st=Math.max(1,Math.round(x1/360));
    const lat=[], thr=[]; let ymaxL=0, ymaxT=0;
    for(const e of engs){ const pl=[], pt=[]; const bm=runs[e].mem.Bmax;
      for(let B=1;B<=x1;B+=st){ const q=infRun(Object.assign(asEng(e),{B}),"decode"); if(q.Breq<B) break; const ok=B<=bm;
        pl.push([B,ok?q.tpot*1e3:NaN]); pt.push([B,ok?q.tokPerGpu:NaN]); if(ok){ ymaxL=Math.max(ymaxL,q.tpot*1e3); ymaxT=Math.max(ymaxT,q.tokPerGpu);} }
      lat.push({pts:pl,color:ENGC[e],w:e===c.engine?2.8:1.6}); thr.push({pts:pt,color:ENGC[e],w:e===c.engine?2.8:1.6}); }
    const vl=[{x:r.Breq,color:css("ink"),label:t("infer.sweep.curB")}];
    plot($("i-sw-lat"),{x0:0,x1,y0:0,y1:Math.max(ymaxL,1e-3)*1.1,fx:v=>fInt(v),fy:v=>+v.toPrecision(3)+"",xl:t("infer.sweep.xl"),yl:t("infer.sweep.ylLat"),series:lat,vlines:vl});
    plot($("i-sw-thr"),{x0:0,x1,y0:0,y1:Math.max(ymaxT,1)*1.1,fx:v=>fInt(v),fy:v=>fN(v),xl:t("infer.sweep.xl"),yl:"tok/s/GPU",series:thr,vlines:vl});
    $("i-sw-legend").innerHTML=engs.map(e=>'<span style="color:'+ENGC[e]+'"><i></i>'+t("infer.sweep.legend",{engine:ENGN[e],moe:MOENAME[compatMoe(e,c.moePar)],bmax:runs[e].mem.Bmax})+'</span>').join("")+'<span>'+t("infer.sweep.legendCut")+'</span>';
  } else {
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
    ...(m.full.kind!=="gqa"?[[t("infer.diff.mlaReexp"),e=>m.full.kind==="dsa"&&e==="tokenspeed"?t("infer.diff.noAbsorbed"):yes(P(e).mlaReexpand)]]:[]),
    [t("infer.diff.sharedOverlap"),e=>({graph:t("infer.diff.shared.graph"),small:t("infer.diff.shared.small"),none:t("infer.diff.shared.none")})[P(e).sharedOverlap]],
    [t("infer.diff.nKern"),e=>fInt(runs[e].nKern)],
    ...(dec?[[t("infer.diff.step"),e=>fT(runs[e].stepT)],["TPOT",e=>fT(runs[e].tpot)],["tok/s/GPU",e=>fN(runs[e].tokPerGpu)],[t("infer.diff.bmax"),e=>fInt(runs[e].mem.Bmax)]]
      :[["TTFT",e=>fT(runs[e].ttft)],["prefill tok/s/GPU",e=>fN(runs[e].tokPerGpu)]])];
  table($("i-diff"),[""].concat(engs.map(e=>e===c.engine?t("infer.diff.headCur",{engine:ENGN[e]}):ENGN[e])),R.map(row=>[row[0]].concat(engs.map(e=>row[1](e)))));
  // 显存
  const M=r.mem, tot=M.total, Bm = dec ? r.Breq*r.dpA/r.dpA : 1;
  const kvU=Math.min(Bm,Math.max(0,M.Bmax))*M.kvTok*c.n, stU=Math.min(Bm,Math.max(0,M.Bmax))*M.statePer;
  stackBar($("i-mem"),$("i-memlist"),[
    {label:t("infer.mem.wAttn"),v:M.wAttn,color:css("c-attn")},{label:t("infer.mem.wLin"),v:M.wLin,color:css("c-lin")},{label:t("infer.mem.wExp"),v:M.wExp,color:css("c-moe")},
    {label:t("infer.mem.wMoeRep"),v:M.wMoeRep,color:css("c-e")},{label:"dense FFN",v:M.wDense,color:css("c-e")},{label:"embedding + lm_head",v:M.wEmb,color:css("c-lm")},{label:t("infer.mem.wMtp"),v:M.wMtp,color:css("c-cp")},{label:specName(c)+" drafter",v:M.wDraft||0,color:css("c-cp")},
    {label:t("infer.mem.kv",{b:Bm,n:fInt(c.n)}),v:kvU,color:css("c-kv")},{label:t("infer.mem.state",{b:Bm,slots:M.slots}),v:stU,color:css("c-st")},
    {label:t("infer.mem.reserve"),v:c.reserveGB*1e9,color:css("c-act")},{label:t("infer.mem.free"),v:Math.max(0,M.avail-kvU-stU),color:css("soft")},{label:t("infer.mem.outside"),v:tot*(1-c.memFrac),color:css("c-free")}],tot,fB);
  // 解析
  const ops=[...r.ops].sort((a,b)=>b.t-a.t), memT=sum(r.ops.filter(o=>o.bound==="memory"),o=>o.t), launchT=sum(r.ops.filter(o=>o.bound==="launch"),o=>o.t), commT=sum(r.comm.filter(x=>!x.overlap),x=>x.t);
  const moeO=r.ops.find(o=>o.key==="moe"), kvO=r.ops.filter(o=>o.scalesWith==="real"), kvBytes=sum(kvO,o=>o.bytes);
  const A=[];
  A.push(t("infer.analysis.slowest",{list:ops.slice(0,3).map(o=>esc(o.name.replace(/（.*?）/g,""))+" "+V(fT(o.t))+" "+pill(o.bound)).join(t("infer.analysis.listSep"))}));
  A.push(t("infer.analysis.shares",{mem:V(pct(memT,r.stepT)),launch:V(pct(launchT,r.stepT)),comm:V(pct(commT,r.stepT))}));
  if(moeO&&dec) A.push(t("infer.analysis.moe",{rows:fInt(r.Tg),touched:V(fI(moeO.touched)),cover:pct(moeO.cover,1),bytes:V(fB(moeO.bytes/moeO.layers))}));
  if(dec&&kvO.length) A.push(t("infer.analysis.kv",{bytes:V(fB(kvBytes)),time:V(fT(sum(kvO,o=>o.t)))})+(m.lin.n>0&&r.specK>0?t("infer.analysis.kvSpec",{x:(E.stateVerify(m,r.w)/E.stateDecode(m)).toFixed(1),engine:ENGN[c.engine]}):""));
  if(dec&&r.graph&&r.rowsG>r.rowsR) A.push(t("infer.analysis.pad",{real:r.rowsR,pad:r.rowsG}));
  if(dec&&!r.graph) A.push(t("infer.analysis.eager",{n:r.nKern,us:g.launchUs,total:V(fT(r.nKern*g.launchUs*1e-6)),tail:r.cpuLaunch>r.gpu?t("infer.analysis.eagerCpu"):t("infer.analysis.eagerEnd")}));
  if(r.tpA>1) A.push((r.fused?t("infer.analysis.tpFused",{tp:fI(r.tpA)}):t("infer.analysis.tpUnfused",{tp:fI(r.tpA)}))+(m.full.kind!=="gqa"?(c.dcp>1?t("infer.analysis.mlaDcp",{dcp:c.dcp}):t("infer.analysis.mlaNoDcp")):m.full.Hkv<r.tpA?t("infer.analysis.kvRepl",{hkv:m.full.Hkv,tp:fI(r.tpA)}):""));
  if(c.engine==="tokenspeed" && dec && r.Breq>E.maxGraph(hw,r.specK>0)) A.push(t("infer.analysis.tsGraph",{max:E.maxGraph(hw,r.specK>0)}));
  if(c.engine==="sglang" && r.specK>0) A.push(t("infer.analysis.sglSpec"));
  if(c.engine==="vllm") A.push(t("infer.analysis.vllmMixed")+(r.dpA>1?t("infer.analysis.vllmDp"):""));
  if(r.moePar==="replicated"&&r.N>1) A.push(t("infer.analysis.replicated",{rows:fInt(r.Tg)}));
  $("i-analysis").innerHTML=ul(A);
}
