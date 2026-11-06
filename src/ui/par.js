/* ---------- 后训练核心的 Train.run 换参数 ---------- */
/* 后训练的并行一节按 torchtitan 的 FSDP + CP + EP 摆：框架是 Megatron 时，同样的 TP / CP / EP 度搬过去，PP 去掉、分布式优化器换成 FSDP2 */
const MG2TT = { disp:{alltoall:"standard",deepep:"deepep",hybridep:"deepep",ncclep:"standard"}, cp:{p2p:"ring",allgather:"allgather",a2a:"ulysses"} };
function parBase(){
  const c=S.trn; if(c.fw==="torchtitan") return clone(c);
  return Object.assign(clone(c), { fw:"torchtitan", pp:1, vpp:1, etp:1, dispatcher:MG2TT.disp[c.dispatcher]||"standard", cpType:MG2TT.cp[c.cpType]||"allgather",
    linCP:"state", recompute:c.recompute==="none"?"none":"full", offload:false, tp:Math.min(c.tp, Math.max(1,c.ep)) });
}
function runP(over, topoKey, hwKey){ const {m,g}=ctx(), topo=TOPO[topoKey||S.topo], hw=HW[hwKey||S.hw];
  return Train.run(m, hw, topo, Object.assign({N:topo.gpus}, parBase(), over||{}), g); }
const runOK = r => r.dp>=1 && Number.isInteger(r.dp) && !r.warn.some(w=>w.bad);      // 计算核心把“配置跑不起来”的警告标成 invalid
const runFit = r => r.mem.used <= r.mem.total;
/* 把 TP / CP / EP 缩到 N 张卡放得下 */
function fitPar(c, N){ c=Object.assign({},c); while(c.tp*c.cp*(c.pp||1)>N || N%(c.tp*c.cp*(c.pp||1))){ if(c.cp>1) c.cp/=2; else if(c.tp>1) c.tp/=2; else break; }
  while(c.ep>1 && (c.ep>N/(c.pp||1) || c.ep<c.tp)) c.ep/=2; return c; }

/* 代表层的每一段（每层、最忙那张卡）：attention 前后投影、核心、MoE 复制部分、expert、dense FFN、线性注意力 */
const PRE_KEYS=["aqkv","arope","aqa","aqb","akvb","auk","aidxp","aidx","acmp"], MOEREP_KEYS=["router","latd","permute","unperm","latu","shared"];
function layerPieces(r){
  const acc = pred => { let f=0,b=0,rc=0; for(const o of r.ops) if(pred(o)){ f+=o.fwd.t/o.layers; b+=(o.dgrad.t+o.wgrad.t)/o.layers; if(o.rc) rc+=o.fwd.t/o.layers; } return {f,b,rc}; };
  const cf=r.par.coreFullOp;
  return { pre:acc(o=>o.group==="attn"&&PRE_KEYS.includes(o.key)), core:cf?acc(o=>o===cf):{f:0,b:0,rc:0}, post:acc(o=>o.group==="attn"&&o.key!=="acore"&&!PRE_KEYS.includes(o.key)),
    rep:acc(o=>MOEREP_KEYS.includes(o.key)), exp:acc(o=>["moe1","act","moe2"].includes(o.key)), den:acc(o=>o.key==="dense"), lin:acc(o=>o.group==="lin") };
}
/* EP 段：torchtitan 直接用 Par.epLayer 的分段；Megatron 从通信表的 a2a 拆成 dispatch / combine */
function epPartsOf(r, P){
  if(r.par.ep) return r.par.ep.parts;
  const {hw,topo}=ctx(), a=r.comm.find(x=>x.key==="a2a"), cn=r.comm.find(x=>x.key==="cnt"), nA=a?a.exch:4;
  const one=a?a.t/a.layers/nA:0, lane=Common.link(hw,topo,r.ep,r.etp).intra?"nvl":"ib", sync=cn?cn.t/cn.layers:0, f=[], b=[];
  if(sync) f.push(["cpu",t("par.seg.epSync"),sync,"sync"]);
  f.push([lane,t("par.seg.disp"),one,"ep"],["sm",t("par.seg.exp"),P.exp.f,"moe"],[lane,t("par.seg.comb"),one,"ep"]);
  if(nA>4) b.push([lane,t("par.seg.rcDisp"),one,"ep"],[lane,t("par.seg.rcComb"),one,"ep"]);
  b.push([lane,t("par.seg.bwdComb"),one,"ep"],["sm",t("par.seg.expBwd"),P.exp.b+P.exp.rc,"moe"],[lane,t("par.seg.bwdDisp"),one,"ep"]);
  return {f,b};
}
/* 一层的时间线：which = attn（只看注意力，比较 CP）、moe（只看 MoE，比较 EP） */
function layerSeq(r, which){
  const {m,d,hw,topo}=ctx(), P=layerPieces(r), ac=r.ops.some(o=>o.rc), f=[], b=[];
  const cp=r.par.cpFull, meth=r.par.cpMeth, C=r.cp, lane=cp&&!cp.lk.intra?"ib":"nvl";
  const cw=cp?r.par.docs*cp.tw:0, cwF=cw*(ac?1/5:1/4), cwB=Math.max(0,cw-cwF*(ac?2:1));
  const tpAll=sum(r.comm.filter(x=>x.cls==="tp"),x=>x.t)/m.L, tpL=Common.link(hw,topo,r.tp,1).intra?"nvl":"ib", H=d.Hattn/r.tp;
  const cpFwd=(arr,coreT,wireT)=>{
    if(C<=1||!cp){ arr.push(["sm",t("par.seg.core"),coreT,"attn"]); return; }
    if(meth==="ulysses"){ arr.push([lane,t("par.seg.uly.in"),wireT/2,"cp"],["sm",t("par.seg.uly.core", { h: Math.floor(H/C) }),coreT,"attn",cp.ulyOverlap],[lane,t("par.seg.uly.out"),wireT/2,"cp"]); return; }
    if(meth==="allgather"){ arr.push([lane,t("par.seg.ag.gather"),wireT,"cp"],["sm",t("par.seg.ag.core"),coreT,"attn",true]); return; }
    const st=Math.min(C,8); for(let i=0;i<st;i++){ arr.push(["sm",t("par.seg.ring.block", { i: i+1, total: i===st-1&&C>st ? t("par.seg.ring.total", { n: C }) : "" }),coreT/st,"attn"]); if(i<st-1) arr.push([lane,t("par.seg.ring.send"),wireT/(st-1),"cp",true]); }
  };
  const moe=m.moe.E>0, EPp=moe?epPartsOf(r,P):null;
  if(which!=="moe"){
    if(tpAll) f.push([tpL,t("par.seg.tpAg"),tpAll/6,"tp"]);
    f.push(["sm",t("par.seg.qkv"),P.pre.f,"proj"]); cpFwd(f,P.core.f,cwF); f.push(["sm",t("par.seg.oGate"),P.post.f,"proj"]);
    if(tpAll) f.push([tpL,t("par.seg.tpRs"),tpAll/6,"tp"]);
  }
  if(which!=="attn"){ if(moe){ f.push(["sm",t("par.seg.rep"),P.rep.f,"proj"]); f.push(...EPp.f); } else f.push(["sm",t("par.seg.dense"),P.den.f,"proj"]); }
  if(which!=="attn"){ if(moe){ b.push(...EPp.b); b.push(["sm",t("par.seg.routerBwd"),P.rep.b,"proj"]); } else b.push(["sm",t("par.seg.denseBwd"),P.den.b,"proj"]); }
  if(which!=="moe"){
    if(tpAll) b.push([tpL,t("par.seg.tpAgBwd"),tpAll/2,"tp"]);
    b.push(["sm",t("par.seg.oBwd"),P.post.b,"proj"]);
    const coreB=P.core.b+(which==="attn"?P.core.rc:0);
    if(C<=1||!cp) b.push(["sm",t("par.seg.coreBwd"),coreB,"attn"]);
    else if(meth==="ulysses") b.push([lane,"all-to-all dO",cwB/2,"cp"],["sm",t("par.seg.coreBwd"),coreB,"attn",cp.ulyOverlap],[lane,"all-to-all dQ / dK / dV",cwB/2,"cp"]);
    else if(meth==="allgather") b.push([lane,t("par.seg.ag.regather"),cwB/2,"cp"],["sm",t("par.seg.coreBwd"),coreB,"attn",true],[lane,"reduce-scatter dK / dV",cwB/2,"cp"]);
    else { const st=Math.min(C,8); for(let i=0;i<st;i++){ b.push(["sm",t("par.seg.ring.blockBwd", { i: i+1 }),coreB/st,"attn"]); if(i<st-1) b.push([lane,t("par.seg.ring.sendBwd"),cwB/(st-1),"cp",true]); } }
    b.push(["sm",t("par.seg.qkvBwd"),P.pre.b,"proj"]);
    if(tpAll) b.push([tpL,t("par.seg.tpRsBwd"),tpAll/6,"tp"]);
  }
  return {f,b};
}
const seqGroup = (r, which, label) => { const sq=layerSeq(r,which), F=layout(sq.f,0), B=layout(sq.b,F.end), segs=F.segs.concat(B.segs);
  return { label, lanes:["cpu","sm","nvl","ib"].filter(x=>x==="sm"||segs.some(z=>z.lane===x)), segs, fEnd:F.end, end:B.end }; };

/* ---------- 后训练 · 并行：网格 ---------- */
let meshDim="cp", meshSel=0;
function meshDims(r){
  const N=r.dp*r.tp*r.cp*r.pp, Nst=N/r.pp, tp=r.tp, cp=r.cp, ep=r.ep, etp=r.etp, fs=r.par.fsdp, D=[];
  if(tp>1) D.push({k:"tp",n:t("par.dim.tp"),c:"c-ar",size:tp,stride:1,g:k=>Math.floor(k/tp)});
  if(cp>1) D.push({k:"cp",n:t("par.dim.cp"),c:"c-cp",size:cp,stride:tp,g:k=>Math.floor(k/(tp*cp))*tp+k%tp});
  if(ep>1) D.push({k:"ep",n:t("par.dim.ep"),c:"c-a2a",size:ep,stride:etp,g:k=>Math.floor(k/(ep*etp))*etp+k%etp});
  if(fs){ const blk=tp*fs.fs; D.push({k:"fsdp",n:t("par.dim.fsdp"),c:"c-fsdp",size:fs.fs,stride:tp,g:k=>Math.floor(k/blk)*tp+k%tp});
    if(ep>1 && fs.fe>1){ const eb=ep*fs.fe; D.push({k:"efsdp",n:t("par.dim.efsdp"),c:"c-moe",size:fs.fe,stride:ep,g:k=>Math.floor(k/eb)*ep+k%ep}); }
    if(fs.R>1){ const rs=Nst/fs.R; D.push({k:"rep",n:t("par.dim.rep"),c:"c-attn",size:fs.R,stride:rs,g:k=>Math.floor(k/Nst)*rs+k%rs}); } }
  else { if(r.dp>1) D.push({k:"dp",n:t("par.dim.dp"),c:"c-fsdp",size:r.dp,stride:tp*cp,g:k=>Math.floor(k/Nst)*(tp*cp)+k%(tp*cp)});
    if(ep>1 && r.edp>1) D.push({k:"edp",n:t("par.dim.edp"),c:"c-moe",size:r.edp,stride:ep*etp,g:k=>Math.floor(k/Nst)*(ep*etp)+k%(ep*etp)}); }
  if(r.pp>1) D.push({k:"pp",n:t("par.dim.pp"),c:"c-dp",size:r.pp,stride:Nst,g:k=>k%Nst});
  return D;
}
const PAR_LINK = { tp:"", cp:"#train/par/cp", ep:"#train/par/ep", fsdp:"#train/par/fsdp", efsdp:"#train/par/fsdp", rep:"#train/par/fsdp", dp:"", edp:"", pp:"" };
function renderMesh(r){
  const {hw,topo}=ctx(), N=Math.round(r.dp*r.tp*r.cp*r.pp), dom=Par.pow2Dom(topo), D=meshDims(r);
  if(meshSel>=N) meshSel=0;
  if(!D.length){ $("p-mesh-dim").innerHTML=""; $("p-mesh").innerHTML='<p class="hint">'+t("par.mesh.single")+'</p>'; $("p-mesh-cap").textContent=""; $("p-groups").innerHTML=""; return; }
  if(!D.some(x=>x.k===meshDim)) meshDim=D[0].k;
  const dim=D.find(x=>x.k===meshDim), selG=dim.g(meshSel);
  $("p-mesh-dim").innerHTML=D.map(x=>'<button type="button" data-v="'+x.k+'" aria-pressed="'+(x.k===meshDim)+'">'+t("par.mesh.colorBy", { dim: esc(x.n) })+'</button>').join("");
  let h="";
  for(let lo=0; lo<N; lo+=dom){ h+='<div class="node"><span class="nl">'+t("par.mesh.domain", { i: lo/dom, kind: topo.domain>8 ? t("par.mesh.nvl72") : t("par.mesh.hgx") })+'</span>';
    for(let k=lo;k<Math.min(N,lo+dom);k++){ const gi=dim.g(k), peer=gi===selG;
      h+='<button type="button" class="g'+(k===meshSel?" sel":"")+(peer?" peer":"")+'" data-k="'+k+'" style="background:'+css(dim.c)+';opacity:'+(peer?1:(gi%2?0.38:0.2))+'" aria-label="'+t("par.mesh.gpuAria", { k, dim: esc(dim.n), g: gi })+'">'+(N<=128?k:"")+'</button>'; }
    h+='</div>'; }
  $("p-mesh").innerHTML=h;
  const peers=[]; for(let k=0;k<N;k++) if(dim.g(k)===selG) peers.push(k);
  $("p-mesh-cap").innerHTML=t("par.mesh.cap", { k: meshSel, dim: esc(dim.n), peers: esc(peers.length>16?peers.slice(0,8).join(", ")+" … "+peers.slice(-4).join(", "):peers.join(", ")), n: peers.length });
  const cls=k=>r.comm.filter(x=>x.cls===k&&!x.perStage), fs=r.par.fsdp;
  const row=(x,exp,note)=>{ const l=Par.lk(hw,topo,x.size,x.stride); return '<tr><td>'+esc(x.n)+'</td><td class="num">'+x.size+'</td><td class="num">'+x.stride+'</td><td>'+l.kind+'</td><td class="num">'+(exp!=null?fT(exp):"—")+'</td><td>'+note+(PAR_LINK[x.k]?' <a href="'+PAR_LINK[x.k]+'">'+t("par.groups.more")+'</a>':'')+'</td></tr>'; };
  const sc=k=>sum(cls(k),x=>x.exp||0);
  $("p-groups").innerHTML='<thead><tr><th>'+t("par.groups.th.dim")+'</th><th>'+t("par.groups.th.size")+'</th><th>'+t("par.groups.th.stride")+'</th><th>'+t("par.groups.th.link")+'</th><th>'+t("par.groups.th.exposed")+'</th><th>'+t("par.groups.th.note")+'</th></tr></thead><tbody>'+D.map(x=>{
    if(x.k==="tp") return row(x,sc("tp"),t("par.groups.note.tp"));
    if(x.k==="cp") return row(x,sc("cp"),r.par.imbCP>1 ? t("par.groups.note.cpImb", { meth: esc(Par.CP[r.par.cpMeth].name), x: r.par.imbCP.toFixed(2) }) : esc(Par.CP[r.par.cpMeth].name));
    if(x.k==="ep") return row(x,sc("ep"),r.par.ep ? (r.par.ep.ib>0 ? t("par.groups.note.epIb", { be: esc(Par.EP[r.par.ep.be].name), ib: fB(r.nM*r.par.ep.ib) }) : esc(Par.EP[r.par.ep.be].name)) : t("par.groups.note.epMg", { disp: esc(S.trn.dispatcher) }));
    if(x.k==="fsdp") return row(x,null,t("par.groups.note.fsdp", { twice: fs.aG===2 ? t("par.groups.note.fsdpTwice") : "", exp: fT(r.fsdpExposedMB) }));
    if(x.k==="efsdp") return row(x,null,t("par.groups.note.efsdp"));
    if(x.k==="rep") return row(x,r.tDPexp,t("par.groups.note.rep"));
    if(x.k==="dp") return row(x,r.tDPexp,t("par.groups.note.dp"));
    if(x.k==="edp") return row(x,null,t("par.groups.note.edp"));
    return row(x,sc("pp"),t("par.groups.note.pp"));
  }).join("")+'</tbody>';
}
/* ---------- “并行”页 ---------- */
const moeLayerT = r => { const P=layerPieces(r), tE=P.exp.f+P.exp.b+P.exp.rc, fs=r.par.fsdp, eF=fs&&r.nM ? fs.tExp*r.pp/r.nM : 0;
  return (r.par.ep ? r.par.ep.t : tE) + Math.max(0, eF - tE); };            // 一层 MoE F+B，加上 expert FSDP 盖不住的部分
const attnLayerT = r => { const P=layerPieces(r), cp=r.par.cpFull; return P.core.f+P.core.b+P.core.rc + (cp ? r.par.docs*cp.exposed : 0); };

/* FSDP / CP / EP 一次看一个。三节都照常渲染，只切显示；地址栏跟着当前这一个，#train/par/ep 这样的链接直接打开它 */
const PAR_DIMS=["fsdp","cp","ep"]; let parDim="fsdp";
function setParDim(v){ parDim=v;
  PAR_DIMS.forEach(k=>{ $("par-"+k).hidden=k!==v; });
  xSeg("p-dim",PAR_DIMS.map(k=>[k,t("par.dims."+k)]),v,pickParDim); }
function pickParDim(v){ setParDim(v);
  try{ history.replaceState(null,"","#train/par/"+v); }catch(e){}
  const el=$("par-"+v);   // 切换条已经吸住时，新的一节从头开始读；没吸住就不动页面
  if(el.getBoundingClientRect().top<$("par-dims").getBoundingClientRect().bottom-1) el.scrollIntoView({block:"start"}); }
function renderPar(){
  const {m,d,hw,topo,g}=ctx(), r=runP(), c=parBase(), mgNow=S.trn.fw==="megatron";
  setWarn(r.warn.concat(runFit(r)?[]:[t("par.warn.mem", { used: fB(r.mem.used), cap: hw.mem })]));
  const fs=r.par.fsdp;
  $("p-lede").innerHTML=(mgNow?t("par.lede.mg", { tp: S.trn.tp, cp: S.trn.cp, ep: S.trn.ep, be: esc(Par.EP[c.dispatcher].name) }):'')+
    t("par.lede.main", { model: esc(m.short), hw: hw.name, topo: esc(topo.name), tok: V(fN(r.Tc)), S: fK(c.S), docs: fN(r.par.docs), fs: fs.fs,
      hsdp: fs.R>1 ? t("par.lede.hsdp", { R: fs.R }) : "",
      cp: r.cp>1 ? t("par.lede.cpOn", { cp: r.cp, meth: Par.CP[r.par.cpMeth].name, lb: r.par.cpMeth!=="ulysses" ? (r.par.cpLB ? t("par.lede.cpLb") : t("par.lede.cpCont")) : "" }) : t("par.lede.cpOff"),
      ep: m.moe.E ? (r.ep>1 ? t("par.lede.epOn", { ep: r.ep, be: Par.EP[r.par.ep.be].name, fe: fs.fe }) : t("par.lede.epOff", { fe: fs.fe })) : t("par.lede.dense") })+
    t("par.lede.rerun");
  kpis($("p-kpis"),[
    [t("par.kpi.step"),fT(r.stepT),t("par.kpi.stepNote")],
    [t("par.kpi.thru"),fN(r.tokPerGpu)+" tok/s/GPU","MFU "+(r.mfu*100).toFixed(1)+"%"],
    [t("par.kpi.cpExp"),fT(sum(r.comm.filter(x=>x.cls==="cp"),x=>x.exp||0)),r.cp>1?t("par.kpi.wire", { t: fT(sum(r.comm.filter(x=>x.cls==="cp"),x=>x.t)) }):t("par.kpi.cpOff")],
    [t("par.kpi.epExp"),fT(sum(r.comm.filter(x=>x.cls==="ep"),x=>x.exp||0)),r.par.ep?t("par.kpi.epSync", { t: fT(r.nM*r.par.ep.tSync) }):t("par.kpi.epOff")],
    [t("par.kpi.fsdpExp"),fT(r.fsdpExposedMB),t("par.kpi.fsdpNote", { wire: fT(r.fsdpMB), first: fT(fs.firstAG) })],
    [t("par.kpi.mem"),fB(r.mem.used),t("par.kpi.memCap", { cap: hw.mem }),!runFit(r)]]);
  renderParRoof(r);
  const N=topo.gpus/(r.tp*r.pp), Nep=topo.gpus/r.pp;
  if(m.moe.E){ const Es=pow2upTo(Nep).filter(v=>m.moe.E%v===0 && v>=r.tp && v>1);
    heat("p-heat-ep","p-heat-ep-legend","EP ",Es,Object.keys(Par.EP),(E,S2,be)=>runP({ep:E,S:S2,tok:Math.max(256,S2/r.cp),dispatcher:be}),k=>Par.EP[k].col,k=>Par.EP[k].short); }
  else { $("p-heat-ep").innerHTML='<text x="340" y="60" text-anchor="middle" class="lbl">'+t("par.noEp")+'</text>'; $("p-heat-ep-legend").innerHTML=""; }
  const Cs=pow2upTo(Math.min(N,64)).filter(v=>v>1);
  heat("p-heat-cp","p-heat-cp-legend","CP ",Cs,Object.keys(Par.CP),(C,S2,meth)=>runP({cp:C,S:S2,tok:Math.max(256,S2/C),cpType:meth}),k=>Par.CP[k].col,k=>Par.CP[k].name);
  renderParFsdp(r); renderParCp(r); renderParEp(r); setParDim(parDim);
}

/* 屋顶线的第二个视图：每个算子分三遍，只画 HBM（效率系数和各精度的拐点都算在内），和“一个 step”的逐算子是同一次 Train.run */
let roofView="agg";
function renderRoofPasses(){
  const {hw,g}=ctx(), r=trnRun();
  const PR=Train.PREC[S.trn.prec]||Train.PREC.bf16, pks=[{v:peakOf(hw,"bf16"),n:"BF16"}];
  for(const d of [PR.f,PR.b]) if(d!=="bf16" && !pks.some(p=>p.v===peakOf(hw,d))) pks.push({v:peakOf(hw,d),n:Common.peakKey(d).toUpperCase()});
  const pts=[], topT=new Set([...r.ops].sort((a,b)=>b.fwd.t-a.fwd.t).slice(0,8));
  for(const o of r.ops){ const col=GROUPC[o.group];
    if(o.fwd.fl>0) pts.push({x:o.fwd.I,y:o.fwd.fl/o.fwd.t,color:col,label:topT.has(o)?shortName(o.name):null});
    if(o.dgrad.fl>0) pts.push({x:o.dgrad.I,y:o.dgrad.fl/o.dgrad.t,color:col,shape:"sq",r:4});
    if(o.wgrad.fl>0) pts.push({x:o.wgrad.I,y:o.wgrad.fl/o.wgrad.t,color:col,shape:"tri",r:4}); }
  plot($("p-roof"),{h:380,x0:0.1,x1:20000,y0:1e10,y1:Math.max(...pks.map(p=>p.v))*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:t("par.roof.pass.xl"),yl:t("par.roof.pass.yl"),
    series:roofSeries(hw,g,pks,0.1,20000),points:pts,vlines:pks.map((p,i)=>({x:Common.ridge(p.v,hw,g),color:css("muted"),label:t("par.roof.pass.knee",{peak:p.n,x:fI(Common.ridge(p.v,hw,g))}),dy:i*14}))});
  $("p-roof-legend").innerHTML='<span>'+t("par.roof.pass.legendShapes")+'</span>'+Object.entries({attn:t("par.roof.pass.group.attn"),lin:t("par.roof.pass.group.lin"),moe:t("par.roof.pass.group.moe"),lm:t("par.roof.pass.group.lm"),misc:t("par.roof.pass.group.misc")}).map(([k,v])=>'<span><span class="sw" style="background:'+GROUPC[k]+'"></span>'+v+'</span>').join("");
}
/* 两条屋顶线：计算算子按 HBM 强度，通信按“能盖它的 FLOP / 线上字节” */
function renderParRoof(r){
  xSeg("p-roof-view",[["agg",t("par.roof.view.agg")],["pass",t("par.roof.view.pass")]],roofView,v=>{ roofView=v; renderParRoof(r); });
  $("p-roof-lede").textContent=t(roofView==="pass"?"par.roof.ledePass":"par.b3.lede");
  if(roofView==="pass"){ renderRoofPasses(); return; }
  const {hw,topo}=ctx(), svg=$("p-roof"), W=680, H=360, m={l:64,r:18,t:16,b:44};
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  const pk=peakOf(hw,(Train.PREC[S.trn.prec]||Train.PREC.bf16).f);
  const roofs=[{k:"HBM",bw:hw.hbm*1e9,col:"mem"},{k:"NVLink",bw:hw.nvl*1e9,col:"accent"}];
  const agg = pred => { let fl=0,by=0,t=0; for(const o of r.ops) if(pred(o)) for(const p of ["fwd","dgrad","wgrad"]){ fl+=o[p].fl; by+=o[p].by; t+=o[p].t; if(p==="fwd"&&o.rc){ fl+=o.fwd.fl; by+=o.fwd.by; t+=o.fwd.t; } } return {fl,by,t}; };
  const groups=[[t("par.roof.op.core"),o=>o.key==="acore","c-attn"],[t("par.roof.op.lin"),o=>o.key==="lcore","c-lin"],[t("par.roof.op.exp"),o=>["moe1","act","moe2"].includes(o.key),"c-moe"],
    [t("par.roof.op.proj"),o=>(o.group==="attn"&&o.key!=="acore")||(o.group==="lin"&&o.key!=="lcore")||["dense","shared","router","latd","latu"].includes(o.key),"c-lm"],
    [t("par.roof.op.lm"),o=>o.group==="lm","c-ar"],[t("par.roof.op.misc"),o=>o.group==="misc"||["permute","unperm"].includes(o.key),"c-oth"]];
  const ops=groups.map(([name,pred,col])=>Object.assign({name,col},agg(pred))).filter(o=>o.fl>0&&o.by>0&&o.t>0);
  const tot=agg(()=>true), core=agg(o=>o.key==="acore"), ex=agg(o=>["moe1","act","moe2"].includes(o.key)), fs=r.par.fsdp, comm=[];
  const addC=(name,fl,by,lk,col)=>{ if(fl>0&&by>0&&lk) comm.push({name,fl,by,lk,col}); };
  const cpB=sum(r.comm.filter(x=>x.cls==="cp"),x=>x.bytes), epB=sum(r.comm.filter(x=>x.key==="a2a"),x=>x.bytes), tpB=sum(r.comm.filter(x=>x.cls==="tp"),x=>x.bytes);
  if(tpB) addC(t("par.roof.comm.tp"),tot.fl,tpB,Common.link(hw,topo,r.tp,1),"c-ar");
  if(cpB && r.par.cpFull) addC(t("par.roof.comm.cp", { meth: Par.CP[r.par.cpMeth].name }),core.fl,cpB,r.par.cpFull.lk,"c-cp");
  if(epB) addC(t("par.roof.comm.ep", { be: r.par.ep?Par.EP[r.par.ep.be].short:S.trn.dispatcher }),ex.fl,epB,r.par.ep?r.par.ep.lk:Common.link(hw,topo,r.ep,r.etp),"c-a2a");
  if(fs && fs.bDense) addC(t("par.roof.comm.fsdp"),tot.fl/r.pp,fs.bDense,fs.lkD,"c-fsdp");
  if(fs && fs.bExp) addC(t("par.roof.comm.efsdp"),ex.fl/r.pp,fs.bExp,fs.lkE,"c-fsdp");
  const X=axis(0.5,1e6,m.l,W-m.r,true), Y=axis(pk/2e4,pk*1.6,H-m.b,m.t,true), clampX=I=>Math.min(Math.max(I,0.6),9e5), clampY=y=>Math.max(y,pk/1.9e4);
  let s="";
  for(const v of [1,10,100,1e3,1e4,1e5,1e6]) s+='<line class="gr" x1="'+X(v)+'" x2="'+X(v)+'" y1="'+m.t+'" y2="'+(H-m.b)+'"/><text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+fN(v)+'</text>';
  for(const v of logTicks(pk/2e4,pk*1.6,10)) s+='<line class="gr" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+fN(v/1e12)+'</text>';
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+t("par.roof.xLab")+'</text><text x="12" y="'+(m.t-4)+'">TFLOP/s</text>';
  for(const rf of roofs){ const ridge=pk/rf.bw;
    s+='<path d="M'+X(0.5)+','+Y(Math.max(pk/2e4,0.5*rf.bw))+' L'+X(Math.min(ridge,1e6))+','+Y(Math.min(pk,Math.min(ridge,1e6)*rf.bw))+' L'+X(1e6)+','+Y(pk)+'" fill="none" style="stroke:'+css(rf.col)+'" stroke-width="2"/>';
    s+='<line x1="'+X(ridge)+'" x2="'+X(ridge)+'" y1="'+Y(pk)+'" y2="'+(H-m.b)+'" style="stroke:'+css(rf.col)+';stroke-dasharray:3 4;opacity:.55"/>'; }
  for(const o of ops){ const I=o.fl/o.by;
    s+='<circle cx="'+X(clampX(I))+'" cy="'+Y(clampY(Math.min(pk,o.fl/o.t)))+'" r="5.5" style="fill:'+css(o.col)+';stroke:var(--panel)" stroke-width="2" '+tipAttr(o.name,[{v:fN(I)+" FLOP/B",l:t("par.roof.tip.hbm")},{v:fN(o.fl/o.t/1e12)+" TFLOP/s",l:t("par.roof.tip.actual")},{v:fT(o.t),l:t("par.roof.tip.perMb")}])+'/>'; }
  const badges=[];
  comm.forEach((o,i)=>{ const I=o.fl/o.by, bw=o.lk.bw, y=Math.min(pk,I*bw), xx=X(clampX(I)), yy=Y(clampY(y)); o.no=i+1;
    s+='<path d="M'+xx+','+(yy-7)+' L'+(xx+7)+','+yy+' L'+xx+','+(yy+7)+' L'+(xx-7)+','+yy+' Z" style="fill:var(--panel);stroke:'+css(o.col)+'" stroke-width="2.2" '+tipAttr(o.name,[{v:fN(I)+" FLOP/B",l:t("par.roof.tip.cover")},{v:o.lk.intra?"NVLink":"IB",l:t("par.roof.tip.link")},{v:I*bw>=pk?t("par.roof.tip.hidden"):t("par.roof.tip.exposed", { p: pc(1-I*bw/pk) }),l:""}])+'/>';
    const near=badges.find(b=>Math.hypot(b.x-xx,b.y-yy)<12); if(near) near.t.push(o.no); else badges.push({x:xx,y:yy,t:[o.no]}); });
  for(const b of badges) s+='<text x="'+b.x+'" y="'+(b.y-11)+'" text-anchor="middle" class="lbl" style="pointer-events:none">'+b.t.join(" · ")+'</text>';
  svg.innerHTML=s; bindTips(svg);
  $("p-roof-legend").innerHTML=roofs.map(rf=>'<span><i style="border-color:'+css(rf.col)+'"></i>'+t("par.roof.legend", { k: esc(rf.k), bw: fN(rf.bw/1e9), ridge: fN(pk/rf.bw) })+'</span>').join("")
    +ops.map(o=>'<span><span class="sw" style="background:'+css(o.col)+';border-radius:50%"></span>'+esc(o.name)+'</span>').join("")
    +comm.map(o=>'<span><span class="sw" style="border:2px solid '+css(o.col)+';background:var(--panel);transform:rotate(45deg) scale(.8)"></span>'+o.no+' '+esc(o.name)+'</span>').join("");
}

/* 相图：每格跑一遍 Train.run，颜色是最快的候选 */
const HEAT_S=[8192,16384,32768,65536,131072,262144,524288,1048576];
function heat(svgId, legId, rowLab, rowVals, cands, evalFn, colOf, nameOf){
  const svg=$(svgId), W=680, m={l:62,r:12,t:12,b:40}, rows=rowVals.length, cols=HEAT_S.length, ch=30, H=m.t+m.b+rows*ch;
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  const cw=(W-m.l-m.r)/cols, used=new Set();
  let s='<defs><pattern id="'+svgId+'-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--line)" stroke-width="2"/></pattern></defs>';
  rowVals.forEach((rv,i)=>{ const y=m.t+(rows-1-i)*ch;
    s+='<text x="'+(m.l-6)+'" y="'+(y+ch/2+4)+'" text-anchor="end">'+esc(rowLab+rv)+'</text>';
    HEAT_S.forEach((S2,j)=>{ const x=m.l+j*cw, res=cands.map(k=>({k,r:evalFn(rv,S2,k)}));
      const ok=res.filter(z=>runOK(z.r)&&runFit(z.r)).sort((a,b)=>b.r.tokPerGpu-a.r.tokPerGpu);
      const tip=res.map(z=>({v:runOK(z.r)?(runFit(z.r)?t("par.heat.tok", { n: fN(z.r.tokPerGpu) }):t("par.heat.noFit")):t("par.heat.na"),l:nameOf(z.k),c:colOf(z.k)}));
      if(ok.length){ const w=ok[0]; used.add(w.k);
        s+='<rect x="'+(x+1)+'" y="'+(y+1)+'" width="'+(cw-2)+'" height="'+(ch-2)+'" rx="3" style="fill:'+css(colOf(w.k))+'" '+tipAttr(t("par.heat.tipWin", { row: rowLab+rv, S: fK(S2), name: nameOf(w.k) }),tip)+'/>';
        s+='<text x="'+(x+cw/2)+'" y="'+(y+ch/2+4)+'" text-anchor="middle" style="fill:var(--panel);font-size:10.5px;pointer-events:none">'+esc(fN(w.r.tokPerGpu))+'</text>'; }
      else s+='<rect x="'+(x+1)+'" y="'+(y+1)+'" width="'+(cw-2)+'" height="'+(ch-2)+'" rx="3" style="fill:url(#'+svgId+'-hatch)" '+tipAttr(t("par.heat.tipCell", { row: rowLab+rv, S: fK(S2) }),tip)+'/>'; }); });
  HEAT_S.forEach((S2,j)=>{ s+='<text x="'+(m.l+j*cw+cw/2)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+fK(S2)+'</text>'; });
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+t("par.heat.xLab")+'</text>';
  svg.innerHTML=s; bindTips(svg);
  legendInto(legId,cands.map(k=>({name:used.has(k)?nameOf(k):t("par.heat.noWin", { name: nameOf(k) }),col:colOf(k)})).concat([{name:t("par.heat.hatch"),col:"line"}]));
}

/* ---------- FSDP ---------- */
function renderParFsdp(r){
  const {m,hw,topo,g}=ctx(), fs=r.par.fsdp, P=layerPieces(r), moe=m.moe.E>0;
  $("p-f-formula").textContent=
t("par.fsdp.f.bytes", { aG: fs.aG, rs: fs.reshard ? t("par.fsdp.f.on") : t("par.fsdp.f.off"), gB: fs.gB, dt: fs.gB===4?"FP32":"BF16" })+"\n"+
t("par.fsdp.f.dense", { p: fN(fs.pFull), lin: fs.pLin ? t("par.fsdp.f.lin", { p: fN(fs.pLin) }) : "", F: fs.fs, link: fs.lkD.kind, bytes: fB(fs.bDense), time: fT(fs.tDense) })+"\n"+
(moe?t("par.fsdp.f.exp", { p: fN(fs.pExpL), F: fs.fe, rest: fs.fe>1 ? t("par.fsdp.f.expOn", { link: fs.lkE.kind, bytes: fB(fs.bExp), time: fT(fs.tExp) }) : t("par.fsdp.f.expOff") })+"\n":"")+
(fs.R>1?t("par.fsdp.f.hsdp", { R: fs.R, link: fs.lkR.kind, time: fT(r.par.tAR) })+"\n":"")+
t("par.fsdp.f.states", { p: fN(r.Pshard), bytes: fB(r.mem.states) });
  /* 数据流示意 */
  const Fs=Math.min(fs.fs,8), x0=20, bw=120, gap=26, box=(x,y,w,h,col,txt,op)=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="4" style="fill:'+css(col)+';opacity:'+(op??1)+'"/>'+(txt?'<text x="'+(x+w/2)+'" y="'+(y+h/2+4)+'" text-anchor="middle" style="fill:var(--panel);font-size:10.5px">'+esc(txt)+'</text>':''), lab=(x,y,txt)=>'<text x="'+x+'" y="'+y+'" class="lbl">'+esc(txt)+'</text>';
  let s=lab(x0,16,t("par.fsdp.flow.resident", { F: fs.fs }));
  for(let i=0;i<Fs;i++) s+=box(x0,26+i*24,bw,18,"c-fsdp",t("par.fsdp.flow.shard", { i }),0.9);
  if(fs.fs>8) s+=lab(x0,26+8*24+12,t("par.fsdp.flow.more", { F: fs.fs }));
  const x1=x0+bw+gap+40; s+=lab(x1,16,t("par.fsdp.flow.gather"));
  for(let i=0;i<Fs;i++) s+=box(x1,26+i*24,bw,18,"c-lm","",0.85);
  s+='<text x="'+(x1+bw/2)+'" y="'+(26+Math.floor(Fs/2)*24+13)+'" text-anchor="middle" style="fill:var(--panel);font-size:11px">'+t("par.fsdp.flow.full")+'</text>';
  const x2=x1+bw+gap; s+=lab(x2,16,t("par.fsdp.flow.compute"))+box(x2,26,bw,Fs*24-6,"c-attn",t("par.fsdp.flow.local", { tok: fN(r.Tc) }),0.9);
  const x3=x2+bw+gap; s+=lab(x3,16,t("par.fsdp.flow.rs"));
  for(let i=0;i<Fs;i++) s+=box(x3,26+i*24,bw-10,18,"c-a2a",t("par.fsdp.flow.sum", { i }),0.75);
  const yb=26+Fs*24+(fs.fs>8?24:0)+14;     // 说明分两行放在方块下面，图的高度跟着分片数走
  s+='<text x="'+x0+'" y="'+yb+'">'+t("par.fsdp.flow.note1", { ag: fB(fs.pFull*2), bwd: fs.reshard ? t("par.fsdp.flow.regather") : t("par.fsdp.flow.reuse"), rs: fB(fs.pFull*fs.gB) })+'</text><text x="'+x0+'" y="'+(yb+16)+'">'+t("par.fsdp.flow.note2")+'</text>';
  for(const xa of [x0+bw,x1+bw,x2+bw]) s+='<path d="M'+(xa+4)+','+(26+Fs*12)+' l'+(gap-10)+',0" style="stroke:var(--muted)" stroke-width="1.5" marker-end="url(#p-f-arr)"/>';
  $("p-f-flow").setAttribute("viewBox","0 0 680 "+(yb+24));
  $("p-f-flow").innerHTML='<defs><marker id="p-f-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" style="fill:var(--muted)"/></marker></defs>'+s;
  /* 连续几层的前向和预取 */
  const ffn=moe?P.rep.f+P.exp.f:P.den.f, compFull=P.pre.f+P.core.f+P.post.f+ffn, compLin=(m.lin.n?P.lin.f/m.lin.n*m.L/m.L:0)+ffn;
  const linPer=m.lin.n? (()=>{ let f=0; for(const o of r.ops) if(o.group==="lin") f+=o.fwd.t/o.layers; return f; })() : 0;
  const lane=fs.lkD.intra?"nvl":"ib", LP=Common.layerPlan(m), isFull=i=>LP[i].attn!=="lin", gT=i=>Par.wireT((isFull(i)?fs.pFull:fs.pLin)*(fs.fs-1)/fs.fs*2,fs.lkD,g.etaAg,1,g);
  const layers=Math.min(6,m.L), segs=[]; let commEnd=gT(0), tc=commEnd;
  if(commEnd>0) segs.push({lane,label:t("par.fsdp.pref.gather", { i: 0 }),t0:0,t1:commEnd,cat:"fsdp"});
  for(let i=0;i<layers;i++){ const comp=isFull(i)?compFull:linPer+ffn, start=Math.max(tc,commEnd);
    if(i+1<layers){ const nx=gT(i+1), cs=Math.max(commEnd,start); if(nx>0) segs.push({lane,label:t("par.fsdp.pref.prefetch", { i: i+1 }),t0:cs,t1:cs+nx,cat:"fsdp"}); commEnd=cs+nx; }
    segs.push({lane:"sm",label:t("par.fsdp.pref.fwd", { i, kind: t("par.fsdp.pref."+LP[i].attn) }),t0:start,t1:start+comp,cat:isFull(i)?"attn":"lin"}); tc=start+comp; }
  gantt("p-f-pref",[{label:"",lanes:["sm",lane],segs}]);
  $("p-f-pref-lede").textContent=t("par.fsdp.pref.lede", { n: layers, full: fT(compFull), lin: m.lin.n ? t("par.fsdp.pref.ledeLin", { t: fT(linPer+ffn) }) : "", gather: fT(gT(Math.max(0,LP.findIndex(x=>x.attn!=="lin")))), link: fs.lkD.kind })+(fs.fs<=1?t("par.fsdp.pref.noGather"):"");
  legendInto("p-f-pref-legend",[{name:t("par.fsdp.pref.legFull"),col:CAT.attn},{name:t("par.fsdp.pref.legLin"),col:CAT.lin},{name:t("par.fsdp.pref.legAg"),col:CAT.fsdp}]);
  /* 每卡 token：两种拓扑各跑一遍 Train.run（各用自己那一代的卡：HGX 用 B200 / B300 / H200，NVL72 用 GB200 / GB300；PCIe 卡在自己的 8 卡服务器上，对照 GB200） */
  const PAIR={B200:"GB200",GB200:"B200",B300:"GB300",GB300:"B300",H200:"GB200",H100:"GB200",RTXP6000:"GB200"}, gb=/^GB/.test(S.hw), hH=gb?PAIR[S.hw]:S.hw, hN=gb?S.hw:PAIR[S.hw];
  const runN=(tk,hk,n)=>{ const c2=fitPar(Object.assign(parBase(),{tok:n,S:Math.max(n*S.trn.cp,1024)}),TOPO[tk].gpus); return { c2, r2:Train.run(m,HW[hk],TOPO[tk],Object.assign({N:TOPO[tk].gpus},c2),g) }; };
  const deg=(tk,hk)=>{ const {c2,r2}=runN(tk,hk,r.Tc), f=r2.par.fsdp; return { fs:f?f.fs:1, fe:f?f.fe:1, tp:c2.tp, cp:c2.cp, ep:c2.ep }; };   // 这套配置放到这个拓扑上实际的 FSDP 路数（TP / CP / EP 可能被缩小）
  const tH=Common.homeTopo(hH), dH=deg(tH,hH), dN=deg("nvl64",hN), dTxt=d=>t("par.fsdp.n.deg", { fs: d.fs, tp: d.tp, cp: d.cp, ep: d.ep });
  const scen=[[tH,hH,t(tH==="pcie8"?"par.fsdp.n.pcie":"par.fsdp.n.hgx", { hw: HW[hH].name, deg: dTxt(dH) }),"s1","8 × "+HW[hH].name],["nvl64",hN,t("par.fsdp.n.nvl", { hw: HW[hN].name, deg: dTxt(dN) }),"s2","64 × "+HW[hN].name]];
  const ns=[512,1024,2048,4096,8192,16384,32768,65536,131072];
  const ser=scen.map(([tk,hk,name,col,short])=>({name,short,col,pts:ns.map(n=>{ const {r2}=runN(tk,hk,n);
    return [n, runOK(r2) ? r2.fsdpExposedMB/r2.tMB : null, {hollow:!runFit(r2)}]; })}));
  lineChart("p-f-n",{series:ser,xLog:true,xBase:2,xFmt:fK,yFmt:pc,xLab:t("par.fsdp.n.xLab"),yLab:t("par.fsdp.n.yLab"),vlines:[{x:r.Tc,label:t("par.chart.now")}],yMin:0});
  legendInto("p-f-n-legend",ser.map(se=>({name:se.name,col:se.col,line:true})));
  const pkb=peakOf(hw,"bf16"), rN=Common.linkRidge(pkb,hw.nvl,g), rI=Common.linkRidge(pkb,hw.ib,g);
  $("p-f-n-lede").textContent=t("par.fsdp.n.lede", { hw: hw.name, rN: fN(rN), ratio: fN(rI/rN), f1: dH.fs, f2: dN.fs, b1: (dH.fs-1)+"/"+dH.fs, b2: (dN.fs-1)+"/"+dN.fs });
  const rep=moe ? t("par.fsdp.tbl.rep") : "";
  $("p-f-tbl").innerHTML='<thead><tr><th>'+t("par.fsdp.tbl.th.kind")+'</th><th>'+t("par.fsdp.tbl.th.layers")+'</th><th>'+t("par.fsdp.tbl.th.params")+'</th><th>'+t("par.fsdp.tbl.th.group")+'</th><th>'+t("par.fsdp.tbl.th.bytes")+'</th><th>'+t("par.fsdp.tbl.th.wire")+'</th><th>'+t("par.fsdp.tbl.th.fwd")+'</th></tr></thead><tbody>'+
    [[t("par.fsdp.tbl.full", { rep }),m.full.n,fs.pFull,fs.fs,fs.lkD,compFull],...(m.lin.n?[[t("par.fsdp.tbl.lin", { rep }),m.lin.n,fs.pLin,fs.fs,fs.lkD,linPer+ffn]]:[]),...(m.dense.n?[[t("par.fsdp.tbl.dense"),m.dense.n,fs.pDen,fs.fs,fs.lkD,P.den.f]]:[]),...(moe?[[t("par.fsdp.tbl.exp"),r.nM,fs.pExpL,fs.fe,fs.lkE,P.exp.f]]:[])]
    .map(([k,nL,Pp,F,l,cf])=>{ const by=Par.fsdpBytes(Pp,F,fs.aG,fs.gB); return '<tr><td>'+esc(k)+'</td><td class="num">'+nL+'</td><td class="num">'+fN(Pp)+'</td><td class="num">'+(F>1?t("par.fsdp.tbl.groupLink", { F, link: l.kind }):F)+'</td><td class="num">'+(F>1?fB(by):"—")+'</td><td class="num">'+(F>1?fT(Par.fsdpT(Pp,F,l,g,fs.aG,fs.gB)):"—")+'</td><td class="num">'+fT(cf)+'</td></tr>'; }).join("")+'</tbody>';
}

/* ---------- CP ---------- */
function cpView(r){ const {topo}=ctx(); if(r.cp>1) return {};
  const C=Math.min(8, topo.gpus/(r.tp*r.pp)); return { cp:C, tok:Math.max(256,S.trn.S/C) }; }
function renderParCp(r){
  const {m,d,topo}=ctx(), V0=cpView(r), rv=Object.keys(V0).length?runP(V0):r, C=rv.cp, c=parBase();
  /* causal 块矩阵 */
  const Cv=Math.min(Math.max(C,2),16), size=280, x0=30, y0=24, cell=size/(2*Cv), uly=rv.par.cpMeth==="ulysses";
  let s='<text x="'+x0+'" y="14" class="lbl">'+t("par.cp.contSplit")+'</text><text x="'+(x0+size+60)+'" y="14" class="lbl">'+(uly?t("par.cp.mat.uly"):t("par.cp.mat.zig"))+'</text>';
  const work={cont:new Array(Cv).fill(0),zig:new Array(Cv).fill(0)}, rk=(i,zig)=>zig?(i<Cv?i:2*Cv-1-i):Math.floor(i/2);
  for(const [ox,zig] of [[x0,false],[x0+size+60,true]]) for(let i=0;i<2*Cv;i++) for(let j=0;j<=i;j++){ const k=rk(i,zig), diag=i===j; work[zig?"zig":"cont"][k]+=diag?0.5:1;
    s+='<rect x="'+(ox+j*cell)+'" y="'+(y0+i*cell)+'" width="'+(cell-1)+'" height="'+(cell-1)+'" style="fill:'+css(uly&&zig?"c-attn":"s"+(k%6+1))+';opacity:'+(diag?0.55:0.9)+'" '+tipAttr(t("par.cp.mat.tip", { i, j }),[{v:"rank "+k,l:zig?t("par.cp.zigzag"):t("par.cp.cont")}])+'/>'; }
  $("p-c-mat").innerHTML=s; bindTips($("p-c-mat"));
  const mx=Math.max(...work.cont), avg=a=>a.reduce((x,y)=>x+y,0)/Cv, bh=Math.min(16,270/Cv-3);
  let s2='<text x="20" y="14" class="lbl">'+t("par.cp.work.title")+'</text>';
  for(let k=0;k<Cv;k++){ const y=28+k*(bh+3);
    s2+='<text x="54" y="'+(y+bh-3)+'" text-anchor="end">rank '+k+'</text><rect x="60" y="'+y+'" width="'+(work.cont[k]/mx*260)+'" height="'+(bh/2-1)+'" rx="2" style="fill:var(--s2)" '+tipAttr("rank "+k,[{v:work.cont[k].toFixed(1),l:t("par.cp.contSplit"),c:"s2"},{v:work.zig[k].toFixed(1),l:t("par.cp.zigzag"),c:"s1"}])+'/><rect x="60" y="'+(y+bh/2)+'" width="'+(work.zig[k]/mx*260)+'" height="'+(bh/2-1)+'" rx="2" style="fill:var(--s1)"/>'; }
  s2+='<text x="340" y="40" class="lbl">'+t("par.cp.work.ratio")+'</text><text x="340" y="58">'+t("par.cp.work.cont", { v: (mx/avg(work.cont)).toFixed(3), f: (2-1/Cv).toFixed(3) })+'</text><text x="340" y="76">'+t("par.cp.work.zig", { v: (Math.max(...work.zig)/avg(work.zig)).toFixed(3) })+'</text><text x="340" y="94">'+t("par.cp.work.uly")+'</text>';
  s2+='<rect x="340" y="110" width="12" height="8" style="fill:var(--s2)"/><text x="358" y="118">'+t("par.cp.contSplit")+'</text><rect x="340" y="126" width="12" height="8" style="fill:var(--s1)"/><text x="358" y="134">'+t("par.cp.zigzag")+'</text>';
  $("p-c-work").innerHTML=s2; bindTips($("p-c-work"));
  $("p-c-mat-lede").textContent=t("par.cp.mat.lede", { n: 2*Cv, C: Cv, cap: C>16 ? t("par.cp.mat.cap") : "" })+(S.trn.fw==="megatron"?t("par.cp.mat.mg"):t("par.cp.mat.lb", { state: c.cpLB!==false ? t("par.cp.mat.on") : t("par.cp.mat.off"), x: rv.par.imbCP.toFixed(3) }));
  /* 三种方法 */
  const runs=Object.keys(Par.CP).map(k=>({k,r:runP(Object.assign({},V0,{cpType:k}))}));
  gantt("p-c-gantt",runs.map(({k,r:r2})=>seqGroup(r2,"attn",t("par.cp.gantt.label", { meth: Par.CP[k].name, t: fT(layerSeq(r2,"attn")&&seqGroup(r2,"attn","").end), why: r2.par.cpFull&&!r2.par.cpFull.ok ? t("par.cp.gantt.why", { why: r2.par.cpFull.why }) : "" }))));
  legendInto("p-c-gantt-legend",[[t("par.cp.leg.attn"),"attn"],[t("par.cp.leg.proj"),"proj"],[t("par.cp.leg.cp"),"cp"],[t("par.cp.leg.tp"),"tp"]].map(([n,k])=>({name:n,col:CAT[k]})));
  $("p-c-gantt-lede").textContent=(Object.keys(V0).length?t("par.cp.gantt.demo", { C, tok: fN(rv.Tc) }):"")+t("par.cp.gantt.lede");
  const ac=rv.ops.some(o=>o.rc), kv=d.kvTokElems, tbl=$("p-c-tbl");
  tbl.innerHTML='<thead><tr><th>'+t("par.cp.tbl.th.meth")+'</th><th>'+t("par.cp.tbl.th.fb", { rc: ac ? t("par.cp.tbl.rcFull") : t("par.cp.tbl.rcNone") })+'</th><th>'+t("par.cp.tbl.th.layer")+'</th><th>'+t("par.cp.tbl.th.calls")+'</th><th>'+t("par.cp.tbl.th.imb")+'</th><th>'+t("par.cp.tbl.th.wire")+'</th></tr></thead><tbody>'+
    runs.map(({k,r:r2})=>{ const cp=r2.par.cpFull; if(!cp) return ""; const form=k==="allgather"?(ac?4:3)+"(C−1)·L":k==="ring"?(ac?"(5C−3)·L":"(4C−2)·L"):(ac?6:4)+"·b·(S/C)·(Q+K′)·d·(C−1)/C";
      return '<tr'+(k===rv.par.cpMeth?' class="hl"':'')+'><td>'+esc(Par.CP[k].name)+'</td><td><code>'+esc(form)+'</code></td><td class="num">'+fB(r2.par.docs*cp.wire)+'</td><td class="num">'+cp.calls+'</td><td class="num">'+r2.par.imbCP.toFixed(3)+'×</td><td class="num">'+fT(r2.par.docs*cp.tw)+' / '+fT(r2.par.docs*cp.exposed)+(cp.ok?'':' <span class="tag">'+esc(cp.why)+'</span>')+'</td></tr>'; }).join("")+
    '<tr><td colspan="6">'+t("par.cp.tbl.foot", { kv, what: m.full.kind==="gqa" ? t("par.cp.tbl.gqa", { Hkv: m.full.Hkv, dh: m.full.dh }) : m.full.kind==="csa" ? t("par.cp.tbl.csa", { src: d.nSrc, dh: m.full.dh, idx: m.full.idxD }) : t("par.cp.tbl.mla", { idx: m.full.kind==="dsa" ? t("par.cp.tbl.dsaIdx", { d: m.full.idxD }) : "" }), uly: m.full.kind==="gqa" ? t("par.cp.tbl.ulyGqa", { k: Math.max(m.full.Hkv,C) }) : t("par.cp.tbl.ulyMla") })+'</td></tr></tbody>';
  /* 随 C、随 S、固定世界 */
  const N=topo.gpus/(r.tp*r.pp), Cs=pow2upTo(Math.min(64,N)).filter(v=>v>1);
  const serC=Object.keys(Par.CP).map(k=>({name:Par.CP[k].name,col:Par.CP[k].col,pts:Cs.map(Cx=>{ const r2=runP({cpType:k,cp:Cx,tok:Math.max(64,S.trn.S/Cx)}); return [Cx, runOK(r2)&&r2.par.cpFull ? attnLayerT(r2) : null]; })}));
  lineChart("p-c-vsC",{series:serC,xLog:true,xBase:2,xLab:t("par.cp.vsC.xLab"),yLab:t("par.cp.vsC.yLab"),yLog:true,yFmt:fT});
  legendInto("p-c-vsC-legend",serC.map(se=>({name:se.name,col:se.col,line:true})));
  const Ss=[8192,16384,32768,65536,131072,262144,524288,1048576];
  const serS=Object.keys(Par.CP).map(k=>({name:Par.CP[k].name,col:Par.CP[k].col,pts:Ss.map(S2=>{ const r2=runP(Object.assign({},V0,{cpType:k,S:S2,tok:Math.max(256,S2/C)})); return [S2, runOK(r2)?r2.tokPerGpu:null, {hollow:!runFit(r2),note:runFit(r2)?"":t("par.cp.vsS.noFit")}]; })}));
  lineChart("p-c-vsS",{series:serS,xLog:true,xBase:2,xFmt:fK,xLab:t("par.cp.vsS.xLab"),yLab:t("par.cp.vsS.yLab"),vlines:[{x:S.trn.S,label:t("par.chart.now")}]});
  legendInto("p-c-vsS-legend",serS.map(se=>({name:se.name,col:se.col,line:true})));
  const S0=Math.min(S.trn.S,32768), Cf=pow2upTo(Math.min(64,N));
  const serF=Object.keys(Par.CP).map(k=>({name:Par.CP[k].name,col:Par.CP[k].col,pts:Cf.map(Cx=>{ const r2=runP({cpType:k,cp:Cx,S:S0,tok:S0}); return [Cx, runOK(r2)?r2.stepT:null, {hollow:!runFit(r2)}]; })}));
  lineChart("p-c-fixed",{series:serF,xLog:true,xBase:2,xLab:t("par.cp.fixed.xLab"),yLab:t("par.cp.fixed.yLab"),yFmt:fT});
  legendInto("p-c-fixed-legend",serF.map(se=>({name:se.name,col:se.col,line:true})));
  $("p-c-fixed-lede").textContent=t("par.cp.fixed.lede", { S: fK(S0), note: fixedWorldNote() });
}

/* 匿名实测（数据点库）给两段说明配的数 */
function fixedWorldNote(){ const at=(meth,cp)=>DB.one("cp_fixed_world",{model:"qwen35",seq:32768,method:meth,cp}); const b=at("single",1);
  if(!b) return ""; const ms=p=>p?Math.round(p.fb_ms)+" ms":"—";
  return t("par.note.fixedWorld", { single: ms(b), uly: ms(at("ulysses",8)), ag: ms(at("allgather",8)), ring: ms(at("ring",8)) }); }
function crossNote(){ const p=DB.one("efsdp_crossover",{case:"same_node"}); return p ? t("par.note.cross", { n: fInt(p.tokens) }) : ""; }

/* ---------- EP ---------- */
const EP_CARDS = {
  standard:{ h:t("par.ep.card.standard.h"), p:t("par.ep.card.standard.p"), m:t("par.ep.card.standard.m") },
  ag:{ h:t("par.ep.card.ag.h"), p:t("par.ep.card.ag.p"), m:t("par.ep.card.ag.m") },
  deepep:{ h:t("par.ep.card.deepep.h"), p:t("par.ep.card.deepep.p"), m:t("par.ep.card.deepep.m") },
  distmoe:{ h:t("par.ep.card.distmoe.h"), p:t("par.ep.card.distmoe.p"), m:t("par.ep.card.distmoe.m") },
  mok:{ h:t("par.ep.card.mok.h"), p:t("par.ep.card.mok.p"), m:t("par.ep.card.mok.m") } };
function epView(r){ const {topo,m}=ctx(); if(r.ep>1) return {}; const N=topo.gpus/r.pp; let E=Math.min(8,N); while(E>1 && m.moe.E%E) E/=2; return { ep:E }; }
function renderParEp(r){
  const {m,d,hw,topo,g}=ctx(), e=m.moe, ids=["p-e-cards","p-e-props","p-e-gantt","p-e-vsE","p-e-vsN","p-e-cross","p-e-fan"];
  if(!e.E){ $("p-e-formula").textContent=t("par.ep.dense"); ids.forEach(id=>{ const el=$(id); el.innerHTML=el.tagName==="svg"?'<text x="340" y="40" text-anchor="middle" class="lbl">'+t("par.noEp")+'</text>':""; });
    ["p-e-gantt-legend","p-e-vsE-legend","p-e-vsN-legend","p-e-cross-legend"].forEach(id=>$(id).innerHTML=""); $("p-e-gantt-lede").textContent=""; $("p-e-cross-lede").textContent=""; return; }
  const V0=epView(r), rv=Object.keys(V0).length?runP(V0):r, n=rv.Tl, K=e.k, Z=e.E, E=rv.ep, De=d.De, pE=Par.pDest(Z,E,K);
  $("p-e-formula").textContent=(Object.keys(V0).length?t("par.ep.f.demo", { E })+"\n":"")+
t("par.ep.f.cur", { n: fN(n), K, Z, E, De, lat: e.Dlat ? t("par.ep.f.lat") : "" })+"\n"+
t("par.ep.f.slot", { rows: fN(n*K*(1-1/E)), bytes: fB(n*K*(1-1/E)*De*2) })+"\n"+
t("par.ep.f.dedup", { rows: fN(n*(E-1)*pE), p: pE.toFixed(3), bytes: fB(n*(E-1)*pE*De*2) })+"\n"+
t("par.ep.f.bcast", { rows: fN((E-1)*n), bytes: fB((E-1)*n*De*2) })+"\n"+
t("par.ep.f.ratio", { r: (K/(E*pE)).toFixed(2) });
  $("p-e-cards").innerHTML=Object.keys(Par.EP).map(k=>'<div class="card"><h4><span class="dot" style="background:'+css(Par.EP[k].col)+'"></span>'+esc(EP_CARDS[k].h)+'</h4><p>'+esc(EP_CARDS[k].p)+'</p><p class="meta">'+esc(EP_CARDS[k].m)+'</p></div>').join("");
  const all=Object.keys(Par.EP).map(k=>({k,r:runP(Object.assign({},V0,{dispatcher:k}))}));
  $("p-e-props").innerHTML='<thead><tr><th>'+t("par.ep.tbl.th.be")+'</th><th>'+t("par.ep.tbl.th.who")+'</th><th>'+t("par.ep.tbl.th.bytes")+'</th><th>'+t("par.ep.tbl.th.wire")+'</th><th>'+t("par.ep.tbl.th.layer")+'</th><th>'+t("par.ep.tbl.th.sync")+'</th><th>'+t("par.ep.tbl.th.buf")+'</th><th>'+t("par.ep.tbl.th.det")+'</th><th>'+t("par.ep.tbl.th.cross")+'</th><th>'+t("par.ep.tbl.th.ok")+'</th><th></th></tr></thead><tbody>'+
    all.map(({k,r:r2})=>{ const ep=r2.par.ep; return '<tr'+(k===rv.par.ep.be?' class="hl"':'')+'><td>'+esc(Par.EP[k].name)+'</td><td>'+(k==="standard"||k==="ag"?t("par.ep.tbl.host"):t("par.ep.tbl.kernel"))+'</td><td class="num">'+fB(ep.wire)+'</td><td class="num">'+fT(ep.tw)+'</td><td class="num">'+fT(ep.t)+'</td><td class="num">'+ep.syncs+'</td><td class="num">'+fB(ep.buf)+'</td><td>'+(ep.det?t("par.ep.tbl.yes"):t("par.ep.tbl.detNo"))+'</td><td>'+(Par.EP[k].fused?t("par.ep.tbl.no"):t("par.ep.tbl.yes"))+'</td><td class="'+(ep.ok?"ok":"bad")+'">'+(ep.ok?t("par.ep.tbl.yes"):esc(ep.why))+'</td><td>'+(S.trn.fw==="torchtitan"&&k!==S.trn.dispatcher?'<button type="button" class="mini" data-disp="'+k+'">'+t("par.ep.tbl.use")+'</button>':'')+'</td></tr>'; }).join("")+'</tbody>';
  gantt("p-e-gantt",all.map(({k,r:r2})=>seqGroup(r2,"moe",t("par.ep.gantt.label", { be: Par.EP[k].name, t: fT(seqGroup(r2,"moe","").end), na: r2.par.ep.ok ? "" : t("par.ep.gantt.na", { why: r2.par.ep.why }) }))));
  legendInto("p-e-gantt-legend",[[t("par.ep.leg.moe"),"moe"],[t("par.ep.leg.ep"),"ep"],[t("par.ep.leg.proj"),"proj"],[t("par.ep.leg.sync"),"sync"]].map(([nm,k])=>({name:nm,col:CAT[k]})));
  $("p-e-gantt-lede").textContent=t("par.ep.gantt.lede", { n: fN(n), imb: S.trn.imb });
  const N=topo.gpus/r.pp, Es=pow2upTo(N).filter(v=>Z%v===0 && v>=rv.tp);
  const serE=Object.keys(Par.EP).map(k=>({name:Par.EP[k].name,short:Par.EP[k].short,col:Par.EP[k].col,pts:Es.map(Ex=>{ const r2=runP({ep:Ex,dispatcher:k}); return [Ex, runOK(r2)?moeLayerT(r2):null]; })}));
  lineChart("p-e-vsE",{series:serE,xLog:true,xBase:2,xLab:t("par.ep.vsE.xLab"),yLab:t("par.ep.vsE.yLab"),yFmt:fT,yLog:true,vlines:N>Par.pow2Dom(topo)?[{x:Par.pow2Dom(topo),label:t("par.ep.vsE.domain")}]:[]});
  legendInto("p-e-vsE-legend",serE.map(se=>({name:se.name,col:se.col,line:true})));
  const ns=[1024,2048,4096,8192,16384,32768,65536,131072];
  const serN=Object.keys(Par.EP).map(k=>({name:Par.EP[k].name,short:Par.EP[k].short,col:Par.EP[k].col,pts:ns.map(nx=>{ const r2=runP(Object.assign({},V0,{dispatcher:k,tok:nx*rv.tp})); return [nx, runOK(r2)?moeLayerT(r2)/nx*1e6:null]; })}));
  lineChart("p-e-vsN",{series:serN,xLog:true,xBase:2,xFmt:fK,xLab:t("par.ep.vsN.xLab"),yLab:t("par.ep.vsN.yLab"),yFmt:v=>fT(v),vlines:[{x:n,label:t("par.chart.now")}]});
  legendInto("p-e-vsN-legend",serN.map(se=>({name:se.name,col:se.col,line:true})));
  /* 交叉：expert FSDP 权重字节对激活字节 */
  const fs=rv.par.fsdp, blk=rv.ep*fs.fe, rep=rv.ops.some(o=>o.rc);
  const Wb=Ex=>{ const F=blk/Ex; return F>=1 ? rv.nM*Z*3*De*e.F/Ex*(F-1)/F*(fs.aG*2+fs.gB) : null; };
  const Vb=(Ex,kind)=>{ const mult=4+(rep&&kind!=="dedup"?2:0); return rv.nM*mult*n*De*2*(kind==="slot"?K*(1-1/Ex):kind==="dedup"?(Ex-1)*Par.pDest(Z,Ex,K):(Ex-1)); };
  const Eb=Es.filter(x=>x<=blk);
  const serX=[{name:t("par.ep.cross.w"),short:t("par.ep.cross.wShort"),col:"c-fsdp",pts:Eb.map(Ex=>[Ex,Wb(Ex)||null])},
    {name:t("par.ep.cross.slot"),short:t("par.ep.cross.slotShort"),col:"s5",pts:Eb.filter(x=>x>1).map(Ex=>[Ex,Vb(Ex,"slot")])},
    {name:t("par.ep.cross.dedup"),short:t("par.ep.cross.dedupShort"),col:"s2",pts:Eb.filter(x=>x>1).map(Ex=>[Ex,Vb(Ex,"dedup")])},
    {name:t("par.ep.cross.bcast"),short:t("par.ep.cross.bcastShort"),col:"s1",pts:Eb.filter(x=>x>1).map(Ex=>[Ex,Vb(Ex,"bcast")])}];
  lineChart("p-e-cross",{series:serX,xLog:true,xBase:2,yLog:true,xLab:t("par.ep.cross.xLab", { n: blk }),yLab:t("par.ep.cross.yLab"),yFmt:fB,tipFmt:fB});
  legendInto("p-e-cross-legend",serX.map(se=>({name:se.name,col:se.col,line:true})));
  const Tstar=(()=>{ const a=Math.max(1,E/2), b=E; if(a===b||!Wb(a)) return null; const dW=Wb(a)-(Wb(b)||0), dV=(Vb(b,"slot")-(a>1?Vb(a,"slot"):0))/n; return dV>0?dW/dV:null; })();
  $("p-e-cross-lede").textContent=t("par.ep.cross.lede", { a: Math.max(1,E/2), b: E, T: Tstar?fN(Tstar):"—", n: fN(n) })+crossNote()+"";
  $("p-e-fan").innerHTML='<thead><tr><th>EP</th><th>'+t("par.ep.fan.th.p")+'</th><th>'+t("par.ep.fan.th.d")+'</th><th>'+t("par.ep.fan.th.slot")+'</th><th>'+t("par.ep.fan.th.ratio")+'</th><th>'+t("par.ep.fan.th.disp")+'</th></tr></thead><tbody>'+
    Es.filter(x=>x>1).map(Ex=>{ const p=Par.pDest(Z,Ex,K); return '<tr'+(Ex===E?' class="hl"':'')+'><td class="num">'+Ex+'</td><td class="num">'+p.toFixed(3)+'</td><td class="num">'+((Ex-1)*p).toFixed(2)+'</td><td class="num">'+(K*(1-1/Ex)).toFixed(2)+'</td><td class="num">'+(K/(Ex*p)).toFixed(2)+'</td><td class="num">'+fB(n*K*(1-1/Ex)*De*2)+' / '+fB(n*(Ex-1)*p*De*2)+' / '+fB((Ex-1)*n*De*2)+'</td></tr>'; }).join("")+'</tbody>';
}

