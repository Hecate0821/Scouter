/* ---------- 一层里的通信：先后顺序和每卡字节（后训练 · 并行、推理 · 部署各一张） ----------
   行按发出的先后排。左边是本卡的计算流，右边每个并行维度一条道；通信挂在“等它”或“和它同时跑”的那段计算的同一行。
   实心格是阻塞的：这一行的计算要等它做完才开始。虚线格在后台流上跑，被这一行的计算盖住。
   每个格子的字节和时间都取自 Train.run / Infer.step 里同一组逐次记录（Par.xc），所以和“一个 step”的通信表加起来一致。 */
const XD = { tp:[t("comm.dim.tp"),"c-ar"], cp:[t("comm.dim.cp"),"c-cp"], ep:[t("comm.dim.ep"),"c-a2a"], etp:[t("comm.dim.etp"),"c-moe"], fsdp:[t("comm.dim.fsdp"),"c-fsdp"], efsdp:[t("comm.dim.efsdp"),"c-moe"],
  pp:[t("comm.dim.pp"),"c-dp"], dp:[t("comm.dim.dp"),"c-fsdp"], edp:[t("comm.dim.edp"),"c-moe"], rep:[t("comm.dim.rep"),"c-attn"], dpa:[t("comm.dim.dpa"),"c-dp"] };
const XORDER = ["tp","dpa","cp","ep","etp","fsdp","efsdp","pp","dp","edp","rep"];
const XOP = { AG:t("comm.op.ag"), RS:t("comm.op.rs"), AR:t("comm.op.ar"), A2A:t("comm.op.a2a"), P2P:t("comm.op.p2p"), sync:t("comm.op.sync") };
const XMODE = { block:t("comm.mode.block"), bg:t("comm.mode.bg"), part:t("comm.mode.part"), fused:t("comm.mode.fused") };
const XV = { trn:{ n:2, show:"both" }, inf:{ n:2, cfg:"cur" } };

/* 事件 → 行：每段计算开一行，前面攒下的通信挂到这一行（阻塞的是它要等的，后台的是和它同时跑的）；
   没有通信隔开的相邻计算合成一行。一层结束时还没被等到的阻塞通信单独成一行“等通信”，后台的留给下一层 */
function xRows(ev){
  const rows=[]; let pend=[], last=null;
  const flushWait=()=>{ const b=pend.filter(x=>x.mode==="block"); if(b.length){ rows.push({comp:null,chips:b}); pend=pend.filter(x=>x.mode!=="block"); } };
  for(const e of ev){
    if(e.k==="s"){ flushWait(); rows.push({sec:e}); last=null; continue; }
    if(e.k==="x"){ pend.push(e); continue; }
    if(!(e.t>0) && !e.ghost) continue;
    if(last && !pend.length && !e.ghost && !e.solo){ last.comp=Object.assign({},last.comp,{label:last.comp.label+" → "+e.label,t:last.comp.t+e.t}); continue; }
    const row={comp:e,chips:pend}; pend=[]; rows.push(row); last=e.ghost||e.solo?null:row;
  }
  if(pend.length) rows.push({comp:null,chips:pend});
  return rows;
}
const xMembers = (n, stride) => { const a=[]; for(let i=0;i<n;i++) a.push(i*stride); return a.length>8 ? a.slice(0,4).join(", ")+" … "+a.slice(-2).join(", ") : a.join(", "); };
function xGrid(elId, ev, groups){
  const rows=xRows(ev), lanes=XORDER.filter(k=>ev.some(e=>e.k==="x"&&e.dim===k)), el=$(elId);
  if(!lanes.length){ el.style.setProperty("--k",1); el.innerHTML='<div class="xs">'+t("comm.grid.none")+'</div>'; return lanes; }
  const mx=Math.max(1,...ev.filter(e=>e.k==="x").map(e=>e.bytes));
  let n=0, h='<div class="xr xh"><div class="xn">#</div><div class="xc0">'+t("comm.grid.comp")+'</div>'+lanes.map(k=>{ const G=groups[k]||{};
    return '<div class="xl" style="--c:'+css(XD[k][1])+'"><b>'+esc(XD[k][0])+'</b> '+t("comm.cards",{n:G.n||""})+(G.lk?' · '+esc(G.lk):'')+(G.mem?'<span>'+t("comm.grid.mem",{mem:esc(G.mem)})+'</span>':'')+'</div>'; }).join("")+'</div>';
  for(const r of rows){
    if(r.sec){ h+='<div class="xs'+(r.sec.sub?' sub':'')+'">'+r.sec.label+'</div>'; continue; }
    const c=r.comp;
    h+='<div class="xr"><div class="xn">'+(++n)+'</div>'+(c ? '<div class="xc0'+(c.ghost?' ghost':'')+'" style="--cc:'+css(CAT[c.cat]||"c-oth")+'"><span>'+esc(c.label)+'</span>'+(c.t>0?'<span class="t">'+fT(c.t)+'</span>':'')+'</div>'
      : '<div class="xc0 wait"><span>'+t("comm.grid.wait")+'</span></div>');
    for(const k of lanes) h+='<div class="xl">'+r.chips.filter(x=>x.dim===k).map(x=>xChip(x,mx,groups[k]||{})).join("")+'</div>';
    h+='</div>';
  }
  el.style.setProperty("--k",lanes.length); el.innerHTML=h; bindTips(el);
  return lanes;
}
function xChip(x, mx, G){
  const bw = x.t>0 && x.bytes>0 ? x.bytes/x.t/1e9 : 0;
  const tip=tipAttr(t("comm.chip.title",{dim:XD[x.dim][0],op:XOP[x.op],what:x.what}),[{v:fB(x.bytes),l:t("comm.chip.sent")},{v:fT(x.t),l:t("comm.chip.wire",{bw:bw?t("comm.chip.bw",{bw:fN(bw)}):""})},
    {v:t("comm.cards",{n:G.n||"?"}),l:(G.lk||"")+(G.mem?t("comm.chip.mem",{mem:G.mem}):"")},{v:XMODE[x.mode],l:x.note||""}]);
  return '<div class="xchip '+x.mode+'" style="--c:'+css(XD[x.dim][1])+'" '+tip+'><div class="x1"><b class="xo">'+x.op+'</b><span>'+esc(x.what)+'</span></div>'+
    '<div class="x2"><b>'+fB(x.bytes)+'</b><span>'+fT(x.t)+'</span>'+(x.mode!=="block"?'<em>'+({bg:t("comm.chip.tag.bg"),part:t("comm.chip.tag.part"),fused:t("comm.chip.tag.fused")}[x.mode])+'</em>':'')+'<span class="xd">'+esc(XD[x.dim][0])+'</span></div>'+
    '<i class="xbar"><i style="width:'+Math.max(1.5,x.bytes/mx*100).toFixed(1)+'%"></i></i></div>';
}
/* 每个样例层的线上字节，按维度堆叠；同一把尺子 */
function xVolume(elId, items){
  const mx=Math.max(1,...items.map(it=>sum(it.parts,p=>p.v)));
  $(elId).innerHTML=items.map(it=>{ const tot=sum(it.parts,p=>p.v);
    return '<div class="xv"><span class="lab">'+it.label+'</span><span class="bar">'+it.parts.filter(p=>p.v>0).map(p=>'<i style="width:'+(p.v/mx*100).toFixed(2)+'%;background:'+css(XD[p.dim][1])+'" '+tipAttr(it.label.replace(/<[^>]+>/g,""),[{v:fB(p.v),l:t("comm.vol.share",{dim:XD[p.dim][0],p:pc(p.v/tot)})}])+'></i>').join("")+'</span><span class="num">'+fB(tot)+'</span></div>'; }).join("");
  bindTips($(elId));
}
const xSeg = (elId, opts, cur, on) => { const el=$(elId);
  el.innerHTML=opts.map(([v,l])=>'<button type="button" data-v="'+v+'" aria-pressed="'+(String(v)===String(cur))+'">'+esc(l)+'</button>').join("");
  el.onclick=e=>{ const b=e.target.closest("button"); if(b) on(b.dataset.v); }; };

/* 样例层：取一个全注意力层，和它前面的一两层（混合模型里是线性 / 滑窗层），都落在 MoE 区 */
function xKinds(m){
  const moe=m.moe.E>0, f=m.full;
  const kind = i => ({ a: m.lin.n ? (i%4===3?"full":"lin") : f.swa ? (i%(f.gEvery||4)===(f.gOff||0)?"full":"swa") : "full", f: (moe && i>=m.dense.n) ? "moe" : "dense" });
  const name = k => (k.a==="lin" ? m.lin.kind==="GDN"?t("comm.kind.gdn"):t("comm.kind.kda") : k.a==="swa" ? t("comm.kind.swa",{W:f.W}) : f.kind==="gqa" ? (f.swa?t("comm.kind.global"):t("comm.kind.full")) : f.kind.toUpperCase())+" + "+(k.f==="moe"?"MoE":"dense FFN");
  return { kind, name };
}
function xSample(m, n){
  const {kind}=xKinds(m), lo=(m.moe.E?m.dense.n:0)+n-1;
  let i=lo; while(i<m.L && kind(i).a!=="full") i++; if(i>=m.L) i=lo;
  const L=[]; for(let j=i-n+1;j<=i;j++) L.push(j); return L;
}

/* ================= 训练：当前框架、当前配置（和“一个 step”同一个 Train.run） ================= */
function trnPlan(r, L, show){
  const {m,g}=ctx(), c=S.trn, mg=c.fw==="megatron", fs=r.par.fsdp, f=m.full, {kind,name}=xKinds(m);
  const per = pred => { let F=0,B=0,RC=0; for(const o of r.ops) if(pred(o)){ const n=o.layers||1; F+=o.fwd.t/n; B+=(o.dgrad.t+o.wgrad.t)/n; if(o.rc) RC+=o.fwd.t/n; } return {f:F,b:B,rc:RC}; };
  const cf=r.par.coreFullOp, cs=r.ops.find(o=>o.key==="acore"&&o!==cf);
  const PC = { pre:per(o=>o.group==="attn"&&PRE_KEYS.includes(o.key)), full:per(o=>o===cf), swa:per(o=>o===cs), post:per(o=>o.group==="attn"&&o.key!=="acore"&&!PRE_KEYS.includes(o.key)),
    lpre:per(o=>o.key==="lproj"), lmid:per(o=>["lconv","lcore","lnorm"].includes(o.key)), lpost:per(o=>o.key==="lout"),
    router:per(o=>["router","latd","permute"].includes(o.key)), exp:per(o=>["moe1","act","moe2"].includes(o.key)), unp:per(o=>["unperm","latu"].includes(o.key)), shared:per(o=>o.key==="shared"), dense:per(o=>o.key==="dense") };
  const xsOf = (key, blk) => { const e=r.comm.find(x=>x.key===key&&(!blk||x.blk===blk)); return e&&e.xs ? e.xs.map(x=>Object.assign({},x,{ent:e})) : []; };
  const P = (xs, pass) => xs.filter(x=>x.pass===pass);
  const ev=[], C=(label,t,cat,o)=>ev.push(Object.assign({k:"c",label,t,cat},o||{})), sec=(label,o)=>ev.push(Object.assign({k:"s",label},o||{}));
  const X=(dim,x,mode,lay,o)=>{ if(x) ev.push(Object.assign({k:"x",dim,mode,lay},x,o||{})); };
  const tpM = c.tpOverlap ? "part" : "block", tpNote = c.tpOverlap ? t("comm.note.tpUb") : "";
  const epAll = xsOf("a2a","moe"), cnt = xsOf("cnt","moe"), etp = xsOf("etp","moe"), epFused = epAll.length && epAll[0].ent.fused;
  const epM = epFused ? "fused" : (mg && c.epOverlap ? "part" : "block"), epNote = epFused ? t("comm.note.epFused") : (mg && c.epOverlap ? t("comm.note.epOverlap") : "");
  const cpMeth = r.par.cpMeth;
  const fsU = i => { const k=kind(i); return { d:(k.a==="lin"?fs.pLin:fs.pFull)+(k.f==="dense"?fs.pDen:0), e:k.f==="moe"?fs.pExpL:0 }; };
  const lb = j => j<0 ? t("comm.lb.emb") : j>=m.L ? t("comm.lb.head") : t("comm.lb.layer",{i:j});
  const fsX = (i, pass, op, note, mode) => { if(!fs || i<0 || i>=m.L) return; const u=fsU(i);
    for(const [dim,P1,F,lk] of [["fsdp",u.d,fs.fs,fs.lkD],["efsdp",u.e,fs.fe,fs.lkE]])
      for(const x of Par.fsdpDet(P1,F,lk,g,fs.aG,fs.gB)) if(x.pass===pass && x.op===op) X(dim,Object.assign({},x,{what:dim==="efsdp"?t("comm.fsdp.whatExp",{i,what:x.what,note:note||""}):t("comm.fsdp.whatDense",{i,what:x.what,note:note||""})}),mode||"bg",i); };
  const tpAttnBlk = k => k.a==="lin" ? "lin" : "attn";

  /* 一层的前向（pass = fwd 或 rc：重算时只走 rc 记录和被重算的算子） */
  const fwdLayer = (i, pass) => { const k=kind(i), rc=pass==="rc", T=p=>rc?p.rc:p.f, rl=s=>rc?t("comm.rc.label",{label:s}):s;
    const tp=P(xsOf("tp",tpAttnBlk(k)),pass);
    X("tp",tp[0],tpM,i,{note:tpNote});
    if(f.kind==="dsa" && k.a==="full") P(xsOf("tpidx","attn"),"fwd").filter(()=>!rc).forEach(x=>X("tp",x,tpM,i));
    if(k.a==="lin"){
      const cl=P(xsOf("cplin","lin"),pass);
      C(rl("in_proj"),T(PC.lpre),"lin");
      if(cl[0] && cl[0].op==="A2A"){ X("cp",cl[0],"block",i); C(rl(t("comm.lin.midA2a",{kind:m.lin.kind,h:fN(m.lin.Hv/r.tp/r.cp)})),T(PC.lmid),"lin"); X("cp",cl[1],"block",i); }
      else if(cl[0]){ C(rl(t("comm.lin.midLocal")),T(PC.lmid)*0.5,"lin",{solo:true}); X("cp",cl[0],"block",i,{note:t("comm.note.cpLin")}); C(rl(t("comm.lin.fix")),T(PC.lmid)*0.5,"lin"); }
      else C(rl(t("comm.lin.mid",{kind:m.lin.kind})),T(PC.lmid),"lin");
      C(rl("out_proj"),T(PC.lpost),"proj");
    } else {
      const cpx=P(xsOf(k.a==="swa"?"cpswa":"cpkv",k.a==="swa"?"swa":"full"),pass), core=k.a==="swa"?PC.swa:PC.full;
      C(rl(f.kind==="gqa"?t("comm.attn.preGqa"):f.kind==="dsa"?t("comm.attn.preDsa"):t("comm.attn.preMla")),T(PC.pre),"proj");
      if(!cpx.length) C(rl(t("comm.attn.core")),T(core),"attn");
      else if(cpMeth==="ulysses"){ X("cp",cpx[0],"block",i); C(rl(t("comm.attn.coreUly")),T(core),"attn"); X("cp",cpx[1],"block",i); }
      else if(k.a==="swa"){ X("cp",cpx[0],"block",i); C(rl(t("comm.attn.coreSwa")),T(core),"attn"); }
      else { X("cp",cpx[0],"part",i,{note:cpMeth==="ring"?t("comm.note.cpRing"):t("comm.note.cpAg")}); C(rl(t("comm.attn.coreCausal")),T(core),"attn"); }
      C(rl(t("comm.attn.post",{gate:f.gate?t("comm.attn.gate"):""})),T(PC.post),"proj");
    }
    X("tp",tp[1],tpM,i,{note:tpNote});
    if(k.f==="moe"){
      C(rl("router + top-k + permute"),T(PC.router),"proj");
      P(cnt,pass).forEach(x=>X("ep",x,"block",i,{note:t("comm.note.cnt")}));
      const ep=P(epAll,pass), et=rc?[]:P(etp,"fwd");
      X("ep",ep[0],epM,i,{note:epNote}); X("etp",et[0],"block",i);
      C(rl(t("comm.moe.exp")),T(PC.exp),"moe");
      X("etp",et[1],"block",i); X("ep",ep[1],epM,i,{note:epNote});
      C(rl(t("comm.moe.unp")),T(PC.unp),"proj");
      if(PC.shared.f>0){ const sh=P(xsOf("tp","moe"),pass); X("tp",sh[0],tpM,i); C(rl("shared expert"),T(PC.shared),"proj"); X("tp",sh[1],tpM,i); }
    } else { const dn=P(xsOf("tp","dense"),pass); X("tp",dn[0],tpM,i); C(rl("dense FFN"),T(PC.dense),"proj"); X("tp",dn[1],tpM,i); }
  };
  /* 重算：一整行，计算是这一层被重算的算子，通信是前向里那些调用再来一遍（顺序同前向） */
  const rcLayer = i => { const k=kind(i), blk=tpAttnBlk(k), rc=p=>p.rc;
    const tRc=(k.a==="lin" ? rc(PC.lpre)+rc(PC.lmid)+rc(PC.lpost) : rc(PC.pre)+rc(k.a==="swa"?PC.swa:PC.full)+rc(PC.post)) + (k.f==="moe" ? rc(PC.router)+rc(PC.exp)+rc(PC.unp)+rc(PC.shared) : rc(PC.dense));
    const cpx=P(k.a==="lin"?xsOf("cplin","lin"):xsOf(k.a==="swa"?"cpswa":"cpkv",k.a==="swa"?"swa":"full"),"rc"), cpBlk=k.a!=="full"||cpMeth==="ulysses";
    const tag = (x, s) => Object.assign({}, x, {what:t("comm.rc.tag",{what:x.what,s})});
    const xs=[...P(xsOf("tp",blk),"rc").map(x=>["tp",tag(x,k.a==="lin"?t("comm.rc.lin"):t("comm.rc.attn")),tpM]), ...cpx.map(x=>["cp",x,cpBlk?"block":"part"]),
      ...(k.f==="moe" ? [...P(cnt,"rc").map(x=>["ep",x,"block"]), ...P(epAll,"rc").map(x=>["ep",x,epM]), ...P(xsOf("tp","moe"),"rc").map(x=>["tp",tag(x,t("comm.rc.shared")),tpM])] : P(xsOf("tp","dense"),"rc").map(x=>["tp",tag(x,t("comm.rc.ffn")),tpM]))];
    if(!(tRc>0) && !xs.length) return;
    xs.forEach(([dim,x,mode])=>X(dim,x,mode,i,{note:t("comm.note.rc")}));
    C(t("comm.rc.row",{x:xs.length?t("comm.rc.rowComm"):""}),tRc,"oth",{solo:true});
  };
  /* 一层的反向：先重算（如果开了），再 MoE / FFN，再 attention */
  const bwdLayer = i => { const k=kind(i), B=p=>p.b;
    rcLayer(i);
    const tpB = blk => P(xsOf("tp",blk),"bwd");   // Megatron：[AG 输出梯度, AG 再 gather 输入, RS 输入梯度]；torchtitan：[AG, RS]
    const tpBlock = (blk, label, dt, cat) => { const x=tpB(blk); X("tp",x[0],tpM,i,{note:tpNote});
      if(mg && x.length===3){ X("tp",x[1],"bg",i,{note:t("comm.note.dgrad")}); X("tp",x[2],"bg",i,{note:t("comm.note.wgrad")}); C(label,dt,cat); }
      else { C(label,dt,cat); X("tp",x[1],tpM,i,{note:tpNote}); } };
    if(k.f==="moe"){
      if(PC.shared.f>0) tpBlock("moe",t("comm.bwd.shared"),B(PC.shared),"proj");
      const ep=P(epAll,"bwd"), et=P(etp,"bwd");
      X("ep",ep[0],epM,i,{note:epNote}); X("etp",et[0],"block",i);
      C("expert dgrad + wgrad",B(PC.exp),"moe");
      X("etp",et[1],"block",i); X("ep",ep[1],epM,i,{note:epNote});
      C(t("comm.bwd.router"),B(PC.unp)+B(PC.router),"proj");
    } else tpBlock("dense",t("comm.bwd.dense"),B(PC.dense),"proj");
    const blk=tpAttnBlk(k), tx=tpB(blk);
    X("tp",tx[0],tpM,i,{note:tpNote});
    if(k.a==="lin"){
      C(t("comm.bwd.outProj"),B(PC.lpost),"proj");
      const cl=P(xsOf("cplin","lin"),"bwd");
      if(cl[0] && cl[0].op==="A2A"){ X("cp",cl[0],"block",i); C(t("comm.bwd.linMid"),B(PC.lmid),"lin"); X("cp",cl[1],"block",i); }
      else if(cl[0]){ C(t("comm.bwd.linLocal"),B(PC.lmid)*0.5,"lin",{solo:true}); X("cp",cl[0],"block",i); C(t("comm.bwd.linFix"),B(PC.lmid)*0.5,"lin"); }
      else C(t("comm.bwd.linMid"),B(PC.lmid),"lin");
      if(mg && tx.length===3){ X("tp",tx[1],"bg",i,{note:t("comm.note.dgrad")}); X("tp",tx[2],"bg",i,{note:t("comm.note.wgrad")}); C(t("comm.bwd.inProj"),B(PC.lpre),"proj"); }
      else { C(t("comm.bwd.inProj"),B(PC.lpre),"proj"); X("tp",tx[1],tpM,i); }
    } else {
      C(t("comm.bwd.oProj"),B(PC.post),"proj");
      const cpx=P(xsOf(k.a==="swa"?"cpswa":"cpkv",k.a==="swa"?"swa":"full"),"bwd"), core=k.a==="swa"?PC.swa:PC.full;
      if(!cpx.length) C(t("comm.bwd.attn"),B(core),"attn");
      else if(cpMeth==="ulysses"){ X("cp",cpx[0],"block",i); C(t("comm.bwd.attn"),B(core),"attn"); X("cp",cpx[1],"block",i); }
      else if(k.a==="swa"){ X("cp",cpx[0],"block",i); C(t("comm.bwd.attnSwa"),B(core),"attn"); X("cp",cpx[1],"block",i); }
      else if(cpMeth==="ring"){ X("cp",cpx[0],"part",i,{note:t("comm.note.ringBwd")}); X("cp",cpx[1],"part",i,{note:t("comm.note.ringDkv")}); C(t("comm.bwd.attnRing"),B(core),"attn"); }
      else { X("cp",cpx[0],"part",i,{note:t("comm.note.agBwd")}); C(t("comm.bwd.attn"),B(core),"attn"); X("cp",cpx[1],"block",i,{note:t("comm.note.agDkv")}); }
      if(f.kind==="dsa" && k.a==="full") P(xsOf("tpidx","attn"),"bwd").forEach(x=>X("tp",x,tpM,i));
      if(mg && tx.length===3){ X("tp",tx[1],"bg",i,{note:t("comm.note.dgrad")}); X("tp",tx[2],"bg",i,{note:t("comm.note.wgrad")}); C(t("comm.bwd.qkv"),B(PC.pre),"proj"); }
      else { C(t("comm.bwd.qkv"),B(PC.pre),"proj"); X("tp",tx[1],tpM,i); }
    }
  };

  const i0=L[0], i1=L[L.length-1];
  if(show!=="bwd"){
    if(fs){ fsX(i0,"fwd","AG",t("comm.fsdp.prefPrev")); C(t("comm.ghost.fwd",{l:lb(i0-1)}),0,"oth",{ghost:true}); }
    for(const i of L){ sec(t("comm.sec.head",{i,name:esc(name(kind(i))),pass:"<span>"+t("comm.sec.fwd")+"</span>"}));
      if(fs) fsX(i+1,"fwd","AG",t("comm.fsdp.prefNext"));
      fwdLayer(i,"fwd"); }
  }
  if(show!=="fwd"){
    if(fs){ if(fs.aG===2) fsX(i1,"bwd","AG",t("comm.fsdp.prefPrevBwd")); C(t("comm.ghost.bwd",{l:lb(i1+1)}),0,"oth",{ghost:true}); }
    for(const i of L.slice().reverse()){ sec(t("comm.sec.head",{i,name:esc(name(kind(i))),pass:"<span>"+t("comm.sec.bwd",{rc:r.ops.some(o=>o.rc)?t("comm.sec.bwdRc"):""})+"</span>"}));
      if(fs && fs.aG===2) fsX(i-1,"bwd","AG",t("comm.fsdp.prefNextBwd"));
      bwdLayer(i);
      if(fs) fsX(i,"bwd","RS",t("comm.fsdp.rs")); }
    if(fs) C(t("comm.ghost.bwd",{l:lb(i0-1)}),0,"oth",{ghost:true});
  }
  /* 每个 micro-batch / 每个 step 各一次 */
  const ppx=xsOf("pp","stage");
  if(ppx.length || r.stepX.length || fs){
    sec(t("comm.sec.perMb"),{sub:true});
    if(ppx.length){ X("pp",ppx[0],"bg","mb",{note:t("comm.note.pp")}); C(t("comm.ghost.ppFwd"),0,"oth",{ghost:true});
      X("pp",ppx[1],"bg","mb",{note:t("comm.note.pp")}); C(t("comm.ghost.ppBwd"),0,"oth",{ghost:true}); }
    if(mg){ const rs=r.stepX.filter(x=>x.op==="RS"), ag=r.stepX.filter(x=>x.op==="AG");
      rs.forEach(x=>X(x.dim,x,c.overlapGrad?"bg":"block","step",{note:c.overlapGrad?t("comm.note.overlapGrad"):""}));
      if(c.overlapGrad && rs.length) C(t("comm.ghost.lastBwd"),0,"oth",{ghost:true});
      C(t("comm.opt.mg",{note:r.optNote}),r.tOpt,"oth",{solo:true});
      ag.forEach(x=>X(x.dim,x,c.overlapParam?"bg":"block","step",{note:c.overlapParam?t("comm.note.overlapParam"):""}));
      if(c.overlapParam && ag.length) C(t("comm.ghost.nextFwd"),0,"oth",{ghost:true}); }
    else { r.stepX.forEach(x=>X(x.dim,x,"block","step",{note:t("comm.note.step")}));
      C(t("comm.opt.fsdp",{note:r.optNote}),r.tOpt,"oth",{solo:true}); }
  }
  return ev;
}
function trnGroups(r){
  const {topo}=ctx(), fs=r.par.fsdp, mg=S.trn.fw==="megatron", N=Math.round(r.dp*r.tp*r.cp*r.pp), Nst=N/r.pp, kd=l=>l?(l.n<=1?"":l.intra===false?"IB":"NVLink"):"NVLink";
  const G = (n, stride, lk) => ({ n, lk:n>1?kd(lk):"", mem:n>1?xMembers(n,stride):"" });
  const o = { tp:G(r.tp,1,r.lk.tp), cp:G(r.cp,r.tp,r.lk.cp), ep:G(r.ep,mg?r.etp:1,r.lk.ep), etp:G(r.etp,1,r.lk.tp), pp:G(r.pp,Nst,r.lk.pp) };
  if(fs){ o.fsdp=G(fs.fs,r.tp,fs.lkD); o.efsdp=G(fs.fe,r.ep,fs.lkE); o.rep=G(fs.R,Nst/fs.R,fs.lkR); }
  for(const x of r.stepX) if(!o[x.dim]) o[x.dim]=G(x.n,x.dim==="dp"?r.tp:r.ep*r.etp,x.lk);
  return o;
}
function renderTrnComm(){
  const {m,hw,topo}=ctx(), c=S.trn, r=trnRun(), mg=c.fw==="megatron", fs=r.par.fsdp, {kind,name}=xKinds(m);
  const L=xSample(m,XV.trn.n), ev=trnPlan(r,L,XV.trn.show), groups=trnGroups(r);
  xSeg("x-t-n",[[1,t("comm.seg.layers",{n:1})],[2,t("comm.seg.layers",{n:2})],[3,t("comm.seg.layers",{n:3})]],XV.trn.n,v=>{ XV.trn.n=+v; renderTrnComm(); });
  xSeg("x-t-show",[["both",t("comm.seg.both")],["fwd",t("comm.seg.fwd")],["bwd",t("comm.seg.bwd")]],XV.trn.show,v=>{ XV.trn.show=v; renderTrnComm(); });
  const lanes=xGrid("x-t-grid",ev,groups);
  $("x-t-lede").innerHTML=t("comm.trn.lede",{fw:mg?"Megatron-Core":"torchtitan",model:esc(m.short),hw:hw.name,topo:esc(topo.name),tp:r.tp,cp:r.cp,ep:r.ep,etp:r.etp>1?" / ETP "+r.etp:"",pp:r.pp>1?" / PP "+r.pp:"",
    par:fs?t("comm.trn.ledeFsdp",{n:fs.fs,exp:fs.fe>1?t("comm.trn.ledeExp",{n:fs.fe}):"",rep:fs.R>1?" × HSDP "+fs.R:""}):t("comm.trn.ledeDp",{n:r.dp}),
    tok:V(fN(r.Tc)),rc:esc({full:t("comm.trn.recompute.full"),selective:t("comm.trn.recompute.selective"),default:t("comm.trn.recompute.default"),none:t("comm.trn.recompute.none")}[c.recompute]||c.recompute)});
  /* 样例层的字节 */
  const lay = i => ev.filter(e=>e.k==="x"&&e.lay===i);
  xVolume("x-t-vol",L.map(i=>({label:t("comm.vol.label",{i,name:"<span>"+esc(name(kind(i)))+"</span>"}),parts:lanes.map(dim=>({dim,v:sum(lay(i).filter(e=>e.dim===dim),e=>e.bytes)}))})));
  /* 按维度汇总：样例层、每 micro-batch、每 step */
  const byMB={}, add=(o,k,b,t)=>{ o[k]=o[k]||{b:0,t:0}; o[k].b+=b; o[k].t+=t; };
  const dimOf = x => x.dim || ({tp:"tp",tpidx:"tp",cpkv:"cp",cpswa:"cp",cplin:"cp",a2a:"ep",cnt:"ep",etp:"etp",pp:"pp"})[x.key];
  for(const x of r.comm){ const dm=dimOf(x); if(!dm) continue; const k=x.perStage?1:1/r.pp; add(byMB,dm,x.bytes*k,x.t*k); }
  if(fs){ add(byMB,"fsdp",fs.bDense,fs.tDense); add(byMB,"efsdp",fs.bExp,fs.tExp); }
  const bySt={}; for(const k in byMB) add(bySt,k,byMB[k].b*r.microsPerStep,byMB[k].t*r.microsPerStep); for(const x of r.stepX) add(bySt,x.dim,x.bytes,x.t);
  const cnt = (dim, i) => { const xs=lay(i).filter(e=>e.dim===dim), n=p=>xs.filter(e=>e.pass===p).length; return xs.length ? [["fwd",k=>t("comm.cnt.fwd",{n:k})],["rc",k=>t("comm.cnt.rc",{n:k})],["bwd",k=>t("comm.cnt.bwd",{n:k})]].filter(([p])=>n(p)).map(([p,l])=>l(n(p))).join(" · ") : "—"; };
  const dims=XORDER.filter(k=>lanes.includes(k)||bySt[k]);
  const stTot=sum(dims,k=>bySt[k]?bySt[k].b:0);
  $("x-t-tbl").innerHTML='<thead><tr><th>'+t("comm.tbl.dim")+'</th><th>'+t("comm.tbl.group")+'</th>'+L.map(i=>'<th>'+t("comm.tbl.layer",{i})+'</th>').join("")+'<th>'+t("comm.tbl.perMb")+'</th><th>'+t("comm.tbl.perStep",{n:r.microsPerStep})+'</th><th>'+t("comm.tbl.stepWire")+'</th></tr></thead><tbody>'+
    dims.map(k=>{ const G=groups[k]||{}, s=bySt[k]||{b:0,t:0}, mb=byMB[k];
      return '<tr><td><span class="sw" style="background:'+css(XD[k][1])+'"></span>'+esc(XD[k][0])+'</td><td>'+t("comm.cards",{n:G.n||"—"})+(G.lk?' · '+G.lk:'')+(G.mem?'<br><span class="tag">'+esc(G.mem)+'</span>':'')+'</td>'+
        L.map(i=>{ const b=sum(lay(i).filter(e=>e.dim===k),e=>e.bytes); return '<td class="num">'+(b?cnt(k,i)+'<br>'+fB(b):'—')+'</td>'; }).join("")+
        '<td class="num">'+(mb?fB(mb.b):"—")+'</td><td class="num">'+fB(s.b)+' <span class="tag">'+pc(s.b/stTot)+'</span></td><td class="num">'+fT(s.t)+'</td></tr>'; }).join("")+
    '<tr class="hl"><td>'+t("comm.tbl.total")+'</td><td></td>'+L.map(i=>'<td class="num">'+fB(sum(lay(i),e=>e.bytes))+'</td>').join("")+'<td class="num">'+fB(sum(dims,k=>byMB[k]?byMB[k].b:0))+'</td><td class="num">'+fB(stTot)+'</td><td class="num">'+fT(sum(dims,k=>bySt[k]?bySt[k].t:0))+'</td></tr></tbody>';
  const big=dims.slice().sort((a,b)=>(bySt[b]?bySt[b].b:0)-(bySt[a]?bySt[a].b:0))[0];
  $("x-t-note").innerHTML=t("comm.trn.note",{b:V(fB(stTot)),big:esc(XD[big]?XD[big][0]:"—"),p:pc((bySt[big]?bySt[big].b:0)/stTot),
    pp:r.pp>1?t("comm.trn.notePp",{n:Math.round(m.L/r.pp)}):""});
}

/* ================= 推理：当前引擎、当前阶段（和“一步 decode / prefill”同一个 Infer） ================= */
function infPlan(r, L){
  const {m}=ctx(), c=S.inf, {kind,name}=xKinds(m), dec=c.phase==="decode";
  const per = pred => sum(r.ops.filter(pred),o=>o.t/o.layers);
  const cores=r.ops.filter(o=>o.key==="acore");
  const PI = { pre:per(o=>o.group==="attn"&&!["acore","agate","auv","aout"].includes(o.key)), full:cores[0]?cores[0].t/cores[0].layers:0, swa:cores[1]?cores[1].t/cores[1].layers:0, post:per(o=>["agate","auv","aout"].includes(o.key)),
    lin:per(o=>o.group==="lin"), router:per(o=>["router","latd","moeq"].includes(o.key)), exp:per(o=>o.key==="moe"), latu:per(o=>o.key==="latu"), shared:per(o=>o.key==="shared"), dense:per(o=>o.key==="dense"),
    lm:per(o=>o.key==="lm"), samp:per(o=>o.key==="samp") };
  const one = key => { const e=r.comm.find(x=>x.key===key); return e ? { op:{ar:"AR",latar:"AR",dpg:"AG",dps:"RS",disp:"A2A",comb:"A2A",cnt:"sync",lmg:"AG"}[key], bytes:e.bytes/e.layers, t:e.t/e.layers, ov:e.overlap } : null; };
  const ev=[], C=(label,t,cat,o)=>ev.push(Object.assign({k:"c",label,t,cat},o||{})), sec=(label,o)=>ev.push(Object.assign({k:"s",label},o||{}));
  const X=(dim,key,what,lay,o)=>{ const x=one(key); if(!x) return; const mode=x.ov==="moe"?"fused":x.ov==="tbo"?"part":"block";
    ev.push(Object.assign({k:"x",dim,mode,lay,pass:"fwd",what,note:x.ov==="tbo"?t("comm.inf.noteTbo"):x.ov==="moe"?t("comm.inf.noteFused"):""},x,o||{})); };
  const ag = r.moePar==="ag", epDim = ag ? "ep" : "dpa";
  const arF = r.fused ? t("comm.inf.arFused") : "";
  for(const i of L){ const k=kind(i);
    sec(t("comm.sec.head",{i,name:esc(name(k)),pass:"<span>"+(dec?t("comm.sec.dec"):t("comm.sec.pre"))+"</span>"}));
    if(k.a==="lin") C(t("comm.inf.lin",{kind:m.lin.kind}),PI.lin,"lin");
    else { C(t("comm.inf.qkv",{mla:m.full.kind!=="gqa"?t("comm.inf.qkvMla"):""}),PI.pre,"proj"); C(k.a==="swa"?t("comm.attn.coreSwa"):t("comm.attn.core"),k.a==="swa"?PI.swa:PI.full,"attn"); C(t("comm.attn.post",{gate:m.full.gate?t("comm.attn.gate"):""}),PI.post,"proj"); }
    X("tp","ar",t("comm.inf.arAttn",{fused:arF}),i);
    if(k.f==="moe"){
      X(epDim,"dpg",ag?t("comm.inf.dpgEp"):t("comm.inf.dpgDpa"),i);
      C(t("comm.inf.router",{lat:m.moe.Dlat?t("comm.inf.routerLat"):""}),PI.router,"proj");
      if(PI.shared>0) C(t("comm.inf.shared"),PI.shared,"proj");
      X("ep","cnt",t("comm.inf.cnt"),i); X("ep","disp",t("comm.inf.disp"),i);
      C("routed experts",PI.exp,"moe");
      X("ep","comb",t("comm.inf.comb"),i);
      X("tp","latar",t("comm.inf.latar",{d:m.moe.Dlat}),i);
      if(PI.latu>0) C(t("comm.inf.latu"),PI.latu,"proj");
      X(epDim,"dps",ag?t("comm.inf.dpsEp"):t("comm.inf.dpsDpa"),i);
      X("tp","ar",t("comm.inf.arMoe",{fused:arF}),i);
    } else { C("dense FFN",PI.dense,"proj"); X("tp","ar",t("comm.inf.arFfn",{fused:arF}),i); }
  }
  sec(t("comm.sec.perStep"),{sub:true});
  C("lm_head",PI.lm,"lm"); X("tp","lmg",t("comm.inf.lmg"),"step"); C(t("comm.inf.samp"),PI.samp,"lm");
  return ev;
}
/* 推理：默认看左侧的当前配置；它一张卡一个实例（没有通信）时，换成这个拓扑上第一个多卡的部署预设 */
function infCommCfg(){
  const cfgs=toConfigs(S.topo), cur=infRun();
  const opts=[["cur",t("comm.inf.cur",{eng:ENGN[S.inf.engine],n:S.inf.N})]].concat(cfgs.filter(cf=>cf.cfg.N>1).map(cf=>[cf.id,cf.eng+" · "+cf.label]));
  let id=XV.inf.cfg; if(!opts.some(o=>o[0]===id)) id="cur";
  let auto=false; if(id==="cur" && !cur.comm.length && opts.length>1 && !XV.inf.pinned){ id=opts[1][0]; auto=true; }
  const sel=$("x-i-cfg"); sel.innerHTML=opts.map(o=>'<option value="'+esc(o[0])+'">'+esc(o[1])+'</option>').join(""); sel.value=id;
  sel.onchange=e=>{ XV.inf.cfg=e.target.value; XV.inf.pinned=true; renderInfComm(); };
  const cf=cfgs.find(x=>x.id===id);
  return { over: cf ? Object.assign({},cf.cfg,{engine:cf.cfg.engine||cf.engine}) : {}, cf, auto };
}
function renderInfComm(){
  const {m,hw,topo}=ctx(), pick=infCommCfg(), c=Object.assign({},S.inf,pick.over), r=infRun(pick.over), {kind,name}=xKinds(m), dec=c.phase==="decode";
  const L=xSample(m,XV.inf.n), ev=infPlan(r,L), tpA=Math.round(r.tpA), lk=n=>n>1?"NVLink":"";
  const groups={ tp:{n:tpA,lk:lk(tpA),mem:tpA>1?xMembers(tpA,1):""}, dpa:{n:r.dpA,lk:lk(r.dpA),mem:r.dpA>1?xMembers(r.dpA,tpA):""}, ep:{n:r.N,lk:lk(r.N),mem:r.N>1?xMembers(r.N,1):""} };
  xSeg("x-i-n",[[1,t("comm.seg.layers",{n:1})],[2,t("comm.seg.layers",{n:2})],[3,t("comm.seg.layers",{n:3})]],XV.inf.n,v=>{ XV.inf.n=+v; renderInfComm(); });
  const lanes=xGrid("x-i-grid",ev,groups);
  $("x-i-lede").innerHTML=t("comm.inf.lede",{auto:pick.auto?t("comm.inf.ledeAuto"):"",eng:ENGN[c.engine],model:esc(m.short),hw:hw.name,n:c.N,tp:tpA,dp:r.dpA,moe:esc(MOENAME[r.moePar]||r.moePar),
    work:dec?t("comm.inf.ledeDec",{b:r.Breq,rows:fN(r.rowsG),spec:r.specK?t("comm.inf.ledeSpec",{k:r.specK+1}):""}):t("comm.inf.ledePre",{c:fN(c.C)})});
  const lay = i => ev.filter(e=>e.k==="x"&&e.lay===i);
  xVolume("x-i-vol",L.map(i=>({label:t("comm.vol.label",{i,name:"<span>"+esc(name(kind(i)))+"</span>"}),parts:lanes.map(dim=>({dim,v:sum(lay(i).filter(e=>e.dim===dim),e=>e.bytes)}))})));
  const dimOf = key => ({ar:"tp",latar:"tp",lmg:"tp",disp:"ep",comb:"ep",cnt:"ep",dpg:r.moePar==="ag"?"ep":"dpa",dps:r.moePar==="ag"?"ep":"dpa"})[key];
  const bySt={}; for(const x of r.comm){ const k=dimOf(x.key); if(!k) continue; bySt[k]=bySt[k]||{b:0,t:0}; bySt[k].b+=x.bytes; bySt[k].t+=x.t; }
  const dims=XORDER.filter(k=>bySt[k]), tot=sum(dims,k=>bySt[k].b), toks = dec ? r.Breq*(1+(r.specK?c.accept:0)) : c.C;
  $("x-i-tbl").innerHTML='<thead><tr><th>'+t("comm.tbl.dim")+'</th><th>'+t("comm.tbl.group")+'</th>'+L.map(i=>'<th>'+t("comm.tbl.layer",{i})+'</th>').join("")+'<th>'+t("comm.tbl.step",{n:m.L})+'</th><th>'+t("comm.tbl.stepWireInf")+'</th><th>'+(dec?t("comm.tbl.perTokOut"):t("comm.tbl.perTokIn"))+'</th></tr></thead><tbody>'+
    dims.map(k=>{ const G=groups[k]||{}, s=bySt[k];
      return '<tr><td><span class="sw" style="background:'+css(XD[k][1])+'"></span>'+esc(XD[k][0])+'</td><td>'+t("comm.cards",{n:G.n||"—"})+(G.lk?' · '+G.lk:'')+(G.mem?'<br><span class="tag">'+esc(G.mem)+'</span>':'')+'</td>'+
        L.map(i=>{ const xs=lay(i).filter(e=>e.dim===k); return '<td class="num">'+(xs.length?t("comm.tbl.times",{n:xs.length})+'<br>'+fB(sum(xs,e=>e.bytes)):'—')+'</td>'; }).join("")+
        '<td class="num">'+fB(s.b)+' <span class="tag">'+pc(s.b/tot)+'</span></td><td class="num">'+fT(s.t)+'</td><td class="num">'+fB(s.b/Math.max(1,toks))+'</td></tr>'; }).join("")+
    (dims.length?'<tr class="hl"><td>'+t("comm.tbl.total")+'</td><td></td>'+L.map(i=>'<td class="num">'+fB(sum(lay(i),e=>e.bytes))+'</td>').join("")+'<td class="num">'+fB(tot)+'</td><td class="num">'+fT(sum(dims,k=>bySt[k].t))+'</td><td class="num">'+fB(tot/Math.max(1,toks))+'</td></tr>':'<tr><td colspan="'+(5+L.length)+'">'+t("comm.tbl.none")+'</td></tr>')+'</tbody>';
  $("x-i-note").innerHTML=!dims.length ? t("comm.inf.noteNone") :
    t("comm.inf.note",{b:V(fB(tot)),wt:V(fT(sum(dims,k=>bySt[k].t))),step:fT(r.stepT)});
}
