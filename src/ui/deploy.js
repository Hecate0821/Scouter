/* ---------- 推理 · 部署与并行：DP-attention + 宽 EP、同样多的卡怎么切、DCP、chunked prefill ----------
   数都来自引擎：Infer.layouts / layoutPoint（布局和每卡并发 b 下的拆解）、toCurve（和推荐配置同一条 Pareto 曲线）、Infer.dcpSweep、Infer.chunked。
   当前配置（S.inf）的引擎、精度、投机、上下文都带进去，只改各块写明要改的那几项 */
const LY = { b:32, all:false };
const FAMS = ["tp","hybrid","dep","dcp","hybridDcp"], FAMCOL = { tp:"s1", hybrid:"s2", dep:"s3", dcp:"s4", hybridDcp:"s5" };
const lyName = l => t("deploy.ly.name."+(l.fam==="tp" && l.inst===1 ? "tp1" : l.fam),{n:l.N, a:l.a, dp:l.N/l.a, d:l.dcp, k:l.inst});
const lyCfg = (l, extra) => Object.assign(clone(S.inf), l.cfg, { phase:"decode", moePar:compatMoe(S.inf.engine, l.cfg.moePar) }, extra||{});
const isCur = l => S.inf.N===l.N && Math.max(1,S.inf.dpA)===l.N/l.a && Math.max(1,S.inf.dcp||1)===l.dcp && (l.fam==="tp"||l.fam==="dcp" ? !["deepep","standard","hybridep"].includes(S.inf.moePar) : ["deepep","standard","hybridep"].includes(S.inf.moePar));
/* 公开数字：deploy_public 里这几个 topic 的行，新的在前 */
function deployPub(el, topics){
  const rows=DB.set("deploy_public").filter(p=>topics.includes(p.topic)).sort((x,y)=>y.date.localeCompare(x.date));
  table(el,[t("deploy.pub.head.date"),t("deploy.pub.head.src"),t("deploy.pub.head.what"),t("deploy.pub.head.setting"),t("deploy.pub.head.value"),t("deploy.pub.head.caveat")],
    rows.map(p=>[V(esc(p.date)),'<span class="nw">'+aOut(p.url,p.src)+'</span><br><span class="tag">'+t("deploy.pub.kind."+p.kind)+'</span>',
      esc(p.what)+([p.model,p.hw].some(x=>x&&x!=="—")?'<br><span class="tag">'+esc([p.model,p.hw].filter(x=>x&&x!=="—").join(" · "))+'</span>':""),esc(p.setting),esc(p.value),esc(p.caveat||"")]));
}
/* 一族布局里所有点的上包络：从单用户速度最高的点往左扫，只留单卡吞吐创新高的点 */
function frontier(pts){ const s=pts.slice().sort((a,b)=>b[0]-a[0]), out=[]; let best=-Infinity;
  for(const p of s) if(p[1]>best){ out.push(p); best=p[1]; } return out.reverse(); }

/* ---- 同样 G 张卡怎么切 ---- */
function renderDeployLayouts(){
  const {m,hw,topo,g}=ctx(), G=topo.gpus, [isl,osl,seqName]=SEQS[TO.seq], E=ENGN[S.inf.engine]||S.inf.engine;
  fillOpts($("dl-seq"),Object.entries(SEQS).map(([k,v])=>[k,v[2]]),TO.seq);
  fillOpts($("dl-target"),[10,20,50,100,200,400].map(v=>[String(v),v+" tok/s/user"]),String(TO.target));
  $("dl-all").checked=LY.all;
  const L=Infer.layouts(m,S.inf,topo,G), moe=Common.derive(m).nMoe>0;
  const cur=L.map(l=>({l, cu:toCurve({cfg:lyCfg(l)},isl,osl)}));
  /* 配置本身跑不起来的切法（例如 vLLM 下 Engram 的列数不能被 EP 整除）照样列出、标明原因，但不进包络和 KPI */
  for(const z of cur){ const ok=z.cu.fit&&z.cu.pts.length>0; z.inv=ok?z.cu.pts[0].r.warn.filter(w=>w.bad):[];
    const use=ok&&!z.inv.length; z.iso=use?isoThr(z.cu.pts,TO.target):null; z.u1=use?z.cu.pts[0].u:null; z.max=use?Math.max(...z.cu.pts.map(p=>p.thr)):null; z.on=false; }
  const ser=[];
  for(const fam of FAMS){ const zs=cur.filter(z=>z.l.fam===fam&&z.cu.fit&&!z.inv.length); if(!zs.length) continue;
    const pts=zs.flatMap(z=>z.cu.pts.map(p=>[p.u,p.thr,{note:t("deploy.ly.note",{name:lyName(z.l),b:p.b}),z}])), fr=frontier(pts);
    for(const p of fr) p[2].z.on=true;
    ser.push({name:t("deploy.ly.fam."+fam),short:t("deploy.ly.fam."+fam),col:FAMCOL[fam],pts:fr}); }
  lineChart("dl-pareto",{series:ser,xLab:t("tradeoff.pareto.xl"),yLab:t("tradeoff.pareto.ylOut"),vlines:[{x:TO.target,label:t("tradeoff.pareto.target",{v:TO.target})}],nearest:true,empty:t("deploy.ly.empty")});
  legendInto("dl-pareto-legend",ser.map(se=>({name:se.name,col:se.col,line:true})));
  $("dl-lede").innerHTML=t("deploy.ly.lede",{eng:esc(E),model:esc(m.short),hw:esc(hw.name),g:G,seq:esc(seqName),n:L.length,dense:moe?"":t("deploy.ly.dense")});
  const by=k=>cur.filter(z=>z[k]!=null).sort((a,b)=>b[k]-a[k])[0], bI=by("iso"), bU=by("u1"), bM=by("max"), now=cur.find(z=>isCur(z.l));
  kpis($("dl-kpis"),[
    [t("tradeoff.kpi.best",{target:TO.target}),bI?fN(bI.iso)+" tok/s/GPU":t("tradeoff.kpi.unreachable"),bI?lyName(bI.l):t("tradeoff.kpi.allSlower")],
    [t("tradeoff.kpi.fastest"),bU?fN(bU.u1)+" tok/s":"—",bU?lyName(bU.l):""],
    [t("tradeoff.kpi.maxThr"),bM?fN(bM.max)+" tok/s/GPU":"—",bM?lyName(bM.l):""],
    [t("deploy.ly.kpi.cur"),now?(now.iso!=null?fN(now.iso)+" tok/s/GPU":t("tradeoff.kpi.unreachable")):t("deploy.ly.kpi.notListed"),now?lyName(now.l)+" @"+TO.target:t("deploy.ly.kpi.custom")]]);
  const rows=cur.filter(z=>LY.all||z.on||z===now||z===bI);
  $("dl-tbl").innerHTML='<thead><tr><th></th><th>'+t("deploy.ly.head.layout")+'</th><th>'+t("deploy.ly.head.gpus")+'</th><th>MoE</th><th>'+t("tradeoff.cfg.head.wt")+'</th><th>'+t("tradeoff.cfg.head.cap")+'</th><th>'+t("deploy.ly.head.u1")+'</th><th>@'+TO.target+'</th><th>'+t("deploy.ly.head.max")+'</th><th></th></tr></thead><tbody>'+
    rows.map(z=>{ const l=z.l, c=lyCfg(l), on=z===now;
      return '<tr'+(on?' class="hl"':'')+'><td><span class="sw" style="display:inline-block;width:12px;height:3px;background:'+css(FAMCOL[l.fam])+'"></span></td><td style="min-width:13em">'+esc(lyName(l))+(z===bI?' <span class="tag">'+t("deploy.ly.best")+'</span>':"")+(z.inv.length?'<br><span class="tag" title="'+esc(z.inv.map(String).join("; "))+'">'+t("deploy.ly.invalid")+'</span>':"")+'</td><td class="num">'+t("deploy.ly.gpus",{n:l.N,k:l.inst})+'</td><td>'+(moe?esc(MOENAME[c.moePar]||c.moePar):"—")+'</td><td class="num">'+fB(z.cu.wt)+'</td><td class="num">'+(z.cu.fit?z.cu.cap+(z.cu.capBy!=="mem"?'<br><span class="tag">'+t("tradeoff.cfg.capBy."+z.cu.capBy,{mem:fInt(z.cu.mem.Bmax)})+'</span>':""):t("tradeoff.cfg.noFit"))+'</td><td class="num">'+(z.u1!=null?fN(z.u1):"—")+'</td><td class="num">'+(z.iso!=null?fN(z.iso):"—")+'</td><td class="num">'+(z.max!=null?fN(z.max):"—")+'</td><td>'+(on?t("tradeoff.cfg.current"):'<button type="button" class="mini" data-ly-apply="'+esc(l.id)+'">'+t("tradeoff.cfg.open")+'</button>')+'</td></tr>'; }).join("")+'</tbody>';
  $("dl-tbl-note").innerHTML=t("deploy.ly.tblNote",{shown:rows.length,all:cur.length});
  renderDeployMech(L, isl, osl);
  deployPub($("dl-pub"),["widep","hybrid"]);
}
/* 每卡并发 b 下拆开看：几种代表性的切法，同一个 Infer.layoutPoint；机制那张表的“当前配置”一栏也从这里填 */
function renderDeployMech(L, isl, osl){
  const {m,hw,topo,g}=ctx(), n=isl+Math.round(osl/2), moe=Common.derive(m).nMoe>0;
  xSeg("dl-b",[8,32,128].map(v=>[v,t("deploy.mech.b",{b:v})]),LY.b,v=>{ LY.b=+v; renderDeployMech(L,isl,osl); });
  const dep=L.filter(l=>l.fam==="dep"), Nx=dep.length?Math.max(...dep.map(l=>l.N)):0, tps=L.filter(l=>l.fam==="tp"&&l.N<=8), tpX=tps.length?tps[tps.length-1]:null;
  const pick=[tpX, L.find(l=>l.fam==="dcp"&&l.N===(tpX&&tpX.N)), ...[8,4,2].map(a=>L.find(l=>l.fam==="hybrid"&&l.N===Nx&&l.a===a)), L.find(l=>l.fam==="dep"&&l.N===Nx), L.find(l=>l.fam==="hybridDcp"&&l.N===Nx&&l.a===8)].filter(Boolean);
  const P=pick.map(l=>({l, p:Infer.layoutPoint(m,hw,topo,lyCfg(l,{n}),g,LY.b)}));
  table($("dl-mech"),[t("deploy.ly.head.layout"),t("deploy.mech.head.w"),t("deploy.mech.head.cap"),t("deploy.mech.head.rows"),t("deploy.mech.head.I"),t("deploy.mech.head.attn"),t("deploy.mech.head.moe"),t("deploy.mech.head.comm"),t("deploy.mech.head.step"),"tok/s/GPU"],
    P.map(({l,p})=>({cls:isCur(l)?"hl":"",cells:[esc(lyName(l))+(p.fit?"":' <span class="tag">'+t("deploy.mech.noFit",{cap:fInt(p.cap)})+'</span>'),fB(p.wGpu),fInt(p.cap),moe?fI(p.rowsPer):"—",moe?fI(p.I)+" / "+fInt(p.ridge):"—",fT(p.tAttn),moe?fT(p.tMoe):"—",fT(p.tComm),fT(p.stepT),fN(p.tok)]})),[1,2,3,4,5,6,7,8,9]);
  $("dl-mech-lede").innerHTML=t("deploy.mech.lede",{b:LY.b,n:fK(n)});
  /* 机制表的“当前配置”：TP 一侧取不超过 8 的最大 TP，宽 EP 一侧取最宽的 DEP */
  const a=P.find(z=>z.l===tpX), d=P.find(z=>z.l.fam==="dep"&&z.l.N===Nx);
  const fill=(id,html)=>{ const el=$(id); if(el) el.innerHTML=html; };
  if(!moe||!a||!d){ for(const k of ["r1","r2","r3"]) fill("dw-"+k,t("deploy.widep.now.none")); return; }
  const A={tp:lyName(a.l),dep:lyName(d.l),b:LY.b};
  fill("dw-r1",t("deploy.widep.now.r1",Object.assign({},A,{r1:fI(a.p.rowsPer),i1:fI(a.p.I),r2:fI(d.p.rowsPer),i2:fI(d.p.I),ridge:fInt(d.p.ridge),m1:fT(a.p.tMoe),m2:fT(d.p.tMoe)})));
  fill("dw-r2",t("deploy.widep.now.r2",Object.assign({},A,{c1:fT(a.p.tComm),c2:fT(d.p.tComm),a1:fT(a.p.tAttn),a2:fT(d.p.tAttn)})));
  fill("dw-r3",t("deploy.widep.now.r3",Object.assign({},A,{w1:fB(a.p.wGpu),w2:fB(d.p.wGpu),k1:fInt(a.p.cap),k2:fInt(d.p.cap)})));
}

/* ---- DCP ---- */
function renderDeployDcp(){
  const {m,hw,topo,g}=ctx(), E=ENGN[S.inf.engine]||S.inf.engine;
  let c=Object.assign(clone(S.inf),{phase:"decode",specK:0}), sw=Infer.dcpSweep(m,hw,topo,c,g,256), fb=false;
  if(sw.why==="dcpNoTp"){ fb=true; c=Object.assign(c,{N:Math.min(8,topo.gpus),dpA:1,moePar:compatMoe(c.engine,"tp")}); sw=Infer.dcpSweep(m,hw,topo,c,g,256); }
  if(sw.why){ $("dd-lede").innerHTML=t("deploy.dcp.why."+sw.why,{eng:esc(E),model:esc(m.short)}); $("dd-tbl").innerHTML=""; $("dd-read").innerHTML=""; deployPub($("dd-pub"),["dcp"]); return; }
  const base=sw.rows[0];
  $("dd-lede").innerHTML=t("deploy.dcp.lede",{eng:esc(E),model:esc(m.short),hw:esc(hw.name),tp:sw.tpA,top:sw.top,B:sw.B,n:fK(sw.n),algo:sw.algo,kind:m.full.kind.toUpperCase(),fb:fb?t("deploy.dcp.fallback",{n:c.N}):""});
  table($("dd-tbl"),["DCP",t("deploy.dcp.head.kv"),t("deploy.dcp.head.cap"),t("deploy.dcp.head.step",{B:sw.B}),t("deploy.dcp.head.comm"),t("deploy.dcp.head.full"),t("deploy.dcp.head.fullTpot"),"tok/s/GPU"],
    sw.rows.map(r=>({cls:!fb&&r.dcp===Math.max(1,S.inf.dcp||1)?"hl":"",cells:[String(r.dcp),fB(r.kvTok)+(r.dcp>1?' <span class="tag">÷'+fI(base.kvTok/r.kvTok)+'</span>':""),fInt(r.Bmax)+(r.dcp>1&&base.Bmax>0?' <span class="tag">×'+fI(r.Bmax/base.Bmax)+'</span>':""),
      r.fit?fT(r.stepT):t("deploy.dcp.noFit"),r.dcp>1?fT(r.dcpT):"—",fInt(r.Bfull),fT(r.fullTpot),fN(r.fullTok)]})),[1,2,3,4,5,6,7]);
  const last=sw.rows[sw.rows.length-1];
  $("dd-read").innerHTML=sw.rows.length>1?t("deploy.dcp.read",{d:last.dcp,cap:fI(base.Bmax>0?last.Bmax/base.Bmax:0),comm:fT(last.dcpT),step:fT(last.stepT),tok:fI(base.fullTok>0?last.fullTok/base.fullTok:0),lin:m.lin&&m.lin.n>0?t("deploy.dcp.lin",{kind:m.lin.kind}):""}):t("deploy.dcp.one");
  deployPub($("dd-pub"),["dcp"]);
}

/* ---- chunked prefill ---- */
function renderDeployChunk(){
  const {m,hw,topo,g}=ctx(), E=ENGN[S.inf.engine]||S.inf.engine, c=Object.assign(clone(S.inf),{phase:"decode"});
  const r=Infer.chunked(m,hw,topo,c,g), cur=r.rows.find(x=>x.C===c.C)||r.rows[r.rows.length-1];
  const mode=r.mixed?t("deploy.chunk.mode.mixed"):r.k>0?t("deploy.chunk.mode.interval",{k:r.k}):t("deploy.chunk.mode.sep");
  $("dc-lede").innerHTML=t("deploy.chunk.lede",{eng:esc(E),model:esc(m.short),hw:esc(hw.name),N:c.N,B:r.B,n:fK(r.n),P:fK(r.P),osl:fInt(c.osl),mode,dec:fT(r.tDec),demand:fInt(r.demand)});
  const pick=x=>r.mixed?x.mix||x.sep:x.sep, pc=pick(cur);
  kpis($("dc-kpis"),[
    [t("deploy.chunk.kpi.stall",{C:fK(cur.C)}),fT(pc.stall),t("deploy.chunk.kpi.stallSub",{dec:fT(r.tDec)})],
    [t("deploy.chunk.kpi.ttft"),fT(pc.ttft),t("deploy.chunk.kpi.ttftSub",{t:fT(cur.ttftAlone)})],
    [t("deploy.chunk.kpi.tpot"),fT(pc.tpot),t("deploy.chunk.kpi.tpotSub",{dec:fT(r.tDec)})],
    [t("deploy.chunk.kpi.tok"),fN(pc.tokPerGpu)+" tok/s/GPU",r.mixed&&cur.mix&&cur.mix.over?t("deploy.chunk.kpi.over"):t("deploy.chunk.kpi.tokSub"),r.mixed&&cur.mix&&cur.mix.over]]);
  const note=(x,s)=>({note:t("deploy.chunk.note",{C:fK(x.C),tpot:fT(s.tpot)})});
  const ser=[{name:t("deploy.chunk.ser.sep",{k:r.k}),short:t("deploy.chunk.ser.sepShort"),col:"s1",pts:r.rows.map(x=>[x.sep.ttft*1e3,x.sep.stall*1e3,note(x,x.sep)])},
    {name:t("deploy.chunk.ser.mix"),short:t("deploy.chunk.ser.mixShort"),col:"s2",pts:r.rows.filter(x=>x.mix).map(x=>[x.mix.ttft*1e3,x.mix.stall*1e3,note(x,x.mix)])}];
  const yTop=Math.max(1e-9,...ser.flatMap(se=>se.pts.map(p=>p[1])).filter(isFinite))*2;   // 对数轴顶上留白，线尾标注不压 y 轴标题
  lineChart("dc-fig",{series:ser,xLab:t("deploy.chunk.xl"),yLab:t("deploy.chunk.yl"),nearest:true,xLog:true,yLog:true,yMax:yTop,xFmt:v=>fN(v),yFmt:v=>fN(v),empty:t("deploy.chunk.empty")});
  legendInto("dc-fig-legend",ser.map(se=>({name:se.name,col:se.col,line:true})));
  table($("dc-tbl"),[t("deploy.chunk.head.C"),t("deploy.chunk.head.n"),t("deploy.chunk.head.sepStall"),t("deploy.chunk.head.sepTtft"),t("deploy.chunk.head.sepTpot"),t("deploy.chunk.head.Cp"),t("deploy.chunk.head.mixStall"),t("deploy.chunk.head.mixTtft"),t("deploy.chunk.head.mixTpot"),t("deploy.chunk.head.f")],
    r.rows.map(x=>({cls:x===cur?"hl":"",cells:[fK(x.C),String(x.nChunks),fT(x.sep.stall),fT(x.sep.ttft),fT(x.sep.tpot),x.mix?fInt(x.Cp):'<span class="tag">'+t("deploy.chunk.noRoom")+'</span>',x.mix?fT(x.mix.stall):"—",x.mix?fT(x.mix.ttft):"—",x.mix?fT(x.mix.tpot):"—",
      x.mix?(x.mix.over?'<span class="tag">'+t("deploy.chunk.over",{f:fI(x.mix.f)})+'</span>':fI(x.mix.f*100)+"%"):"—"]})),[1,2,3,4,5,6,7,8,9]);
  deployPub($("dc-pub"),["chunk"]);
}
function renderDeployPage(){ renderDeployLayouts(); renderDeployDcp(); renderDeployChunk(); }

$("dl-seq").addEventListener("change",e=>{ TO.seq=e.target.value; renderDeployLayouts(); renderInferTO(); });
$("dl-target").addEventListener("change",e=>{ TO.target=+e.target.value; renderDeployLayouts(); renderInferTO(); });
$("dl-all").addEventListener("change",e=>{ LY.all=e.target.checked; renderDeployLayouts(); });
$("dl-tbl").addEventListener("click",e=>{ const b=e.target.closest("[data-ly-apply]"); if(!b) return;
  const l=Infer.layouts(S.model,S.inf,TOPO[S.topo],TOPO[S.topo].gpus).find(x=>x.id===b.dataset.lyApply); if(!l) return;
  S.inf=Object.assign(S.inf,{N:l.cfg.N,dpA:l.cfg.dpA,dcp:l.cfg.dcp,moePar:compatMoe(S.inf.engine,l.cfg.moePar)}); S.infPreset="custom";
  syncInputs(); location.hash="infer/step"; render(); });
