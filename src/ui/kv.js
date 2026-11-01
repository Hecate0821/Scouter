/* ================= 推理 · KV cache：每种层存什么、多轮 agent 会话留得住多少、分层存储下每轮的 TTFT（KV.plan） ================= */
const KV_ENG = ["sglang","vllm","trtllm","tokenspeed"];
const KV_ROWS = ["prefix","evict","swa","lin","dtype","tier","route","agent"];
const KV_ECO = ["dynamo","lmcache","mooncake","flashinfer","llmd"];
function kvRun(over){ const {m,hw,topo,g}=ctx(); return KV.plan(m,hw,topo,Object.assign({},S.inf,S.kv,over||{}),g); }

function renderKV(){
  const {m,hw}=ctx(), c=S.kv, ci=S.inf, r=kvRun(), pt=r.perTok, fr=r.frac;
  setWarn(r.warn);
  const kvDt=ci.kvDt.toUpperCase(), lay=r.layers;
  const mix=[lay.full?t("kv.mix.full",{n:lay.full,kind:esc(m.full.kind.toUpperCase())}):"",lay.swa?t("kv.mix.swa",{n:lay.swa,w:fInt(pt.W)}):"",lay.lin?t("kv.mix.lin",{n:lay.lin,kind:esc(m.lin.kind)}):""].filter(Boolean).join(t("kv.mix.sep"));
  $("kv-lede").innerHTML=t("kv.lede",{model:esc(m.short),hw:hw.name,engine:ENGN[ci.engine],n:r.N,tp:+r.tpA.toFixed(2),dp:r.dpA,mix,kv:kvDt,
    sys:fInt(c.sys),turnIn:fInt(c.turnIn),turnOut:fInt(c.turnOut),turns:c.turns,sessions:fInt(c.sessions)});
  const tokB=pt.full+pt.swa;
  kpis($("kv-kpis"),[
    [t("kv.kpi.tok"),fB(tokB),t("kv.kpi.tokFoot",{kv:kvDt})],
    [t("kv.kpi.snap"),lay.lin?fB(pt.snap):"—",lay.lin?t("kv.kpi.snapFoot",{n:pt.full>0?fInt(pt.snapTok):"—"}):t("kv.kpi.snapNone")],
    [t("kv.kpi.fit"),fInt(r.fit.hbm),t("kv.kpi.fitFoot",{ws:fB(r.ws1)}),r.fit.hbm<c.sessions],
    [t("kv.kpi.hit"),pct(r.hitRate,1),t("kv.kpi.hitFoot",{hbm:pct(fr.hbm,1)})],
    [t("kv.kpi.ttft"),fT(r.ttft),t("kv.kpi.ttftFoot",{none:fT(r.ttftNone),hit:fT(r.ttftHit)})]]);

  // ---- 空间：每个请求的缓存字节随上下文 ----
  const dts=r.dtypes, cur=dts.find(d=>d.kvDt===ci.kvDt)||dts[0], ser=[{name:t("kv.space.this",{kv:kvDt}),short:t("kv.space.thisShort"),col:"c-kv",pts:r.space.map(x=>[x.L,x.bytes])}];
  for(const d of dts) if(d.kvDt!==ci.kvDt && d.full+d.swa<cur.full+cur.swa){ const k=d.full+d.swa>0?(d.full+d.swa)/(cur.full+cur.swa):1;
    ser.push({name:t("kv.space.dt",{kv:d.kvDt.toUpperCase()}),short:d.kvDt.toUpperCase(),col:d.kvDt==="fp8"?"s3":"s4",dash:"5 4",pts:r.space.map(x=>[x.L,x.full*k+x.swa*k+x.lin])}); }
  if(lay.swa||lay.lin) ser.push({name:t("kv.space.allFull"),short:t("kv.space.allFullShort"),col:"c-oth",dash:"2 3",pts:r.space.map(x=>[x.L,x.allFull])});
  lineChart("kv-space",{series:ser,xLog:true,xBase:4,yLog:true,xFmt:fK,yFmt:fB,xLab:t("kv.space.x"),yLab:t("kv.space.y"),tipFmt:fB});
  const at=L=>r.space.find(x=>x.L===L)||r.space[r.space.length-1], s128=at(131072), s1m=at(1048576);
  const rows=[];
  if(lay.full) rows.push([t("kv.layer.full",{kind:esc(m.full.kind.toUpperCase())}),lay.full,fB(pt.full),fB(s128.full),t("kv.need.full",{page:r.page})]);
  if(lay.swa) rows.push([t("kv.layer.swa",{w:fInt(pt.W)}),lay.swa,fB(pt.swa),fB(s128.swa),t("kv.need.swa",{w:fInt(pt.W)})]);
  if(lay.lin) rows.push([t("kv.layer.lin",{kind:esc(m.lin.kind)}),lay.lin,"—",fB(pt.snap),t("kv.need.lin",{gran:fInt(r.gran)})]);
  table($("kv-layers"),[t("kv.layers.head.type"),t("kv.layers.head.n"),t("kv.layers.head.tok"),t("kv.layers.head.at128k"),t("kv.layers.head.need")],rows,[1,2,3]);
  table($("kv-dt"),[t("kv.dt.head.dt"),t("kv.dt.head.tok"),t("kv.dt.head.at128k"),t("kv.dt.head.fit")],
    dts.map(d=>({cls:d.kvDt===ci.kvDt?"hl":"",cells:[d.kvDt.toUpperCase(),fB(d.full+d.swa),fB(d.at128k),fInt(d.perGpu128k*r.dpA)]})),[1,2,3]);
  const save=(lay.swa||lay.lin)&&s1m.allFull>0 ? s1m.allFull/s1m.bytes : 0;
  $("kv-space-read").innerHTML=ul([
    save>1?t("kv.space.readHybrid",{x:V(fX(save)),at:"1M"}):t("kv.space.readFull"),
    lay.lin?t("kv.space.readSnap",{snap:fB(pt.snap),n:pt.full>0?fInt(pt.snapTok):"—"}):"",
    dts.length>1?t("kv.space.readDt",{fp8:fInt(dts[1].perGpu128k*r.dpA),bf16:fInt(dts[0].perGpu128k*r.dpA),fp4:fInt(dts[2].perGpu128k*r.dpA)}):""]);

  // ---- 命中：会话数 → 驻留比例和命中率 ----
  const tiers=[{name:t("kv.hit.hbm"),col:"c-kv",o:{hostGB:0,remoteGB:0}}];
  if(c.hostGB>0) tiers.push({name:t("kv.hit.host",{gb:fInt(c.hostGB)}),col:"s2",o:{remoteGB:0}});
  if(c.remoteGB>0) tiers.push({name:t("kv.hit.remote",{gb:fInt(c.remoteGB)}),col:"s4",o:{}});
  const hitSer=tiers.map(x=>{ const rr=Object.keys(x.o).length?kvRun(x.o):r; return {name:x.name,col:x.col,pts:rr.sweep.map(s=>[s.sessions,s.hitRate])}; });
  hitSer.push({name:t("kv.hit.none"),col:"c-oth",dash:"2 3",pts:r.sweep.map(s=>[s.sessions,0])});
  lineChart("kv-hit",{series:hitSer,xLog:true,xBase:2,xFmt:fN,yFmt:v=>(v*100).toFixed(0)+"%",yMin:0,yMax:1,xLab:t("kv.hit.x"),yLab:t("kv.hit.y"),
    vlines:[{x:c.sessions,label:t("kv.hit.now",{n:fInt(c.sessions)})}]});
  stackBar($("kv-bar"),$("kv-barlist"),[{label:t("kv.frac.hbm"),v:fr.hbm,color:css("c-kv")},{label:t("kv.frac.host"),v:fr.host,color:css("s2")},
    {label:t("kv.frac.remote"),v:fr.remote,color:css("s4")},{label:t("kv.frac.miss"),v:fr.miss,color:css("c-oth")}],1,x=>t("kv.frac.n",{n:fInt(x*c.sessions)}));
  $("kv-hit-read").innerHTML=ul([
    t("kv.hit.readFit",{hbm:fInt(r.fit.hbm),host:c.hostGB>0?t("kv.hit.readHost",{n:fInt(r.fit.host)}):"",ws:fB(r.ws1)}),
    c.routing==="random"&&c.replicas>1?t("kv.hit.readRandom",{r:c.replicas,p:pct(1/c.replicas)}):t("kv.hit.readAware"),
    t("kv.hit.readCliff")]);

  // ---- 速度：每一轮的 TTFT ----
  const T=r.turns, tx=x=>x.t;
  const turnSer=[{name:t("kv.turn.none"),col:"c-oth",dash:"2 3",pts:T.map(x=>[tx(x),x.none])},{name:t("kv.turn.miss"),col:"s5",pts:T.map(x=>[tx(x),x.miss])},
    {name:t("kv.turn.host",{bw:fInt(hw.host)}),col:"s2",pts:T.map(x=>[tx(x),x.host])},{name:t("kv.turn.remote",{bw:fInt(hw.ib)}),col:"s4",pts:T.map(x=>[tx(x),x.remote])},
    {name:t("kv.turn.hbm"),col:"c-kv",pts:T.map(x=>[tx(x),x.hbm])}];
  lineChart("kv-turn",{series:turnSer,yLog:true,xFmt:v=>String(Math.round(v)),yFmt:fT,xLab:t("kv.turn.x"),yLab:t("kv.turn.y")});
  const last=T[T.length-1];
  $("kv-turn-lede").innerHTML=t("kv.turn.lede",{host:fInt(hw.host),ib:fInt(hw.ib),ovl:r.overlap?t("kv.turn.ovl"):t("kv.turn.serial")});
  $("kv-turn-read").innerHTML=ul([
    t("kv.turn.readLast",{t:last.t,L:fK(Math.round(last.Lin/1024)*1024),none:fT(last.none),hit:fT(last.hbm),x:V(fX(last.none/last.hbm))}),
    t("kv.turn.readLoad",{bytes:fB(last.bytes),load:fT(last.loadHost),remote:fT(last.loadRemote),re:fT(last.miss)}),
    !r.overlap?t("kv.turn.readSerial",{load:fT(last.loadHost)}):last.loadHost<=last.hbm?t("kv.turn.readHide",{load:fT(last.loadHost),hit:fT(last.hbm)}):t("kv.turn.readExposed",{load:fT(last.loadHost),hit:fT(last.hbm)}),
    hw.grace?t("kv.turn.readGrace"):t("kv.turn.readPcie",{bw:fInt(hw.host)})]);

  // ---- 引擎做法（文字和链接都在 locales/kv.js） ----
  table($("kv-eng"),["",...KV_ENG.map(e=>t("kv.eng.head."+e))],KV_ROWS.map(k=>['<span class="nw">'+t("kv.eng.row."+k)+'</span>',...KV_ENG.map(e=>t("kv.eng."+k+"."+e))]));
  $("kv-eco").innerHTML=ul(KV_ECO.map(k=>t("kv.eco."+k)));
  // ---- 公开数字 ----
  table($("kv-pub"),[t("kv.pub.head.date"),t("kv.pub.head.src"),t("kv.pub.head.sys"),t("kv.pub.head.setting"),t("kv.pub.head.value")],
    DB.set("kv_public").slice().sort((x,y)=>y.date.localeCompare(x.date))
      .map(p=>['<span class="v">'+esc(p.date)+'</span>','<span class="nw">'+aOut(p.url,p.src)+'</span>',esc(p.system)+(p.hw&&p.hw!=="—"?"<br>"+'<span class="tag">'+esc(p.hw)+'</span>':""),esc(p.setting),esc(p.value)]));
}
