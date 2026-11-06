/* ---------- 推理 · 部署与并行：推荐配置的取舍（同一个 Infer.decode / prefill） ---------- */
const TO = { seq:"8k1k", target:50, total:false };
const SEQS = { "1k1k":[1024,1024,t("tradeoff.seq.1k1k")],"1k8k":[1024,8192,t("tradeoff.seq.1k8k")], "8k1k":[8192,1024,t("tradeoff.seq.8k1k")], "32k2k":[32768,2048,t("tradeoff.seq.32k2k")], "128k1k":[131072,1024,t("tradeoff.seq.128k1k")] };
/* DeepInfer 的形状：prefill 取放得下的最小 EP（从 2 起翻倍），decode 取拓扑里能拿到的最宽 EP（最多 64），attention 全 DP，PD 分离 */
function diConfig(mk, topoKey, hwKey){
  const m=mk===S.modelKey?S.model:MODELS[mk], hw=HW[hwKey||S.hw], G=TOPO[topoKey].gpus; if(!m.moe.E) return null;
  const p=Infer.fitShards(m,hw);
  const N=Math.min(64,G), f8=m.wDefault==="fp8";
  return { eng:"DeepInfer", id:"deepinfer", label:t("tradeoff.di.label",{p,n:N}), pd:{N:Math.min(p,G)},
    src:t("tradeoff.di.src",{p,n:N,ib:N>TOPO[topoKey].domain?t("tradeoff.di.srcIb"):""}),
    cfg:{ engine:"sglang", N, dpA:N, moePar:"deepep", B:96, wDt:m.wDefault, dDt:f8?"fp8":"bf16", kvDt:"fp8", stateDt:"fp32" } };
}
function toConfigs(topoKey, hwKey){
  const L=[];
  for(const eng of ["sglang","vllm"]) for(const p of infPresets(eng,S.modelKey,topoKey)) if(p.cfg && p.cfg.N<=TOPO[topoKey].gpus)
    L.push({ eng:ENGN[eng], id:eng+":"+p.id, preset:p.id, engine:eng, label:p.label, src:p.src, cfg:Object.assign({engine:eng},p.cfg,{moePar:compatMoe(eng,p.cfg.moePar)}) });
  const di=diConfig(S.modelKey,topoKey,hwKey); if(di) L.push(di);
  return L;
}
const toShort = cf => cf.eng+" "+t("common.fmt.presetShort",{s:cf.label});
/* 一条 Pareto 曲线：每个 DP rank 的并发从 1 扫到显存 / 引擎上限 */
function toCurve(cf, isl, osl, hwKey, topoKey){
  const {m,g}=ctx(), hw=HW[hwKey||S.hw], topo=TOPO[topoKey||S.topo];
  const base=Object.assign(clone(BASE_INF),cf.cfg,{phase:"decode"});
  const probe=Infer.decode(m,hw,topo,Object.assign({},base,{B:1,n:isl+osl}),g), E=Infer.ENGINES[base.engine];
  const lim=base.specK>0?E.specCap:256, cap=Math.min(probe.mem.Bmax, lim), capBy=probe.mem.Bmax<=lim?"mem":base.specK>0?"spec":"sweep", pts=[];   // 256：这张图只扫到每 rank 256 并发
  if(probe.mem.avail<=0 || cap<1) return { pts, fit:false, wt:probe.mem.weights, cap:Math.max(0,cap), mem:probe.mem };
  const preC=Object.assign({},base,{phase:"prefill",specK:0,C:Math.min(isl,8192),P:isl,p:0});
  const pre=cf.pd ? Infer.prefill(m,hw,topo,Object.assign({},preC,{N:cf.pd.N,dpA:cf.pd.N}),g) : Infer.prefill(m,hw,topo,preC,g);
  const ladder=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256].filter(b=>b<cap).concat([cap]);
  for(const b of ladder){
    const r=Infer.decode(m,hw,topo,Object.assign({},base,{B:b,n:isl+osl/2}),g), out=r.tokPerStep*r.dpA;
    /* 聚合部署：每步完成 out/osl 个请求，要顺带 prefill 同样多个新请求（摊到 DP rank 上）；PD 分离：prefill 卡按卡秒摊进分母 */
    const preT = cf.pd ? 0 : out/osl*pre.ttft/Math.max(1,r.dpA), t=r.stepT+preT;
    const gpus = cf.pd ? base.N + out/(t*osl)*pre.ttft : base.N;   // prefill 实例 DEP：每张卡一条 prompt，一条占 ttft 卡秒
    pts.push({ b, B:b*r.dpA, u:r.tokPerStep/r.Breq/t, thr:out/t/gpus*(TO.total?(isl+osl)/osl:1), t, preT, r });
  }
  /* 空载 TTFT：prefill 一条 isl 的 prompt；PD 分离时再加上把这条请求的 KV 搬到 decode 实例（Infer.pdXfer） */
  const xfer = cf.pd ? Infer.pdXfer(m,hw,topo,base,g,isl,cf.pd.N).t : 0;
  return { pts, fit:true, wt:probe.mem.weights, cap, capBy, mem:probe.mem, pre, xfer, ttft:pre.ttft+xfer };
}
function isoThr(pts, u){ let best=null;
  for(let i=0;i<pts.length;i++){ const p=pts[i], q=pts[i+1];
    if(p.u>=u) best=Math.max(best??0,p.thr);
    if(q && p.u>=u && q.u<u){ const a=(p.u-u)/(p.u-q.u); best=Math.max(best??0,p.thr+a*(q.thr-p.thr)); } }
  return best; }
function renderInferTO(){
  const {m,hw,topo,g}=ctx(), [isl,osl,seqName]=SEQS[TO.seq];
  fillOpts($("i-to-seq"),Object.entries(SEQS).map(([k,v])=>[k,v[2]]),TO.seq);
  fillOpts($("i-to-target"),[10,20,50,100,200,400].map(v=>[String(v),v+" tok/s/user"]),String(TO.target));
  $("i-to-total").checked=TO.total;
  const cfgs=toConfigs(S.topo), curves=cfgs.map((cf,i)=>Object.assign({cf,col:cf.eng==="DeepInfer"?"c-lin":"s"+Math.min(i+1,6)},toCurve(cf,isl,osl)));
  const ser=curves.filter(cu=>cu.fit).map(cu=>({name:cu.cf.eng+" · "+cu.cf.label,short:toShort(cu.cf),col:cu.col,pts:cu.pts.map(p=>[p.u,p.thr,{note:t("tradeoff.pareto.note",{b:p.b})}])}));
  const yl=TO.total?t("tradeoff.pareto.ylTotal"):t("tradeoff.pareto.ylOut");
  lineChart("i-to-pareto",{series:ser,xLab:t("tradeoff.pareto.xl"),yLab:yl,vlines:[{x:TO.target,label:t("tradeoff.pareto.target",{v:TO.target})}],nearest:true,empty:t("tradeoff.pareto.empty")});
  legendInto("i-to-pareto-legend",ser.map(se=>({name:se.name,col:se.col,line:true})));
  const iso=cu=>cu.fit?isoThr(cu.pts,TO.target):null, cur=curves.find(cu=>cu.cf.engine===S.inf.engine&&cu.cf.preset===S.infPreset), best=curves.map(cu=>({cu,v:iso(cu)})).filter(z=>z.v!=null).sort((a,b)=>b.v-a.v)[0];
  const fast=curves.filter(cu=>cu.fit&&cu.pts.length).sort((a,b)=>b.pts[0].u-a.pts[0].u)[0];
  kpis($("i-to-kpis"),[
    [t("tradeoff.kpi.best",{target:TO.target}),best?fN(best.v)+" tok/s/GPU":t("tradeoff.kpi.unreachable"),best?toShort(best.cu.cf):t("tradeoff.kpi.allSlower")],
    [t("tradeoff.kpi.fastest"),fast?fN(fast.pts[0].u)+" tok/s":"—",fast?t("tradeoff.kpi.fastestSub",{cfg:toShort(fast.cf)}):""],
    [t("tradeoff.kpi.maxThr"),(()=>{ const z=curves.filter(cu=>cu.fit).map(cu=>({cu,v:Math.max(...cu.pts.map(p=>p.thr))})).sort((a,b)=>b.v-a.v)[0]; return z?fN(z.v)+" tok/s/GPU":"—"; })(),t("tradeoff.kpi.maxThrSub")],
    [t("tradeoff.kpi.current"),cur?(iso(cur)!=null?fN(iso(cur))+" tok/s/GPU":t("tradeoff.kpi.unreachable")):t("tradeoff.kpi.notListed"),cur?toShort(cur.cf)+" @"+TO.target:t("tradeoff.kpi.custom",{engine:ENGN[S.inf.engine]})]]);
  $("i-to-cfg").innerHTML='<thead><tr><th></th><th>'+t("tradeoff.cfg.head.cfg")+'</th><th>'+t("tradeoff.cfg.head.gpus")+'</th><th>attention</th><th>MoE</th><th>'+t("tradeoff.cfg.head.wt")+'</th><th>'+t("tradeoff.cfg.head.cap")+'</th><th>'+t("tradeoff.cfg.head.ttft",{isl:fK(isl)})+'</th><th>@'+TO.target+'</th><th>'+t("tradeoff.cfg.head.src")+'</th><th></th></tr></thead><tbody>'+
    curves.map(cu=>{ const c=cu.cf.cfg, tp=c.N/c.dpA, on=cu===cur;
      return '<tr'+(on?' class="hl"':'')+'><td><span class="sw" style="display:inline-block;width:12px;height:3px;background:'+css(cu.col)+'"></span></td><td style="min-width:12em">'+esc(cu.cf.eng+" · "+cu.cf.label)+'</td><td class="num">'+(cu.cf.pd?t("tradeoff.cfg.gpusPd",{n:c.N,pd:cu.cf.pd.N}):c.N)+'</td><td>'+(tp>1?"TP"+tp:"")+(c.dpA>1?(tp>1?" × ":"")+"DP"+c.dpA:"")+(tp===1&&c.dpA===1?t("tradeoff.cfg.single"):"")+(c.dcp>1?" + DCP"+c.dcp:"")+(c.specK?t("tradeoff.cfg.spec",{k:c.specK}):"")+'</td><td>'+(m.moe.E?esc(MOENAME[c.moePar]):"—")+'</td><td class="num">'+fB(cu.wt)+'</td><td class="num">'+(cu.fit?cu.cap+(cu.capBy!=="mem"?'<br><span class="tag">'+t("tradeoff.cfg.capBy."+cu.capBy,{mem:fInt(cu.mem.Bmax)})+'</span>':""):t("tradeoff.cfg.noFit"))+'</td><td class="num">'+(cu.fit?fT(cu.ttft)+(cu.xfer>0?'<br><span class="tag">'+t("tradeoff.cfg.xfer",{t:fT(cu.xfer)})+'</span>':""):"—")+'</td><td class="num">'+(iso(cu)!=null?fN(iso(cu)):"—")+'</td><td style="min-width:20em;font-size:12.5px">'+esc(cu.cf.src)+'</td><td>'+(on?t("tradeoff.cfg.current"):'<button type="button" class="mini" data-to-apply="'+esc(cu.cf.id)+'">'+t("tradeoff.cfg.open")+'</button>')+'</td></tr>'; }).join("")+'</tbody>';
  /* 随上下文 */
  const isls=[1024,2048,4096,8192,16384,32768,65536,131072];
  const serC=curves.map(cu=>({name:cu.cf.eng+" · "+cu.cf.label,short:toShort(cu.cf),col:cu.col,pts:isls.map(L=>{ const c2=toCurve(cu.cf,L,osl); return [L,c2.fit?isoThr(c2.pts,TO.target):null]; })}));
  lineChart("i-to-ctx",{series:serC,xLog:true,xBase:2,xFmt:fK,xLab:t("tradeoff.ctx.xl",{osl:fK(osl)}),yLab:t("tradeoff.ctx.yl",{target:TO.target}),labelEnds:false,empty:t("tradeoff.ctx.empty")});
  legendInto("i-to-ctx-legend",serC.map(se=>({name:se.name,col:se.col,line:true})));
  $("i-to-ctx-lede").textContent=t("tradeoff.ctx.lede",{target:TO.target});
  /* 怎么选 */
  const di=curves.find(cu=>cu.cf.eng==="DeepInfer"), b1=di&&di.fit?di.pts[0].r:null;
  const diNote = !di ? t("tradeoff.rules.diDense") : !b1 ? t("tradeoff.rules.diNoFit") : (()=>{ const mm=b1.mem, nonExp=mm.wAttn+mm.wLin+mm.wMoeRep+mm.wDense+mm.wEmb, tW=Common.memT(nonExp,hw,g);
    return t("tradeoff.rules.diPremise",{model:esc(m.short),bytes:fB(nonExp),read:fT(tW),step:fT(b1.stepT),u:fN(di.pts[0].u),verdict:tW>0.01?t("tradeoff.rules.diBad"):t("tradeoff.rules.diOk")}); })();
  $("i-to-rules").innerHTML=[
    t("tradeoff.rules.fit",{model:esc(m.short),bytes:fB(Common.derive(m).total*Common.dtB(m.wDefault)),dt:m.wDefault.toUpperCase(),mem:hw.mem}),
    t("tradeoff.rules.tp"),
    t("tradeoff.rules.dp"),
    t("tradeoff.rules.dpAttn"),
    t("tradeoff.rules.wideEp",{note:diNote}),
    best?t("tradeoff.rules.best",{seq:esc(seqName),target:TO.target,cfg:esc(best.cu.cf.eng+" · "+best.cu.cf.label),thr:fN(best.v)}):t("tradeoff.rules.none",{target:TO.target})
  ].map(x=>"<li>"+x+"</li>").join("");
}
$("i-to-seq").addEventListener("change",e=>{ TO.seq=e.target.value; renderInferTO(); });
$("i-to-target").addEventListener("change",e=>{ TO.target=+e.target.value; renderInferTO(); });
$("i-to-total").addEventListener("change",e=>{ TO.total=e.target.checked; renderInferTO(); });
$("i-to-cfg").addEventListener("click",e=>{ const b=e.target.closest("[data-to-apply]"); if(!b) return;
  const cf=toConfigs(S.topo).find(x=>x.id===b.dataset.toApply); if(!cf) return;
  if(cf.preset){ S.inf.engine=cf.engine; applyInf(cf.preset); }
  else { S.inf=Object.assign(clone(BASE_INF),{n:S.inf.n,P:S.inf.P,C:S.inf.C},cf.cfg,{phase:"decode"}); S.infPreset="custom"; }
  syncInputs(); location.hash="infer/step"; render(); });

$("p-mesh-dim").addEventListener("click",e=>{ const b=e.target.closest("button[data-v]"); if(!b) return; meshDim=b.dataset.v; renderMesh(trnRun()); });
$("p-mesh").addEventListener("click",e=>{ const b=e.target.closest("button[data-k]"); if(!b) return; meshSel=+b.dataset.k; renderMesh(trnRun()); });
$("p-e-props").addEventListener("click",e=>{ const b=e.target.closest("[data-disp]"); if(!b) return; S.trn.dispatcher=b.dataset.disp; S.trnPreset="custom"; syncInputs(); render(); });
