/* ================= 架构：模型总览 / 模块 roofline / 跨模型对比 =================
   每个模型拆成模块（全注意力、滑窗、线性注意力、MoE、dense FFN），每个模块看“一层”。数字来自 Arch.measure（推理核心的参考画像 / 训练核心）。
   结构滑块直接改全局的模型副本 S.model（和右侧的模型编辑器是同一份），预设值 MODELS[key] 作为对照画成灰线 / 灰点。 */
const ArchUI = (function(){
const MODC = { full:"c-attn", swa:"c-cp", lin:"c-lin", moe:"c-moe", dense:"c-lm" };
const MODEL_COL = ["s1","s2","s3","s4","s5","s6"];
const archWl = () => Object.assign({}, S.arch);
const baseModel = () => MODELS[S.modelKey];
const modelEdited = () => JSON.stringify(S.model)!==JSON.stringify(baseModel());
const modName = mod => t("arch.mod."+mod.id+"."+mod.kind);
const kindName = k => t("arch.kind."+k);
const phaseName = p => t("arch.phase."+p);
const pillB = b => '<span class="pill '+b+'">'+esc(t("arch.bound."+b))+'</span>';
const modShape = id => Arch.isMlp(id) ? "sq" : id==="lin" ? "tri" : null;
const shortName = m => m.short.split(/[- ]/)[0];
function pickMod(m){ const ps=Arch.present(m); if(!ps.some(p=>p.id===S.arch.mod)) S.arch.mod=ps[0].id; return ps.find(p=>p.id===S.arch.mod); }
const fTF = v => fN(v/1e12)+" TFLOP/s";
const fX = x => isFinite(x) ? (x>=10?x.toFixed(0):x.toFixed(2))+"×" : "—";
/* 峰值键（bf16 / fp8 / fp4）或 dtype（mxfp4、nvfp4、fp8a…）都可以传 */
const pkOf = k => ["bf16","fp8","fp4"].includes(k) ? k : Common.peakKey(k);
const ridgeOf = (hw, g, k) => hw[pkOf(k)]*1e12*g.etaC/(hw.hbm*1e9*g.etaM);
const ctxList = [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576];
const bList = [1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024];
const cList = [512,1024,2048,4096,8192,16384,32768];
const wlText = wl => wl.phase==="decode" ? t("arch.wl.decode",{b:fInt(wl.B),n:fInt(wl.n)}) : wl.phase==="prefill" ? t("arch.wl.prefill",{n:fInt(wl.n),c:fInt(Math.min(wl.n,Arch.chunkOf(wl,HW[S.hw])))}) : t("arch.wl.train",{s:fInt(wl.S),p:wl.prec.toUpperCase()});

/* 扫一个负载参数：同一次渲染里同一组（模型, 参数, 取值, 负载）只算一遍，曲线和 roofline 轨迹共用。每次渲染开头清空（模型可能被原地改过） */
let SWEEP = new WeakMap();
function sweep(model, wl, key, xs){
  let c=SWEEP.get(model); if(!c) SWEEP.set(model, c=new Map());
  const id=key+"|"+xs.join(",")+"|"+JSON.stringify(wl); if(c.has(id)) return c.get(id);
  const {hw,g}=ctx(), out=xs.map(x=>[x, Arch.measure(model,hw,g,Object.assign({},wl,{[key]:x}))]);
  c.set(id,out); return out;
}
function measure(model, wl){ return sweep(model, wl, "_", [0])[0][1]; }

/* 右侧参数栏：只显示当前阶段用得上的 */
function archRack(){ const ph=S.arch.phase;
  $("ar-B").hidden=ph!=="decode"; $("ar-n").hidden=ph==="train"; $("ar-C").hidden=ph!=="prefill"; $("ar-S").hidden=ph!=="train";
  $("ar-inf").hidden=ph==="train"; $("ar-prec").hidden=ph!=="train"; }
$("ar-reset").addEventListener("click",()=>{ S.arch=Object.assign({mod:S.arch.mod}, Arch.defaults(S.model)); syncInputs(); render(); });

/* ---------- 模型总览 ---------- */
function cacheParts(mm){ const dd=derive(mm), f=mm.full, l=mm.lin, sw=f.swa||0;
  return { kv: (f.n-sw)*(dd.kvTokElems*2 + (f.kind==="dsa"?(f.idxD+4)/(f.kpool||1):0)), st: l.n*(dd.linState*4 + dd.linConvDim*(l.conv-1)*2) + sw*dd.kvTokElems*2*(f.W||0) }; }
function stackCells(m){
  const mods=Arch.present(m);
  return Common.layerPlan(m).map((c,i)=>{ const mod=mods.find(p=>p.id===c.attn), tip=t("arch.ov.cellTip",{i, attn:modName(mod), mlp:t("arch.ov.cell."+c.mlp)});
    return '<a href="#arch/module/'+c.attn+'" class="'+(c.mlp==="moe"?"moe":"")+'" style="background:'+css(MODC[c.attn])+'" title="'+esc(tip)+'" aria-label="'+esc(tip)+'"></a>'; }).join("");
}
function renderArchOverview(){
  SWEEP=new WeakMap(); archRack();
  const {m,d,hw}=ctx(), keys=Object.keys(MODELS), wl=archWl();
  table($("m-compare"),[t("arch.ov.th.model"),t("arch.ov.th.total"),t("arch.ov.th.active"),t("arch.ov.th.official"),t("arch.ov.th.ratio"),t("arch.ov.th.attn"),t("arch.ov.th.experts"),t("arch.ov.th.F"),t("arch.ov.th.kv"),t("arch.ov.th.state")],
    keys.map(k=>{ const mm=k===S.modelKey?S.model:MODELS[k], dd=derive(mm), p=cacheParts(mm);
      return {cls:k===S.modelKey?"hl":"",cells:[esc(mm.short),fN(dd.total),fN(dd.active),esc(mm.official),(dd.active/dd.total*100).toFixed(1)+"%",
        mm.full.n+" "+mm.full.kind.toUpperCase()+(mm.full.swa?esc(t("arch.ov.swaOf",{n:mm.full.swa,w:mm.full.W})):"")+(mm.lin.n?" + "+mm.lin.n+" "+mm.lin.kind:""),mm.moe.E?mm.moe.E+" / "+mm.moe.k+(mm.moe.Fs?" + shared":""):"dense",mm.moe.E?mm.moe.F+(mm.moe.Dlat?" @"+mm.moe.Dlat:""):"FFN "+mm.dense.F,fB(p.kv),p.st?fB(p.st):"—"]}; }),[1,2,4,6,7,8,9]);
  $("m-title").textContent=m.name;
  $("m-note").textContent=m.note+(m.mtp?t("arch.ov.mtp",{n:m.mtp}):"")+(modelEdited()?t("arch.ov.edited"):"");
  $("m-stack").innerHTML=stackCells(m);
  const mods=Arch.present(m);
  $("m-stack-legend").innerHTML=mods.filter(p=>!Arch.isMlp(p.id)).map(p=>'<span><span class="sw" style="background:'+css(MODC[p.id])+'"></span>'+esc(modName(p))+' × '+p.layers+'</span>').join("")+'<span>'+esc(t("arch.ov.moeBar",{moe:d.nMoe,dense:m.dense.n}))+'</span>';
  // 模块卡片：当前负载下一层的时间、强度、瓶颈
  const r=measure(m,wl), fx=Arch.facts(m,wl);
  $("ao-mods").innerHTML=mods.map(p=>{ const a=r.mods[p.id], fc=fx[p.id]||{};
    const mem = fc.kvTok!=null ? t("arch.ov.card.kv",{v:fB(fc.kvTok)}) : fc.state!=null ? t("arch.ov.card.state",{v:fB(fc.state)}) : fc.expert!=null ? t("arch.ov.card.experts",{e:m.moe.E,k:m.moe.k,v:fB(fc.expert)}) : t("arch.ov.card.weights",{v:fB(fc.weights||0)});
    return '<a class="card" href="#arch/module/'+p.id+'"><h4><span class="dot" style="background:'+css(MODC[p.id])+'"></span>'+esc(modName(p))+' × '+p.layers+'</h4>'+
      '<p>'+esc(t("arch.ov.card.desc."+p.kind))+'</p><p class="meta">'+esc(t("arch.ov.card.params",{p:fN(fc.params||0)}))+' · '+esc(mem)+'</p>'+
      '<p class="meta">'+esc(t("arch.ov.card.now",{ph:phaseName(wl.phase),t:fT(a.t),i:fI(a.I)}))+' '+pillB(a.bound)+'</p></a>'; }).join("");
  $("ao-mods-lede").textContent=t("arch.ov.modsLede",{ph:phaseName(wl.phase),hw:hw.name,wl:wlText(wl)});
  const pr=[[t("arch.ov.p.full"),d.pAttn,m.full.n],[t("arch.ov.p.rep"),d.pAttnRep+d.pIdx,m.full.n],[t("arch.ov.p.lin"),d.pLin,m.lin.n],[t("arch.ov.p.expert"),d.pExpert,0],[t("arch.ov.p.routed"),d.pRouted,d.nMoe],
    [t("arch.ov.p.shared"),d.pShared,d.nMoe],[t("arch.ov.p.lat"),d.pLat,d.nMoe],[t("arch.ov.p.router"),d.pRouter,d.nMoe],[t("arch.ov.p.dense"),d.pDense,m.dense.n],["embedding",d.pEmb,1],["lm_head",d.pHead,1]];
  table($("m-params"),[t("arch.ov.pth.part"),t("arch.ov.pth.each"),t("arch.ov.pth.n"),t("arch.ov.pth.sum"),t("arch.ov.pth.share")],
    pr.filter(x=>x[1]>0&&(x[2]>0||x===pr[3])).map(x=>{ const one=x[2]===0; return [esc(x[0]),fN(x[1]),one?"—":fInt(x[2]),one?"—":fN(x[1]*x[2]),one?"—":pct(x[1]*x[2],d.total)]; })
      .concat([{cls:"hl",cells:["<b>"+esc(t("arch.ov.p.total"))+"</b>","","",fN(d.total),esc(t("arch.ov.p.activeIs",{v:fN(d.active)}))]}]),[1,2,3,4]);
  // 缓存曲线
  const X0=128, X1=1048576, xs=logspace(X0,X1,140), series=[], colors=[css("c-attn"),css("c-lin"),css("c-moe"),css("c-a2a"),css("c-fsdp")], points=[];
  let ymax=0, ymin=Infinity;
  keys.forEach((k,i)=>{ const mm=k===S.modelKey?S.model:MODELS[k], p=cacheParts(mm);
    const pts=xs.map(n=>[n,(p.kv*n+p.st)/1e9]); pts.forEach(q=>{ ymax=Math.max(ymax,q[1]); ymin=Math.min(ymin,q[1]); });
    series.push({pts,color:colors[i%5],w:k===S.modelKey?3:1.4,dash:k===S.modelKey?null:"6 4"});
    if(p.st>0&&p.kv>0){ const ns=p.st/p.kv; if(ns>=X0&&ns<=X1) points.push({x:ns,y:2*p.st/1e9,color:colors[i%5],r:k===S.modelKey?5:3.5}); } });
  const cur=cacheParts(S.model), ci=keys.indexOf(S.modelKey);
  if(cur.st>0) series.push({pts:[[X0,cur.st/1e9],[X1,cur.st/1e9]],color:css("muted"),w:1.2,dash:"2 3",label:t("arch.ov.cache.state")});
  if(cur.kv>0&&cur.st>0){ series.push({pts:xs.map(n=>[n,cur.kv*n/1e9]),color:css("muted"),w:1.2,dash:"2 3"}); ymin=Math.min(ymin,cur.kv*X0/1e9); }
  plot($("m-cache"),{x0:X0,x1:X1,y0:Math.pow(10,Math.floor(Math.log10(Math.max(ymin,1e-4)))),y1:ymax*1.5,logx:true,logy:true,fx:fxPow,
    fy:v=>v>=1?v+" GB":v>=1e-3?+(v*1000).toPrecision(3)+" MB":+(v*1e6).toPrecision(3)+" KB",xl:t("arch.ov.cache.x"),yl:t("arch.ov.cache.y"),series,points,
    vlines:cur.st>0&&cur.kv>0?[{x:cur.st/cur.kv,color:colors[ci%5],label:t("arch.ov.cache.cross",{n:fN(cur.st/cur.kv)})}]:[]});
  $("m-cache-legend").innerHTML=keys.map((k,i)=>'<span style="color:'+colors[i%5]+'"><i'+(k===S.modelKey?'':' class="d"')+'></i>'+esc(MODELS[k].short)+'</span>').join("")+
    (cur.st>0?'<span style="color:var(--muted)"><i class="d"></i>'+esc(t("arch.ov.cache.parts",{m:S.model.short}))+'</span><span>'+esc(t("arch.ov.cache.dots"))+'</span>':'<span>'+esc(t("arch.ov.cache.noLin"))+'</span>');
}

/* ---------- 模块 roofline ---------- */
/* 结构滑块：[路径, 最小, 最大, 步长]；最大值写成字符串时取当前模型上的字段 */
const KNOBS = {
  gqa:[["full.Hq",4,128,4],["full.Hkv",1,128,1],["full.dh",32,256,32]],
  swa:[["full.W",256,32768,256],["full.swa",0,"full.n",1],["full.Hkv",1,128,1]],
  mla:[["full.H",8,128,8],["full.kvLora",64,2048,64],["full.rope",0,128,32],["full.nope",32,256,32],["full.vh",32,256,32]],
  dsa:[["full.topk",256,16384,256],["full.idxH",4,128,4],["full.idxD",32,256,32],["full.kpool",1,16,1],["full.kvLora",64,2048,64],["full.H",8,128,8]],
  gdn:[["lin.Hv",4,128,4],["lin.Hk",4,128,4],["lin.dk",32,256,32],["lin.dv",32,256,32]],
  kda:[["lin.Hv",4,128,4],["lin.Hk",4,128,4],["lin.dk",32,256,32],["lin.dv",32,256,32]],
  moe:[["moe.E",8,1024,8],["moe.k",1,32,1],["moe.F",128,8192,128],["moe.Fs",0,16384,256],["moe.Dlat",0,"D",256]],
  dense:[["dense.F",1024,65536,512]] };
let knobSig="";
function buildKnobs(mod){
  const m=S.model, b=baseModel(), list=(KNOBS[mod.kind]||[]).concat([["D",512,16384,256]]);
  const get=(o,p)=>p.split(".").reduce((x,k)=>x==null?x:x[k],o);
  let h="";
  if(mod.id==="full") h+='<label class="ctl"><span>'+esc(t("arch.knob.full.kind"))+'</span><select data-bind="model.full.kind"><option value="gqa">GQA</option><option value="mla">MLA</option><option value="dsa">'+esc(t("arch.editor.dsaOpt"))+'</option></select></label>';
  if(mod.id==="lin") h+='<label class="ctl"><span>'+esc(t("arch.knob.lin.kind"))+'</span><select data-bind="model.lin.kind"><option value="GDN">Gated DeltaNet</option><option value="KDA">KDA</option></select></label>';
  for(const [p,lo,hi,st] of list){ const mx=typeof hi==="string"?get(m,hi):hi, pv=get(b,p);
    h+='<label class="ctl"><span>'+t("arch.knob."+p)+(pv!=null?' <small>'+esc(t("arch.knob.preset",{v:fInt(pv)}))+'</small>':'')+'</span><output data-out="model.'+p+'"></output>'+
      '<input type="range" min="'+lo+'" max="'+Math.max(lo,mx)+'" step="'+st+'" data-bind="model.'+p+'"></label>'; }
  h+='<button type="button" class="mini" id="am-reset">'+esc(t("arch.knob.reset"))+'</button>';
  $("am-knobs").innerHTML=h;
  $("am-reset").onclick=()=>{ S.model=clone(baseModel()); buildEditor(); knobSig=""; render(); };
  syncInputs();
}
function chips(el, items, cur, onPick){
  el.innerHTML=items.map(it=>'<button type="button" data-v="'+esc(it.v)+'" aria-pressed="'+(it.v===cur)+'">'+it.html+'</button>').join("");
  el.querySelectorAll("button").forEach(b=>b.onclick=()=>onPick(b.dataset.v));
}
function renderArchModule(){
  SWEEP=new WeakMap(); archRack();
  const {m,hw,g}=ctx(), wl=archWl();
  if(Arch.ORDER.includes(S.anchor)){ S.arch.mod=S.anchor; S.anchor=""; }
  const mod=pickMod(m), mods=Arch.present(m), edited=modelEdited(), base=baseModel();
  const sig=S.modelKey+"|"+mod.id+"|"+mod.kind+"|"+m.full.kind+"|"+m.lin.kind;
  if(sig!==knobSig){ knobSig=sig; buildKnobs(mod); }
  chips($("am-mods"), mods.map(p=>({v:p.id, html:'<span class="sw" style="background:'+css(MODC[p.id])+'"></span>'+esc(modName(p))+' × '+p.layers})), mod.id, v=>{ location.hash="#arch/module/"+v; });
  chips($("am-phase"), ["decode","prefill","train"].map(p=>({v:p, html:esc(phaseName(p))})), wl.phase, v=>{ S.arch.phase=v; syncInputs(); render(); });
  const cur=measure(m,wl), base0=edited?measure(base,wl):cur, a=cur.mods[mod.id], a0=base0.mods[mod.id]||a;
  const fx=Arch.facts(m,wl)[mod.id]||{}, fx0=Arch.facts(base,wl)[mod.id]||{};
  const peaks=Arch.peaks(hw,wl), pk=Arch.peakKeyFor(mod.id,wl), ridge=ridgeOf(hw,g,pk), pkV=hw[pk]*1e12;
  $("am-title").textContent=t("arch.mod.title",{m:m.short,mod:modName(mod),n:mod.layers});
  $("am-lede").innerHTML=t("arch.mod.lede",{wl:esc(wlText(wl)),hw:esc(hw.name)})+(edited?" "+t("arch.mod.ledeEdited"):"");
  const totalAll=mods.reduce((s,p)=>s+cur.mods[p.id].t*p.layers,0);
  kpis($("am-kpis"),[
    [t("arch.kpi.t"),V(fT(a.t)),edited?t("arch.kpi.vsPreset",{v:fT(a0.t)}):t("arch.kpi.tLayers",{n:mod.layers,v:fT(a.t*mod.layers)})],
    [t("arch.kpi.share"),V(pct(a.t*mod.layers,totalAll)),t("arch.kpi.shareF")],
    [t("arch.kpi.I"),V(fI(a.I)),t("arch.kpi.IF",{r:fI(ridge),p:pk.toUpperCase()})],
    [t("arch.kpi.perf"),V(fTF(a.perf)),t("arch.kpi.perfF",{p:pct(a.perf,pkV),k:pk.toUpperCase()})],
    [t("arch.kpi.bytes"),V(fB(a.bytes)),t("arch.kpi.bytesF",{f:fF(a.flops)})],
    [t("arch.kpi.bound"),pillB(a.bound),t("arch.kpi.boundF",{c:pct(a.byBound.compute,a.t),m:pct(a.byBound.memory,a.t),l:pct(a.byBound.launch,a.t)})]]);
  // 示意图
  const dg=VZ.draw(mod,m,wl,hw,g), ex=VZ.extra(mod,m,wl);
  $("am-diagram").setAttribute("viewBox","0 0 680 "+dg.h); $("am-diagram").innerHTML=dg.svg;
  $("am-diagram2-fig").hidden=!ex; if(ex){ $("am-diagram2").setAttribute("viewBox","0 0 680 "+ex.h); $("am-diagram2").innerHTML=ex.svg; }
  $("am-diagram-cap").innerHTML=t("arch.viz.cap."+mod.kind)+' '+t("arch.viz.credit",{href:ALAB[mod.kind]||ALAB.gqa});
  renderModRoof(mod, wl, cur, a, a0, edited, peaks);
  renderModCurves(mod, wl, edited);
  renderModRead(mod, wl, a, a0, fx, fx0, edited, ridge, pk);
  table($("am-ops"),[t("arch.ops.th.op"),"FLOPs",t("arch.ops.th.bytes"),t("arch.ops.th.I"),t("arch.ops.th.t"),t("arch.ops.th.bound")],
    a.ops.map(o=>[esc(o.name),fF(o.flops),fB(o.bytes),fI(o.I),fT(o.t),o.t>0?pillB(o.bound in {compute:1,memory:1,launch:1}?o.bound:"memory"):"—"]),[1,2,3,4]);
  $("am-ops-lede").textContent=t("arch.ops.lede."+(wl.phase==="train"?"train":"infer"));
  $("am-src").innerHTML=srcList(mod.kind, wl.phase);
}

/* 轨迹用的负载：prefill 每个取值只抽 4 个 chunk（注意力随前缀线性涨，按四分位中点取样的平均就是精确值） */
const sweepWl = wl => wl.phase==="prefill" ? Object.assign({}, wl, {maxChunks:4}) : wl;
const trajKey = wl => wl.phase==="decode" ? ["B", bList] : wl.phase==="prefill" ? ["n", ctxList.filter(x=>x<=262144)] : ["S", ctxList.filter(x=>x<=262144)];
/* 改了注意力 / 线性注意力的类型后，预设结构里没有同一个模块，不画对照 */
const sameKinds = () => baseModel().full.kind===S.model.full.kind && baseModel().lin.kind===S.model.lin.kind;

/* roofline：屋顶（η 之后）+ 本模型各模块的点 + 当前模块随 B（decode）/ prompt 长度（prefill）/ 序列长度（训练）的轨迹 */
function renderModRoof(mod, wl, cur, a, a0, edited, peaks){
  const {m,hw,g}=ctx(), [key,xs]=trajKey(wl), w=sweepWl(wl);
  const tr = model => sweep(model,w,key,xs).map(([x,r])=>[r.mods[mod.id].I, r.mods[mod.id].perf, x]).filter(p=>isFinite(p[0])&&p[0]>0&&p[1]>0);
  const traj=tr(S.model), traj0=edited&&sameKinds()?tr(baseModel()):[];
  const others=Arch.present(m).filter(p=>p.id!==mod.id);
  const allI=traj.concat(traj0).map(p=>p[0]).concat([a.I]).filter(v=>isFinite(v)&&v>0), allP=traj.concat(traj0).map(p=>p[1]).concat([a.perf]).filter(v=>v>0);
  others.forEach(p=>{ const x=cur.mods[p.id]; if(x&&x.I>0&&isFinite(x.I)){ allI.push(x.I); allP.push(x.perf); } });
  const x0=Math.pow(10,Math.floor(Math.log10(Math.max(0.05,Math.min(...allI)/2)))), x1=Math.pow(10,Math.ceil(Math.log10(Math.max(...allI,ridgeOf(hw,g,"bf16"))*3)));
  const y1=Math.max(...peaks.map(p=>p.v))*1.6, y0=Math.pow(10,Math.floor(Math.log10(Math.max(1e9,Math.min(...allP)/3))));
  const series=roofSeries(hw,g,peaks,x0,x1);
  series.push({pts:traj.map(p=>[p[0],p[1]]),color:css(MODC[mod.id]),w:1.6});
  if(traj0.length) series.push({pts:traj0.map(p=>[p[0],p[1]]),color:css("muted"),w:1.2,dash:"5 4"});
  const points=[], lab=x=>key+"="+fxPow(x), marks=key==="B"?[8,64,512]:[8192,131072];
  traj.forEach((p,i)=>{ if(i===0||i===traj.length-1||marks.includes(p[2])) points.push({x:p[0],y:p[1],color:css(MODC[mod.id]),r:2.6,label:lab(p[2])}); });
  others.forEach(p=>{ const x=cur.mods[p.id]; if(x) points.push({x:x.I,y:x.perf,color:css(MODC[p.id]),r:4.5,shape:modShape(p.id),label:modName(p)}); });
  if(edited&&sameKinds()) points.push({x:a0.I,y:a0.perf,color:css("muted"),r:5,label:t("arch.roof.preset")});
  points.push({x:a.I,y:a.perf,color:css(MODC[mod.id]),r:7,shape:modShape(mod.id),label:modName(mod)+" · "+t("arch.roof.now")});
  const vl=peaks.map((p,i)=>{ const rx=ridgeOf(hw,g,p.k); return {x:rx,color:css("muted"),label:t("arch.roof.ridge",{p:p.k.toUpperCase(),v:fI(rx)}),dy:i*13}; });
  plot($("am-roof"),{x0,x1,y0,y1,logx:true,logy:true,fx:fxPow,fy:v=>fyFlops(v)+"FLOP/s",xl:t("arch.roof.x"),yl:t("arch.roof.y"),series,points,vlines:vl,h:380});
  $("am-roof-lede").innerHTML=t("arch.roof.lede."+key,{c:css(MODC[mod.id])})+(edited&&sameKinds()?" "+t("arch.roof.ledeEdited"):"");
  legendInto("am-roof-legend",[{name:t("arch.roof.legRoof",{hw:hw.name}),col:"ink",line:true},{name:t("arch.roof.legTraj",{mod:modName(mod)}),col:MODC[mod.id],line:true}].concat(traj0.length?[{name:t("arch.roof.legPreset"),col:"muted",line:true}]:[]));
}

/* 两张曲线：decode 随上下文 / 随 B，prefill 随 prompt 长度 / chunk 大小，训练随序列长度。同一模型里同类的其他模块画成点线 */
function renderModCurves(mod, wl, edited){
  const {m,hw}=ctx(), part=Arch.isMlp(mod.id)?"mlp":"attn", sibs=Arch.present(m).filter(p=>p.id!==mod.id && (Arch.isMlp(p.id)?"mlp":"attn")===part);
  const pts=(model, key, xs, id, fy)=>sweep(model,sweepWl(wl),key,xs).map(([x,r])=>[x, r.mods[id] ? fy(r.mods[id],r) : NaN]);
  const tok=(x,r)=>x.t/r.tokens, tm=x=>x.t, perf=x=>x.perf, ctxS=ctxList.filter(x=>x<=262144);
  const spec = wl.phase==="decode"
    ? (part==="attn" ? [["n",ctxList,tm,"c1n"],["B",bList,tok,"c2b"]] : [["B",bList,tok,"c1b"],["B",bList,perf,"c2perf"]])
    : wl.phase==="prefill"
    ? (part==="attn" ? [["n",ctxS,tok,"p1n"],["C",cList,tm,"p2c"]] : [["C",cList,tok,"p1c"],["C",cList,perf,"p2perf"]])
    : [["S",ctxS,tok,"t1s"],["S",ctxS,perf,"t2perf"]];
  spec.forEach(([key,xs,fy,id],i)=>{
    const series=[{name:modName(mod)+(edited?" · "+t("arch.curve.now"):""),short:modName(mod),col:MODC[mod.id],pts:pts(S.model,key,xs,mod.id,fy)}];
    if(edited&&sameKinds()) series.push({name:modName(mod)+" · "+t("arch.curve.preset"),short:t("arch.curve.preset"),col:"muted",dash:"5 4",pts:pts(baseModel(),key,xs,mod.id,fy)});
    sibs.forEach(p=>series.push({name:modName(p),short:modName(p),col:MODC[p.id],dash:"2 3",pts:pts(S.model,key,xs,p.id,fy)}));
    const isPerf=id.endsWith("perf"), at=key==="B"?wl.B:key==="n"?wl.n:key==="C"?Arch.chunkOf(wl,hw):wl.S;
    lineChart("am-c"+(i+1),{series,xLog:true,yLog:!isPerf,xFmt:fxPow,yFmt:isPerf?(v=>fN(v/1e12)+"T"):fT,tipFmt:isPerf?fTF:fT,
      xLab:t("arch.curve.x."+key),yLab:t("arch.curve.y."+id),vlines:[{x:at,label:t("arch.curve.cur")}]});
    $("am-c"+(i+1)+"-cap").textContent=t("arch.curve.cap."+id+"."+part);
  });
}

/* 读数：位置 + 机制专属的几句话 + 和预设比 */
function renderModRead(mod, wl, a, a0, fx, fx0, edited, ridge, pk){
  const {m,hw,g}=ctx(), f=m.full, l=m.lin, e=m.moe, R=[];
  R.push(t("arch.read.where",{t:V(fT(a.t)),b:t("arch.bound."+a.bound),i:V(fI(a.I)),r:fI(ridge),p:pk.toUpperCase(),side:t(a.I<ridge?"arch.read.left":"arch.read.right")}));
  if(a.byBound.launch>0.2*a.t) R.push(t("arch.read.launch",{p:pct(a.byBound.launch,a.t),us:g.kernelUs}));
  const B=wl.B, n=wl.n;
  if(wl.phase==="decode"){
    if(mod.kind==="gqa"||mod.kind==="swa"){ const keys=mod.id==="swa"?Math.min(n,f.W):n, kv=fx.kvTok, rd=B*keys*kv;
      R.push(t("arch.read.gqa",{kv:V(fB(kv)),hkv:f.Hkv,dh:f.dh,rd:V(fB(rd)),share:pct(rd,a.bytes),g:fI(f.Hq/f.Hkv)}));
      if(mod.id==="swa") R.push(t("arch.read.swa",{w:fInt(f.W),n:fInt(n),glob:fB(B*n*kv)})); }
    else if(mod.kind==="mla"){ const lat=f.kvLora+f.rope; R.push(t("arch.read.mla",{lat,kv:V(fB(fx.kvTok)),h:f.H,i:fI(2*f.H*(lat+f.kvLora)/(lat*dtB(wl.kvDt)))})); }
    else if(mod.kind==="dsa"){ const k=Math.min(n,f.topk), pools=Math.ceil(n/(f.kpool||1));
      R.push(t(n>f.topk?"arch.read.dsa":"arch.read.dsaShort",{n:fInt(n),k:fInt(f.topk),pools:fInt(pools),att:fB(B*k*(f.kvLora+f.rope)*dtB(wl.kvDt)),idx:fB(B*pools*(f.idxD+4))})); }
    else if(mod.kind==="gdn"||mod.kind==="kda"){ const sb=wl.stateDt==="bf16"?2:4, st=l.Hv*l.dk*l.dv*sb, kvTok=f.n>0?derive(m).kvTokElems*dtB(wl.kvDt):0;
      R.push(t("arch.read.lin",{st:V(fB(st)),dt:wl.stateDt.toUpperCase(),rw:fB(2*st*B),i:fI(8/(2*sb)),n:kvTok>0?fN(st/kvTok):"—"})); }
    else if(mod.kind==="moe"){ const cov=Infer.cover(e.k,e.E,B), hit=e.E*cov, rows=B*e.k/Math.max(1,hit), need=ridgeOf(hw,g,wl.eDt)*dtB(wl.eDt)/2;
      R.push(t("arch.read.moe",{b:fInt(B),hit:fInt(hit),e:e.E,pct:pc(cov),rows:fI(rows),w:V(fB(hit*fx.expert)),need:fInt(need),bneed:fInt(need*e.E/e.k)})); }
    else if(mod.kind==="dense"){ R.push(t("arch.read.dense",{w:V(fB(fx.weights)),i:fI(a.I),bs:fInt(ridgeOf(hw,g,wl.wDt)*dtB(wl.wDt)/2)})); }
  } else if(wl.phase==="prefill") R.push(t("arch.read.prefill."+mod.kind,{c:fInt(Math.min(n,Arch.chunkOf(wl,hw))),n:fInt(n),k:fInt(f.topk||0)}));
  else R.push(t("arch.read.train."+mod.kind,{s:fInt(wl.S)}));
  if(edited&&sameKinds()){ const dkv = fx.kvTok!=null&&fx0.kvTok!=null ? t("arch.read.dKv",{a:fB(fx0.kvTok),b:fB(fx.kvTok)}) : fx.state!=null&&fx0.state!=null ? t("arch.read.dState",{a:fB(fx0.state),b:fB(fx.state)}) : fx.weights!=null&&fx0.weights!=null ? t("arch.read.dW",{a:fB(fx0.weights),b:fB(fx.weights)}) : "";
    R.push(t("arch.read.vsPreset",{a:fT(a0.t),b:V(fT(a.t)),x:fX(a0.t/Math.max(a.t,1e-30)),i0:fI(a0.I),i1:fI(a.I),d:dkv})); }
  $("am-read").innerHTML=ul(R);
}

/* 来源：每种机制的 kernel 假设对应的开源实现（引擎代码 / 文档）；示意图借鉴的公开讲解 */
const ALAB = { gqa:"https://divagr.com/attention-lab/gqa", swa:"https://divagr.com/attention-lab/window", mla:"https://divagr.com/attention-lab/mla", dsa:"https://divagr.com/attention-lab/dsa",
  gdn:"https://divagr.com/attention-lab/delta", kda:"https://divagr.com/attention-lab/kda", moe:"https://divagr.com/attention-lab/", dense:"https://divagr.com/attention-lab/" };
const GH = "https://github.com/";
const SRC = {
  gqa:[["fa",GH+"Dao-AILab/flash-attention"],["sglFa",GH+"sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashattention_backend.py"],["vllmFa",GH+"vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py"],["trtAttn",GH+"NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/attention_backend"]],
  swa:[["fa",GH+"Dao-AILab/flash-attention"],["sglFa",GH+"sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashattention_backend.py"],["vllmFa",GH+"vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py"]],
  mla:[["flashmla",GH+"deepseek-ai/FlashMLA"],["sglMla",GH+"sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v2.py"],["vllmMla",GH+"vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py"]],
  dsa:[["ds32",GH+"deepseek-ai/DeepSeek-V3.2-Exp"],["deepgemm",GH+"deepseek-ai/DeepGEMM"],["flashmla",GH+"deepseek-ai/FlashMLA"],["sglDsa",GH+"sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa_backend.py"],["vllmDsa",GH+"vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/flashmla_sparse.py"],["vllmIdx",GH+"vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/indexer.py"]],
  gdn:[["fla",GH+"fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule"],["sglLin",GH+"sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"],["vllmGdn",GH+"vllm-project/vllm/blob/main/vllm/v1/attention/backends/gdn_attn.py"]],
  kda:[["flaKda",GH+"fla-org/flash-linear-attention/tree/main/fla/ops/kda"],["sglLin",GH+"sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"]],
  moe:[["sglMoe",GH+"sgl-project/sglang/tree/main/python/sglang/srt/layers/moe/fused_moe_triton"],["vllmMoe",GH+"vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe"],["mcoreMoe",GH+"NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/experts.py"]],
  dense:[["mcoreMlp",GH+"NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/mlp.py"],["vllmLinear",GH+"vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py"]] };
const SRC_PHASE = { decode:[], prefill:[["sglChunk",GH+"sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/memory_hook.py"],["vllmChunk",GH+"vllm-project/vllm/blob/main/vllm/engine/arg_utils.py"]],
  train:[["te",GH+"NVIDIA/TransformerEngine"],["mcore",GH+"NVIDIA/Megatron-LM"]] };
function srcList(kind, phase){
  const a=(SRC[kind]||[]).concat(SRC_PHASE[phase]||[]);
  return '<p class="lede">'+t("arch.src.assume."+kind)+'</p><ul class="tight src">'+a.map(([k,u])=>'<li>'+t("arch.src.item",{u,name:esc(t("arch.src.name."+k)),what:t("arch.src.what."+k)})+'</li>').join("")+'</ul>';
}

/* ---------- 跨模型对比 ---------- */
function renderArchCompare(){
  SWEEP=new WeakMap(); archRack();
  const {hw,g}=ctx(), wl=archWl(), w=Object.assign({}, wl, wl.phase==="prefill"?{maxChunks:3}:{}), keys=Object.keys(MODELS), peaks=Arch.peaks(hw,wl);
  const res=keys.map(k=>{ const m=k===S.modelKey?S.model:MODELS[k]; return {k, m, r:measure(m,wl), mods:Arch.present(m)}; });
  // roofline：所有模型的所有模块
  const points=[], allI=[], allP=[];
  res.forEach((x,i)=>x.mods.forEach(p=>{ const a=x.r.mods[p.id]; if(!a||!(a.I>0)||!isFinite(a.I)) return; allI.push(a.I); allP.push(a.perf);
    points.push({x:a.I,y:a.perf,color:css(MODEL_COL[i]),r:x.k===S.modelKey?6:4.5,shape:modShape(p.id),label:shortName(x.m)+" "+kindName(p.kind)}); }));
  const x0=Math.pow(10,Math.floor(Math.log10(Math.max(0.05,Math.min(...allI)/2)))), x1=Math.pow(10,Math.ceil(Math.log10(Math.max(...allI,ridgeOf(hw,g,"bf16"))*3)));
  const y0=Math.pow(10,Math.floor(Math.log10(Math.max(1e9,Math.min(...allP)/3)))), y1=Math.max(...peaks.map(p=>p.v))*1.6;
  plot($("ac-roof"),{x0,x1,y0,y1,logx:true,logy:true,fx:fxPow,fy:v=>fyFlops(v)+"FLOP/s",xl:t("arch.roof.x"),yl:t("arch.roof.y"),series:roofSeries(hw,g,peaks,x0,x1),points,h:400,
    vlines:peaks.map((p,i)=>{ const rx=ridgeOf(hw,g,p.k); return {x:rx,color:css("muted"),label:t("arch.roof.ridge",{p:p.k.toUpperCase(),v:fI(rx)}),dy:i*13}; })});
  legendInto("ac-roof-legend",res.map((x,i)=>({name:x.m.short,col:MODEL_COL[i]})).concat([{name:t("arch.cmp.shapes"),col:"muted",line:true}]));
  $("ac-lede").textContent=t("arch.cmp.lede",{wl:wlText(wl),hw:hw.name,w:wl.wDt.toUpperCase(),e:wl.eDt.toUpperCase(),kv:wl.kvDt.toUpperCase()});
  // 曲线：注意力每层时间随上下文；MLP 每 token 时间随 B（decode）/ chunk（prefill）
  const attnSeries=[], mlpSeries=[], aKey=wl.phase==="train"?"S":"n", aXs=wl.phase==="decode"?ctxList:ctxList.filter(v=>v<=262144);
  const mKey=wl.phase==="decode"?"B":"C", mXs=wl.phase==="decode"?bList:cList, inf=wl.phase!=="train";
  res.forEach((x,i)=>x.mods.forEach(p=>{
    if(!Arch.isMlp(p.id)) attnSeries.push({name:x.m.short+" · "+modName(p),short:shortName(x.m)+" "+kindName(p.kind),col:MODEL_COL[i],dash:p.id==="lin"?"2 3":p.id==="swa"?"6 3":null,
      pts:sweep(x.m,w,aKey,aXs).map(([v,r])=>[v,wl.phase==="decode"?r.mods[p.id].t:r.mods[p.id].t/r.tokens])});
    else if(inf) mlpSeries.push({name:x.m.short+" · "+modName(p),short:shortName(x.m)+" "+kindName(p.kind),col:MODEL_COL[i],dash:p.id==="dense"?"2 3":null,
      pts:sweep(x.m,w,mKey,mXs).map(([v,r])=>[v,r.mods[p.id].t/r.tokens])});
  }));
  lineChart("ac-attn",{series:attnSeries,xLog:true,yLog:true,xFmt:fxPow,yFmt:fT,xLab:t("arch.curve.x."+aKey),yLab:t("arch.cmp.attnY."+wl.phase),labelEnds:false,
    vlines:[{x:wl.phase==="train"?wl.S:wl.n,label:t("arch.curve.cur")}]});
  $("ac-mlp-fig").hidden=!inf;
  if(inf) lineChart("ac-mlp",{series:mlpSeries,xLog:true,yLog:true,xFmt:fxPow,yFmt:fT,xLab:t("arch.curve.x."+mKey),yLab:t("arch.cmp.mlpY"),labelEnds:false,
    vlines:[{x:wl.phase==="decode"?wl.B:Arch.chunkOf(wl,hw),label:t("arch.curve.cur")}]});
  legendInto("ac-attn-legend",res.map((x,i)=>({name:x.m.short,col:MODEL_COL[i],line:true})).concat([{name:t("arch.cmp.dashLin"),col:"muted",line:true}]));
  legendInto("ac-mlp-legend",res.map((x,i)=>({name:x.m.short,col:MODEL_COL[i],line:true})).concat([{name:t("arch.cmp.dashDense"),col:"muted",line:true}]));
  // 每个模型一层层加起来的时间按模块拆
  $("ac-split").innerHTML=res.map(x=>{ const items=x.mods.map(p=>({id:p.id,label:modName(p)+" × "+p.layers,v:x.r.mods[p.id].t*p.layers})), tot=items.reduce((s,q)=>s+q.v,0);
    return '<div class="ac-row"><div class="lab">'+esc(x.m.short)+' <span class="v">'+fT(tot)+'</span></div><div class="bar">'+items.map(q=>'<span style="width:'+(q.v/tot*100).toFixed(2)+'%;background:'+css(MODC[q.id])+'" title="'+esc(t("arch.cmp.splitTip",{l:q.label,t:fT(q.v),p:pct(q.v,tot)}))+'"></span>').join("")+'</div></div>'; }).join("");
  legendInto("ac-split-legend",Arch.ORDER.map(id=>({name:t("arch.cmp.part."+id),col:MODC[id]})));
  table($("ac-tbl"),[t("arch.cmp.th.model"),t("arch.cmp.th.mod"),t("arch.cmp.th.layers"),t("arch.cmp.th.params"),t("arch.cmp.th.mem"),t("arch.cmp.th.t"),t("arch.cmp.th.I"),t("arch.cmp.th.bound"),t("arch.cmp.th.share")],
    res.flatMap(x=>{ const fx=Arch.facts(x.m,wl), tot=x.mods.reduce((s,p)=>s+x.r.mods[p.id].t*p.layers,0);
      return x.mods.map(p=>{ const a=x.r.mods[p.id], fc=fx[p.id]||{};
        const mem=fc.kvTok!=null?t("arch.cmp.memKv",{v:fB(fc.kvTok)}):fc.state!=null?t("arch.cmp.memState",{v:fB(fc.state)}):fc.expert!=null?t("arch.cmp.memExp",{v:fB(fc.weights)}):fB(fc.weights||0);
        return {cls:x.k===S.modelKey?"hl":"",cells:[esc(x.m.short),'<a href="#arch/module/'+p.id+'" data-model="'+x.k+'">'+esc(modName(p))+'</a>',p.layers,fN(fc.params||0),esc(mem),fT(a.t),fI(a.I),pillB(a.bound),pct(a.t*p.layers,tot)]}; }); }),[2,3,5,6,8]);
  qa("#ac-tbl a[data-model]").forEach(el=>el.addEventListener("click",()=>{ const k=el.dataset.model; if(k!==S.modelKey){ $("g-model").value=k; $("g-model").dispatchEvent(new Event("change")); } }));
}

return { renderArchOverview, renderArchModule, renderArchCompare };
})();
const { renderArchOverview, renderArchModule, renderArchCompare } = ArchUI;
