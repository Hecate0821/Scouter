/* ---------- 附录 · 数据与发布：数据点库里 origin = "public" 的点 ---------- */
const EVID = DB.set("quant_evidence");
const pubSrc = p => (p.url ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.src)+'</a>' : esc(p.src))+'<br><span class="tag">'+esc(p.date)+'</span>';
const isPareto = p => (p.tags||[]).includes("pareto");
const PARETO_TREND = "inference_public:inferencex-pareto-trend";   // 跨模型的趋势（model 一栏是“多种”）：对象只写 config
function mokRatio(g){
  const t=DB.set("training_public").find(p=>p.replay&&p.replay.kind==="mok_ratio"); if(!t) return null;
  return { t, ratio:Calib.mokRatio(t.replay, g) };
}
/* 实测 kernel 和三条屋顶：理想（峰值）、建模（η）、实测（最好的 GEMM 算力 / 拷贝带宽） */
function renderKernels(){
  const k=Scouter.kernels(), hw=HW[k.hw], pk=peakOf(hw,"bf16"), x0=1, x1=1e4, xs=logspace(x0,x1,160);
  const line=(cap,bw)=>xs.map(I=>[I,Math.min(cap,bw*I)]);
  const gemm=k.points.filter(p=>p.flops>0);
  plot($("d-kern"),{h:360,x0,x1,y0:1e12,y1:pk*2,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:t("infer.roof.xl"),yl:t("infer.roof.yl"),
    series:[{pts:line(pk,hw.hbm*1e9),color:css("ink"),w:1.8},
      {pts:line(pk*k.modeled.peaks.bf16.attainableTFLOPs/k.ideal.peaks.bf16.peakTFLOPs,k.modeled.attainableGBs*1e9),color:css("muted"),w:1.4,dash:"6 4"},
      {pts:line(k.measured.TFLOPs*1e12,k.measured.GBs*1e9),color:css("ok"),w:1.6,dash:"2 3"}],
    points:gemm.map(p=>({x:p.I,y:p.TFLOPs*1e12,color:css("s1"),r:4.5}))});
  $("d-kern-legend").innerHTML='<span><i></i>'+t("appendix.data.kern.ideal")+'</span><span><i class="d"></i>'+t("appendix.data.kern.modeled")+'</span><span style="color:var(--ok)"><i class="d"></i>'+t("appendix.data.kern.measured")+'</span>'+
    '<span><span class="sw" style="background:'+css("s1")+'"></span>'+t("appendix.data.kern.gemm")+'</span>';
  $("d-kern-lede").innerHTML=t("appendix.data.kern.lede",{hw:hw.name,tf:V(fInt(k.measured.TFLOPs)),tfp:pct(k.measured.TFLOPs,k.ideal.peaks.bf16.peakTFLOPs),
    gb:V(fInt(k.measured.GBs)),gbp:pct(k.measured.GBs,k.ideal.hbmGBs),etaC:Common.G0.etaC,etaM:Common.G0.etaM});
}
function renderPublic(){
  const g=S.glob, hw=HW[S.q.hw], inf=DB.set("inference_public");
  table($("pub-infer"),[t("appendix.pub.infer.head.what"),t("appendix.pub.infer.head.config"),t("appendix.pub.infer.head.value"),t("appendix.pub.infer.head.src")],
    inf.filter(p=>!isPareto(p)).map(p=>[esc(p.model+" · "+p.engine+" · "+p.hw),esc(p.config),esc(p.value),pubSrc(p)]));
  table($("pub-pareto"),[t("appendix.pub.pareto.head.what"),t("appendix.pub.pareto.head.value"),t("appendix.pub.pareto.head.caveat"),t("appendix.pub.pareto.head.src")],
    inf.filter(isPareto).map(p=>[esc(p.id===PARETO_TREND?p.config:p.model+" · "+p.config),esc(p.value),esc(p.caveat||""),pubSrc(p)]));
  const mk=mokRatio(g);
  table($("pub-train"),[t("appendix.pub.train.head.what"),t("appendix.pub.train.head.shape"),t("appendix.pub.train.head.value"),t("appendix.pub.train.head.page"),t("appendix.pub.train.head.src")],
    DB.set("training_public").map(p=>[esc(p.engine+" · "+p.hw),esc(t("appendix.pub.train.shape", {model:p.model, config:p.config})),esc(p.value),
      mk&&mk.t===p ? t("appendix.pub.train.mok", {x:V(mk.ratio.toFixed(2)+"×")})+'<br><span class="tag">'+t("appendix.pub.train.mokNote")+'</span>' : "—",pubSrc(p)]));
  const topics=[...new Set(EVID.map(r=>r.topic))], ts=$("q-ev-topic");   // 第一项“全部”的 value 是空串
  if(ts.options.length!==topics.length+1){ ts.innerHTML='<option value="">'+esc(t("appendix.ev.topicAll"))+'</option>'+topics.map(x=>'<option>'+esc(x)+'</option>').join(""); ts.value=""; }
  const bwOnly=$("q-ev-bw").checked, srt=$("q-ev-sort").value, tp=ts.value;
  const rows=EVID.filter(r=>(!tp||r.topic===tp)&&(!bwOnly||r.arch==="bw")).sort((a,b)=>srt==="w"?b.w-a.w:(b.date.localeCompare(a.date)||b.w-a.w));
  const archN={bw:"Blackwell",bw120:"SM120",hopper:"Hopper",amd:"AMD",any:"—"};
  table($("q-evid"),[t("appendix.ev.head.date"),t("appendix.ev.head.src"),t("appendix.ev.head.hw"),t("appendix.ev.head.claim"),t("appendix.ev.head.value"),t("appendix.ev.head.w"),t("appendix.ev.head.page")],
    rows.map(r=>[r.date+(r.dk==="read"?'<br><span class="tag">'+t("appendix.ev.dateRead")+'</span>':r.dk==="measured"?'<br><span class="tag">'+t("appendix.ev.dateMeasured")+'</span>':""),'<a href="'+esc(r.url)+'" target="_blank" rel="noopener">'+esc(r.src)+'</a><br><span class="tag">'+esc(r.topic)+'</span>',
      esc(r.hw)+'<br><span class="tag">'+archN[r.arch]+'</span>',esc(r.claim),esc(r.value),
      '<span class="wbar"><span style="width:'+(r.w*100).toFixed(0)+'%"></span></span>'+r.w.toFixed(2),r.chk?qCheck(r.chk,hw):"—"]));
}
