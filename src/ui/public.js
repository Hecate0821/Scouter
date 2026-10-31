/* ---------- 附录 · 数据与发布：数据点库里 origin = "public" 的点 ---------- */
const EVID = DB.set("quant_evidence");
const pubSrc = p => (p.url ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.src)+'</a>' : esc(p.src))+'<br><span class="tag">'+esc(p.date)+'</span>';
const isPareto = p => (p.tags||[]).includes("pareto");
const PARETO_TREND = "inference_public:inferencex-pareto-trend";   // 跨模型的趋势（model 一栏是“多种”）：对象只写 config
function mokRatio(g){
  const t=DB.set("training_public").find(p=>p.replay&&p.replay.kind==="mok_ratio"); if(!t) return null;
  return { t, ratio:Calib.mokRatio(t.replay, g) };
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
