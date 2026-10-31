/* ---------- 附录 · 数据与发布：过去一年的发布与博客（数据点库的 releases 集）；RL 和投机两节的出处也从这里取链接 ---------- */
const REL = DB.set("releases");
const REL_N = { lmsys:"LMSYS", sglang:"SGLang", vllm:"vLLM", slime:"slime", miles:"Miles", areal:"AReaL" };
const REL_TAG = { rl:t("releases.tag.rl"), mismatch:t("releases.tag.mismatch"), quant:t("releases.tag.quant"), spec:t("releases.tag.spec"), parallel:t("releases.tag.parallel"), kernel:t("releases.tag.kernel"), model:t("releases.tag.model"), hw:t("releases.tag.hw"), api:t("releases.tag.api"), other:t("releases.tag.other") };
const REL_SUM = { lmsys:t("releases.sum.lmsys"), sglang:t("releases.sum.sglang"), vllm:t("releases.sum.vllm"), slime:t("releases.sum.slime"), miles:t("releases.sum.miles"), areal:t("releases.sum.areal") };
const relOf = (p, v) => REL.find(r => r.project===p && r.version===v);
const aOut = (url, text) => '<a href="'+esc(url)+'" target="_blank" rel="noopener">'+esc(text)+'</a>';
function relLink(p, v, label){
  const r=relOf(p,v); if(!r) throw new Error(t("releases.err.noRelease", {project:p, version:v}));
  return aOut(r.url, label || (p==="lmsys" ? "LMSYS "+r.date : REL_N[p]+" "+v));
}
/* 公开数字按出处取：量化证据集按 url 正则，确定性一节按 id；库里没有就报错 */
function evOf(re){ const p=DB.set("quant_evidence").find(q=>re.test(q.url)); if(!p) throw new Error(t("releases.err.noPoint", {what:re})); return p; }
function detOf(id){ const p=DB.set("determinism_public").find(q=>q.id==="determinism_public:"+id); if(!p) throw new Error(t("releases.err.noPoint", {what:id})); return p; }
const evLink = (re, label) => { const p=evOf(re); return aOut(p.url, label||p.src); };
const detLink = (id, label) => { const p=detOf(id); return aOut(p.url, label||p.src); };
/* 精选条目 → 按日期排的表。条目是 [项目, 版本, 说明]，{ev: 正则, t: 说明}（量化证据集里 url 匹配的那条），或 {det: id, t: 说明} */
function ossTable(el, items){
  const rows=items.map(x=>{
    if(x.ev||x.det){ const p=x.ev?evOf(x.ev):detOf(x.det); return { date:p.date, src:aOut(p.url,p.src), t:x.t }; }
    const r=relOf(x[0],x[1]); if(!r) throw new Error(t("releases.err.noRelease", {project:x[0], version:x[1]}));
    return { date:r.date, src:relLink(x[0],x[1],x[0]==="lmsys"?"LMSYS · "+r.title:null), t:x[2] }; });
  rows.sort((a,b)=>a.date.localeCompare(b.date));
  table(el,[t("releases.oss.head.date"),t("releases.oss.head.src"),t("releases.oss.head.what")],rows.map(r=>['<span class="v">'+r.date+'</span>',r.src,esc(r.t)]));
}
function renderReleases(){
  const by={}; REL.forEach(r=>(by[r.project]=by[r.project]||[]).push(r));
  table($("rel-sum"),[t("releases.sum.head.project"),'<span class="nw">'+t("releases.sum.head.n")+'</span>',t("releases.sum.head.time"),t("releases.sum.head.story")],Object.keys(REL_N).map(p=>{ const ds=(by[p]||[]).map(r=>r.date).sort();
    return ['<span class="nw">'+(p==="lmsys"?t("releases.sum.blog", {name:REL_N[p]}):REL_N[p])+'</span>',ds.length,ds.length?'<span class="nw">'+ds[0]+' →</span><br><span class="nw">'+ds[ds.length-1]+'</span>':"—",esc(REL_SUM[p])]; }),[1]);
  const ps=$("rel-proj"), ts=$("rel-tag");
  if(!ps.options.length){ ps.innerHTML='<option value="">'+t("releases.filter.all")+'</option>'+Object.keys(REL_N).map(p=>'<option value="'+p+'">'+REL_N[p]+'</option>').join("");
    ts.innerHTML='<option value="">'+t("releases.filter.all")+'</option>'+Object.keys(REL_TAG).map(k=>'<option value="'+k+'">'+REL_TAG[k]+'</option>').join(""); }
  const pv=ps.value, tv=ts.value, q=$("rel-q").value.trim().toLowerCase();
  const rows=REL.filter(r=>(!pv||r.project===pv)&&(!tv||r.tags.includes(tv))&&(!q||JSON.stringify(r).toLowerCase().includes(q)))
    .sort((a,b)=>b.date.localeCompare(a.date)||a.project.localeCompare(b.project));
  $("rel-n").textContent=t("releases.list.count", {n:rows.length, total:REL.length});
  table($("rel-list"),[t("releases.list.head.date"),t("releases.list.head.project"),t("releases.list.head.release"),t("releases.list.head.highlights"),t("releases.list.head.tags")],rows.map(r=>['<span class="v">'+r.date+'</span>',REL_N[r.project],aOut(r.url,r.title||r.version),
    '<ul class="tight rel-hl">'+r.highlights.map(h=>'<li>'+esc(h)+'</li>').join("")+'</ul>',r.tags.map(k=>'<span class="tag">'+esc(REL_TAG[k]||k)+'</span>').join(" ")]));
}
["rel-proj","rel-tag"].forEach(id=>$(id).addEventListener("change",renderReleases));
$("rel-q").addEventListener("input",renderReleases);
