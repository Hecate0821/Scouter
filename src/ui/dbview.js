/* ---------- 附录 · 数据与发布：数据点库浏览 ---------- */
let dbSel = "cp_attention";
const DB_STAGE = { train:t("appendix.db.stage.train"), infer:t("appendix.db.stage.infer"), quant:t("appendix.db.stage.quant"), kernel:t("appendix.db.stage.kernel"), rl:t("appendix.db.stage.rl"), meta:t("appendix.db.stage.meta"), cluster:t("appendix.db.stage.cluster"), profile:t("appendix.db.stage.profile") };
const DB_HIDE = ["id","set","m","replay","chk","url","tags","dk","arch","nature","kind"];
const DB_CONF = { measured:t("appendix.db.conf.measured"), reported:t("appendix.db.conf.reported"), derived:t("appendix.db.conf.derived") };
const dbVal = v => v==null ? "—" : typeof v==="number" ? (Number.isInteger(v)?v.toLocaleString("en-US"):String(+v.toPrecision(4))) : typeof v==="boolean" ? (v?t("appendix.db.yes"):t("appendix.db.no")) : String(v);
function renderDB(){
  const ids=Object.keys(DB.sets);
  $("m-db-sets").innerHTML='<thead><tr><th>'+t("appendix.db.sets.head.set")+'</th><th>'+t("appendix.db.sets.head.origin")+'</th><th>'+t("appendix.db.sets.head.stage")+'</th><th>'+t("appendix.db.sets.head.hw")+'</th><th>'+t("appendix.db.sets.head.n")+'</th><th>'+t("appendix.db.sets.head.conf")+'</th><th>'+t("appendix.db.sets.head.asOf")+'</th></tr></thead><tbody>'+ids.map(id=>{ const s=DB.meta(id);
    return '<tr data-db="'+id+'"'+(id===dbSel?' class="hl"':'')+' style="cursor:pointer"><td>'+esc(s.title)+'<br><span class="tag">'+esc(id)+'</span></td><td>'+(s.origin==="anon"?t("appendix.db.origin.anon"):t("appendix.db.origin.public"))+'</td><td>'+(DB_STAGE[s.stage]||esc(s.stage))+'</td><td>'+(s.hw?s.gpus+" × "+esc(s.hw):t("appendix.db.hwVarious"))+'</td><td class="num">'+DB.count(id)+'</td><td>'+esc(DB_CONF[s.confidence]||"")+'</td><td>'+esc(s.as_of||"—")+'</td></tr>'; }).join("")+'</tbody>';
  const sel=$("m-db-set"); if(sel.options.length!==ids.length) sel.innerHTML=ids.map(id=>'<option value="'+id+'">'+esc(id)+'</option>').join(""); sel.value=dbSel;
  const s=DB.meta(dbSel), q=$("m-db-q").value.trim().toLowerCase(), all=DB.set(dbSel);
  const cmp=(a,b)=>{ for(const k of s.dims||[]){ const x=a[k], y=b[k]; if(x===y) continue; if(x==null) return 1; if(y==null) return -1; return typeof x==="number"&&typeof y==="number" ? x-y : String(x).localeCompare(String(y)); } return 0; };
  const pts=(q ? all.filter(p=>JSON.stringify(p).toLowerCase().includes(q)) : all.slice()).sort(s.origin==="anon"?cmp:(a,b)=>String(b.date||"").localeCompare(String(a.date||"")));
  const cols=s.origin==="anon" ? (s.dims||[]).concat(Object.keys(s.metrics||{})) : Object.keys(all[0]||{}).filter(k=>!DB_HIDE.includes(k));
  const head=k=>{ const mt=s.metrics&&s.metrics[k]; return mt ? esc(mt.label)+(mt.unit?'<br><span class="tag">'+esc(mt.unit)+'</span>':"") : esc(k); };
  const cell=(p,k)=>k==="confidence" ? esc(DB_CONF[p[k]]||p[k]) : k==="src"&&p.url ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.src)+'</a>' : esc(dbVal(p[k]));
  $("m-db-n").textContent=t("appendix.db.count", {n:pts.length, total:all.length});
  $("m-db-cond").textContent=s.origin==="anon" ? t("appendix.db.cond.anon", {gpus:s.gpus, hw:s.hw, cond:s.conditions}) : t("appendix.db.cond.public", {cond:s.conditions, asOf:s.as_of||"—", conf:DB_CONF[s.confidence]||""});
  $("m-db-pts").innerHTML='<thead><tr>'+cols.map(k=>'<th>'+head(k)+'</th>').join("")+'</tr></thead><tbody>'+
    pts.map(p=>'<tr>'+cols.map(k=>'<td'+(typeof p[k]==="number"?' class="num"':'')+'>'+cell(p,k)+'</td>').join("")+'</tr>').join("")+'</tbody>';
}
$("m-db-sets").addEventListener("click",e=>{ const tr=e.target.closest("[data-db]"); if(!tr) return; dbSel=tr.dataset.db; renderDB(); });
$("m-db-set").addEventListener("change",e=>{ dbSel=e.target.value; renderDB(); });
$("m-db-q").addEventListener("input",renderDB);
