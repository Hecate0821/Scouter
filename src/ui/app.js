/* ---------- 页面与渲染：hash = 页 / 子页 / 锚点 ---------- */
I18N.apply(document);
const NAV = { overview:[], arch:["overview","module","compare"], train:["step","par","quant","calib"], infer:["step","deploy","quant","spec","kv","calib"], rl:["mismatch","determinism","lowprec","qat"], cluster:["fabric","impact"], appendix:["data","quant","method"] };
const OLD = { models:"arch/overview", "appendix/models":"arch/overview", par:"train/par", "par/calib":"train/calib", quant:"infer/quant", public:"appendix/data", method:"appendix/method" };   // 旧的七个 tab
const VIEW = {
  "overview": renderOverview,
  "arch/overview": renderArchOverview, "arch/module": renderArchModule, "arch/compare": renderArchCompare,
  "train/step": renderTrain, "train/par": ()=>{ syncTrnRack(); renderTrnComm(); renderMesh(trnRun()); renderPar(); }, "train/quant": renderQuantTrain, "train/calib": renderCalib,
  "infer/step": renderInfer, "infer/deploy": ()=>{ syncInfRack(); renderInfComm(); renderInferTbo(); renderInferTO(); }, "infer/quant": renderQuantInfer, "infer/spec": ()=>{ syncInfRack(); renderSpec(); }, "infer/kv": ()=>{ syncInfRack(); renderKV(); }, "infer/calib": renderInfCalib,
  "rl/mismatch": renderMismatch, "rl/determinism": renderDeterminism, "rl/lowprec": renderLowprec, "rl/qat": renderQat,
  "cluster/fabric": ()=>{ syncInfRack(); renderClusterFabric(); }, "cluster/impact": ()=>{ syncInfRack(); renderClusterImpact(); },
  "appendix/data": ()=>{ renderKernels(); renderEff(); renderPublic(); renderReleases(); renderDB(); }, "appendix/quant": renderFormats, "appendix/method": ()=>{} };
const lastSub = {};
const viewKey = () => S.tab+(S.sub?"/"+S.sub:"");
function render(){
  try{ setWarn([]); VIEW[viewKey()](); }
  catch(e){ console.error(e); setWarn([t("common.calcError",{m:e.message})]); }
}
function route(h){
  let p=String(h||"").replace(/^#/,"").split("/").filter(Boolean);
  const two=p.slice(0,2).join("/"), k=OLD[two]?two:(p.length && !NAV[p[0]] && OLD[p[0]])?p[0]:null;     // 旧地址（包括搬走的子页，例如 appendix/models）
  if(k) p=OLD[k].split("/").concat(p.slice(k.split("/").length));
  else if(p.length && !NAV[p[0]]) p=[];
  const page=NAV[p[0]]?p[0]:"overview";
  if(!NAV[page].length) return { page, sub:"", anchor:p[1]||"" };
  let sub=p[1]||"", anchor=p[2]||"";
  if(!NAV[page].includes(sub)){      // 不是子页名：当作锚点，找它所在的子页（例如旧的 #infer/tradeoff）
    const el=sub && ($(page+"-"+sub)||$(sub)), pn=el && el.closest("[data-pane]");
    if(pn && pn.dataset.pane.startsWith(page+"/")){ anchor=sub; sub=pn.dataset.pane.split("/")[1]; }
    else { anchor=""; sub=lastSub[page]||NAV[page][0]; } }
  return { page, sub, anchor };
}
function showTab(h){
  const {page,sub,anchor}=route(h), key=page+(sub?"/"+sub:""), same=viewKey()===key;
  S.tab=page; S.sub=sub; S.anchor=anchor; if(sub) lastSub[page]=sub;
  const pick = key==="arch/module" && Arch.ORDER.includes(anchor);      // #arch/module/<模块>：锚点是选哪个模块，不是滚动位置
  pageHead(key);
  qa("[data-pane]").forEach(p=>p.hidden=p.dataset.pane!==key);
  qa("[data-rack]").forEach(p=>p.hidden=!p.dataset.rack.split(" ").includes(key));
  try{ localStorage.setItem("moe-roof-nav3",JSON.stringify({h:key,last:lastSub})); }catch(e){}
  if(key==="train/par" && PAR_DIMS.includes(anchor)) setParDim(anchor);   // 同页跳转不重新渲染，所以在这里切
  if(!(same && anchor) || pick) render();
  applySource();
  const tgt=anchor && !pick && ($(sub+"-"+anchor)||$(page+"-"+anchor)||$(anchor));
  if(tgt && tgt.scrollIntoView){ syncCtx(); tgt.scrollIntoView({block:"start"}); }   // 首次打开深链接时观察器还没回调
  else if(!same) (document.scrollingElement||document.documentElement).scrollTop=0;
}
window.addEventListener("hashchange",()=>showTab(location.hash.slice(1)));
// 地址没变时浏览器不发 hashchange：已经停在 #train/par/ep，再点同一个链接也要回到那里
document.addEventListener("click",e=>{ const a=e.target.closest('a[href^="#"]'); if(a && a.getAttribute("href")===location.hash && route(location.hash.slice(1)).anchor) showTab(location.hash.slice(1)); });
let first=location.hash.slice(1);
try{ const st=JSON.parse(localStorage.getItem("moe-roof-nav3")||"null"); if(st){ Object.assign(lastSub,st.last||{}); if(!first) first=st.h; } }catch(e){}
for(const k in lastSub) if(!NAV[k] || !NAV[k].includes(lastSub[k])) delete lastSub[k];     // 记住的子页可能已经搬走
/* 语言：只有一种语言时不显示 */
const LANGS=Object.keys(I18N.LOCALES);
if(LANGS.length>1){ fillOpts($("g-lang"),LANGS.map(k=>[k,I18N.LOCALES[k].name]),I18N.locale); $("g-lang-wrap").hidden=false;
  $("g-lang").addEventListener("change",e=>I18N.setLocale(e.target.value)); }
["q-ev-topic","q-ev-bw","q-ev-sort"].forEach(id=>$(id).addEventListener("change",render));
applyAll(); qShape(); buildEditor(); syncInputs(); loadUrl(); themeLabel(); showTab(first||"overview");
