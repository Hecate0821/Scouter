/* ---------- 和实测对照：数据点库里能复现的实测，按同样的设置喂给计算核心。后训练页是匿名实测（Train.run / Par），推理页是 cookbook（Infer.serve） ---------- */
/* 每个数据集的显示：名字、颜色、点的标签；怎么复现（筛点、实测字段、预测）在计算核心的 Calib.SETS 里。rel = 只看组内相对快慢 */
const BE_NAME = { local:t("calib.be.local"), nccl_a2a:"NCCL a2a", deepep_legacy:"DeepEP legacy", deepep_v2:"DeepEP v2" };
/* 每个点的标签 = what（排序按它）+ 可选的 tail（序列长度等） */
const CALIB = [
  { set:"cp_attention", col:"s1", name:t("calib.set.cpAttention.name"),
    note:t("calib.set.cpAttention.note"),
    what:p=>p.method==="single"?t("calib.label.single"):"CP"+p.cp+" "+Par.CP[p.method].name, tail:p=>fK(p.seq), },
  { set:"cp_full_model", col:"s2", name:t("calib.set.cpFullModel.name"),
    note:t("calib.set.cpFullModel.note"),
    what:p=>"CP"+p.cp+" "+Par.CP[p.method].name, tail:p=>fK(p.seq), },
  { set:"efsdp_full_model", col:"s3", name:t("calib.set.efsdpFullModel.name"), rel:true,
    note:t("calib.set.efsdpFullModel.note"),
    what:p=>BE_NAME[p.backend]+" EP"+p.ep, tail:p=>fK(p.seq), },
  { set:"efsdp_comm", col:"s4", name:t("calib.set.efsdpComm.name"),
    note:t("calib.set.efsdpComm.note"),
    what:p=>BE_NAME[p.backend]+" EP"+p.ep, tail:p=>t("calib.label.perGpu",{n:fK(p.seq/8)}), },
  { set:"h200_kernels", col:"s5", name:t("calib.set.h200Kernels.name"),
    note:t("calib.set.h200Kernels.note"),
    what:p=>p.op==="bf16_gemm"?"GEMM "+[p.m,p.n,p.k].map(fK).join("×"):t("calib.label.copy",{bytes:fB(p.rw_bytes)}), },
];
/* 推理：SGLang cookbook 的闭环压测，按每行的部署形状喂给 Infer.serve */
const cookWhat = p => t("calib.label.cook",{model:MODELS[p.model].short, n:p.N, hw:String(HW[p.hw].name), quant:p.quant.toUpperCase(),
  spec:p.spec==="mtp"?t("calib.label.cookMtp",{k:p.specK,acc:p.acc_len}):"", c:p.conc});
const cookTail = p => t("calib.label.cookLen",{isl:p.isl, osl:p.osl});
const CALIB_INF = [
  { set:"cookbook_tpot", col:"s1", name:t("calib.set.cookbookTpot.name"), note:t("calib.set.cookbookTpot.note"), what:cookWhat, tail:cookTail },
  { set:"cookbook_ttft", col:"s3", name:t("calib.set.cookbookTtft.name"), note:t("calib.set.cookbookTtft.note"), what:cookWhat, tail:cookTail },
];
/* 两个对照页：元素 id 前缀、数据集列表、当前选中的数据集 */
const CV = { train:{ pre:"p-calib", list:CALIB, sel:"cp_attention", cache:null }, infer:{ pre:"i-calib", list:CALIB_INF, sel:"cookbook_tpot", cache:null } };
function calibReplay(V, g){
  const key=JSON.stringify(g); if(V.cache && V.cache.key===key) return V.cache.out;
  const out=V.list.map(c=>{ const K=Calib.SETS[c.set], sid=K.set||c.set, pts=K.where?DB.where(sid,K.where):DB.set(sid);
    return { c, meta:DB.meta(sid), rows:pts.map(p=>{ let pred=null; try{ pred=K.predict(p,g); }catch(e){}
    const meas=K.meas(p), what=c.what(p); return { p, what, label:c.tail?t("calib.label.pair",{what,tail:c.tail(p)}):what, meas, pred, ratio:isFinite(pred)&&pred>0&&meas>0 ? pred/meas : null }; }).filter(x=>x.ratio!=null)
    .sort((a,b)=>a.what.localeCompare(b.what)||a.meas-b.meas) }; });
  V.cache={key,out}; return out;
}
function quantile(a,q){ const s=[...a].sort((x,y)=>x-y), i=(s.length-1)*q, lo=Math.floor(i); return s.length ? s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo) : NaN; }

const renderCalib = () => renderCalibView(CV.train), renderInfCalib = () => { syncInfRack(); renderCalibView(CV.infer); renderTrace(); };
function renderCalibView(V){
  const {g}=ctx(), res=calibReplay(V, g), el=k=>$(V.pre+"-"+k);
  el("sum").innerHTML='<thead><tr><th>'+t("calib.sum.head.set")+'</th><th>'+t("calib.sum.head.n")+'</th><th>'+t("calib.sum.head.median")+'</th><th>'+t("calib.sum.head.mid80")+'</th><th>'+t("calib.sum.head.within25")+'</th><th>'+t("calib.sum.head.how")+'</th></tr></thead><tbody>'+
    res.map(({c,meta,rows})=>{ const rs=rows.map(x=>x.ratio), md=quantile(rs,0.5), inb=rs.filter(x=>x>=0.8&&x<=1.25).length;
      return '<tr data-calib="'+c.set+'"'+(c.set===V.sel?' class="hl"':'')+' style="cursor:pointer"><td><span class="dot" style="background:'+css(c.col)+'"></span>'+esc(c.name)+'<br><span class="tag">'+esc(c.set)+'</span></td><td class="num">'+rows.length+'</td><td class="num '+(c.rel?"":md>1.35||md<0.74?"bad":"ok")+'">'+md.toFixed(2)+'</td><td class="num">'+quantile(rs,0.1).toFixed(2)+' – '+quantile(rs,0.9).toFixed(2)+'</td><td class="num">'+(c.rel?"—":pc(inb/rs.length))+'</td><td>'+(c.rel?t("calib.sum.noteRel",{note:esc(c.note)}):esc(c.note))+'</td></tr>'; }).join("")+'</tbody>';
  calibScatter(V, res);
  legendInto(V.pre+"-legend",res.map(({c})=>({name:c.name,col:c.col})).concat([{name:t("calib.legend.parity"),col:"muted",line:true}]));
  const sel=res.find(x=>x.c.set===V.sel)||res[0], meta=sel.meta, pub=meta.origin==="public";
  el("cond").textContent=pub ? t("calib.condPublic",{title:meta.title,ref:String(meta.ref||"").slice(0,10),date:meta.as_of||"",cond:meta.conditions})
    : t("calib.cond",{gpus:meta.gpus,hw:meta.hw,cond:meta.conditions});
  el("pts").innerHTML='<thead><tr><th>'+t("calib.pts.head.setting")+'</th><th>'+t("calib.term.meas")+'</th><th>'+t("calib.term.pred")+'</th><th>'+t("calib.term.ratio")+'</th>'+(pub?'<th>'+t("calib.pts.head.src")+'</th>':'')+'</tr></thead><tbody>'+
    sel.rows.map(x=>'<tr><td>'+esc(x.label)+'</td><td class="num">'+fT(x.meas)+'</td><td class="num">'+fT(x.pred)+'</td><td class="num '+(sel.c.rel?"":x.ratio>1.35||x.ratio<0.74?"bad":"ok")+'">'+x.ratio.toFixed(2)+'</td>'+
      (pub?'<td>'+(x.p.url?'<a href="'+esc(x.p.url)+'" target="_blank" rel="noopener">'+esc(x.p.src)+'</a>':esc(x.p.src||""))+(x.p.sglang?' <span class="tag">'+esc(String(x.p.sglang).slice(0,12))+'</span>':'')+'</td>':'')+'</tr>').join("")+'</tbody>';
}
function calibScatter(V, res){
  const svg=$(V.pre+"-plot"), W=680, H=380, m={l:64,r:18,t:16,b:44};
  const all=res.flatMap(x=>x.rows), vs=all.flatMap(x=>[x.meas,x.pred]);
  const lo=Math.min(...vs)/1.5, hi=Math.max(...vs)*1.5, X=axis(lo,hi,m.l,W-m.r,true), Y=axis(lo,hi,H-m.b,m.t,true), tk=logTicks(lo,hi,10);
  let s="";
  for(const v of tk) s+='<line class="gr" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+esc(fT(v))+'</text><text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+esc(fT(v))+'</text>';
  s+='<line class="ax" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+(H-m.b)+'" y2="'+(H-m.b)+'"/>';
  s+='<polygon points="'+[X(lo)+","+Y(lo*1.25),X(hi/1.25)+","+Y(hi),X(hi)+","+Y(hi),X(hi)+","+Y(hi*0.8),X(lo/0.8)+","+Y(lo),X(lo)+","+Y(lo)].join(" ")+'" style="fill:var(--soft)"/>';
  s+='<line x1="'+X(lo)+'" y1="'+Y(lo)+'" x2="'+X(hi)+'" y2="'+Y(hi)+'" style="stroke:var(--muted)" stroke-dasharray="4 4"/>';
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+t("calib.term.meas")+'</text><text x="12" y="'+(m.t-4)+'">'+t("calib.term.pred")+'</text>';
  for(const {c,rows} of res) for(const x of rows){ const on=c.set===V.sel;
    s+='<circle cx="'+X(x.meas).toFixed(1)+'" cy="'+Y(x.pred).toFixed(1)+'" r="'+(on?4.5:3.5)+'" style="fill:'+css(c.col)+';stroke:var(--panel);opacity:'+(on?1:0.55)+'" stroke-width="1.5" '+
      tipAttr(t("calib.tip.title",{name:c.name,label:x.label}),[{v:fT(x.meas),l:t("calib.term.meas")},{v:fT(x.pred),l:t("calib.term.pred"),c:c.col},{v:x.ratio.toFixed(2),l:t("calib.term.ratio")}])+'/>'; }
  svg.setAttribute("viewBox","0 0 "+W+" "+H); svg.innerHTML=s; bindTips(svg);
}
for(const V of Object.values(CV)) $(V.pre+"-sum").addEventListener("click",e=>{ const tr=e.target.closest("[data-calib]"); if(!tr) return; V.sel=tr.dataset.calib; renderCalibView(V); });

/* 推理页：导入 torch profiler / Chrome trace，按右侧的推理设置跑一步，逐类别对照（Scouter.trace） */
let traceData = null;
function renderTrace(){
  const msg=$("i-trace-msg"), tb=$("i-trace-tbl");
  if(!traceData){ tb.innerHTML=""; msg.textContent=t("calib.trace.none"); return; }
  let r; try{ r=Scouter.trace(Object.assign({}, S.inf, { model:S.model, hw:S.hw, topo:S.topo, g:S.glob, trace:traceData.json, steps:+$("i-trace-steps").value||1 })); }
  catch(e){ tb.innerHTML=""; msg.textContent=t("calib.trace.err",{m:e.message}); return; }
  msg.textContent=t("calib.trace.sum",{file:traceData.name, n:r.kernels, steps:r.steps, meas:fT(r.total.meas), pred:fT(r.total.pred)})+
    (r.ranks>1 ? " "+t("calib.trace.ranks",{n:r.ranks}) : "")+(r.overlap>0 ? " "+t("calib.trace.overlap",{t:fT(r.overlap)}) : "")+
    (r.gpu ? " "+t("calib.trace.gpu",{busy:fT(r.gpu.busy), wall:fT(r.gpu.wall), idle:fT(r.gpu.idle), step:fT(r.gpu.step)}) : "")+
    (r.comm ? " "+t("calib.trace.comm",{meas:fT(r.comm.meas), floor:fT(r.comm.floor), wait:fT(r.comm.wait), pred:fT(r.comm.pred), ratio:r.comm.ratio==null?"—":r.comm.ratio.toFixed(2)}) : "")+
    (r.unmatched.length ? " "+t("calib.trace.unmatched",{names:r.unmatched.slice(0,3).map(k=>k.name.slice(0,48)).join(", ")}) : "");
  const rt=x=>x==null?"—":x.toFixed(2);
  tb.innerHTML='<thead><tr><th>'+t("calib.trace.head.cat")+'</th><th>'+t("calib.term.meas")+'</th><th>'+t("calib.term.pred")+'</th><th>'+t("calib.term.ratio")+'</th><th>'+t("calib.trace.head.top")+'</th></tr></thead><tbody>'+
    r.rows.map(x=>'<tr><td>'+t("calib.trace.cat."+x.cat)+'</td><td class="num">'+(x.meas?fT(x.meas):"—")+'</td><td class="num">'+(x.pred?fT(x.pred):"—")+'</td><td class="num '+(x.ratio==null?"":x.ratio>1.35||x.ratio<0.74?"bad":"ok")+'">'+rt(x.ratio)+'</td><td>'+
      x.top.slice(0,3).map(k=>'<span class="tag">'+esc(k.name.slice(0,56))+'</span> '+fT(k.s)).join("<br>")+'</td></tr>').join("")+
    '<tr><td><b>'+t("calib.trace.total")+'</b></td><td class="num"><b>'+fT(r.total.meas)+'</b></td><td class="num"><b>'+fT(r.total.pred)+'</b></td><td class="num">'+rt(r.total.ratio)+'</td><td></td></tr></tbody>';
}
$("i-trace-file").addEventListener("change",e=>{ const f=e.target.files&&e.target.files[0]; if(!f) return;
  /* .gz 用浏览器自带的 DecompressionStream 解开 */
  (/\.gz$/i.test(f.name) ? new Response(f.stream().pipeThrough(new DecompressionStream("gzip"))).text() : f.text()).then(txt=>{ try{ traceData={ name:f.name, json:JSON.parse(txt) }; } catch(err){ traceData=null; $("i-trace-msg").textContent=t("calib.trace.err",{m:err.message}); return; } renderTrace(); }); });
$("i-trace-steps").addEventListener("input",renderTrace);
