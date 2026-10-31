/* ---------- 和实测对照：数据点库里能复现的匿名实测，按同样的设置喂给 Train.run / Par ---------- */
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
let calibCache = null, calibSel = "cp_attention";
function calibReplay(g){
  const key=JSON.stringify(g); if(calibCache && calibCache.key===key) return calibCache.out;
  const out=CALIB.map(c=>{ const K=Calib.SETS[c.set], pts=K.where?DB.where(c.set,K.where):DB.set(c.set);
    return { c, meta:DB.meta(c.set), rows:pts.map(p=>{ let pred=null; try{ pred=K.predict(p,g); }catch(e){}
    const meas=K.meas(p), what=c.what(p); return { p, what, label:c.tail?t("calib.label.pair",{what,tail:c.tail(p)}):what, meas, pred, ratio:isFinite(pred)&&pred>0&&meas>0 ? pred/meas : null }; }).filter(x=>x.ratio!=null)
    .sort((a,b)=>a.what.localeCompare(b.what)||a.meas-b.meas) }; });
  calibCache={key,out}; return out;
}
function quantile(a,q){ const s=[...a].sort((x,y)=>x-y), i=(s.length-1)*q, lo=Math.floor(i); return s.length ? s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo) : NaN; }

function renderCalib(){
  const {g}=ctx(), res=calibReplay(g);
  $("p-calib-sum").innerHTML='<thead><tr><th>'+t("calib.sum.head.set")+'</th><th>'+t("calib.sum.head.n")+'</th><th>'+t("calib.sum.head.median")+'</th><th>'+t("calib.sum.head.mid80")+'</th><th>'+t("calib.sum.head.within25")+'</th><th>'+t("calib.sum.head.how")+'</th></tr></thead><tbody>'+
    res.map(({c,meta,rows})=>{ const rs=rows.map(x=>x.ratio), md=quantile(rs,0.5), inb=rs.filter(x=>x>=0.8&&x<=1.25).length;
      return '<tr data-calib="'+c.set+'"'+(c.set===calibSel?' class="hl"':'')+' style="cursor:pointer"><td><span class="dot" style="background:'+css(c.col)+'"></span>'+esc(c.name)+'<br><span class="tag">'+esc(c.set)+'</span></td><td class="num">'+rows.length+'</td><td class="num '+(c.rel?"":md>1.35||md<0.74?"bad":"ok")+'">'+md.toFixed(2)+'</td><td class="num">'+quantile(rs,0.1).toFixed(2)+' – '+quantile(rs,0.9).toFixed(2)+'</td><td class="num">'+(c.rel?"—":pc(inb/rs.length))+'</td><td>'+(c.rel?t("calib.sum.noteRel",{note:esc(c.note)}):esc(c.note))+'</td></tr>'; }).join("")+'</tbody>';
  calibScatter(res);
  legendInto("p-calib-legend",res.map(({c})=>({name:c.name,col:c.col})).concat([{name:t("calib.legend.parity"),col:"muted",line:true}]));
  const sel=res.find(x=>x.c.set===calibSel)||res[0], meta=sel.meta;
  $("p-calib-cond").textContent=t("calib.cond",{gpus:meta.gpus,hw:meta.hw,cond:meta.conditions});
  $("p-calib-pts").innerHTML='<thead><tr><th>'+t("calib.pts.head.setting")+'</th><th>'+t("calib.term.meas")+'</th><th>'+t("calib.term.pred")+'</th><th>'+t("calib.term.ratio")+'</th></tr></thead><tbody>'+
    sel.rows.map(x=>'<tr><td>'+esc(x.label)+'</td><td class="num">'+fT(x.meas)+'</td><td class="num">'+fT(x.pred)+'</td><td class="num '+(sel.c.rel?"":x.ratio>1.35||x.ratio<0.74?"bad":"ok")+'">'+x.ratio.toFixed(2)+'</td></tr>').join("")+'</tbody>';
}
function calibScatter(res){
  const svg=$("p-calib-plot"), W=680, H=380, m={l:64,r:18,t:16,b:44};
  const all=res.flatMap(x=>x.rows), vs=all.flatMap(x=>[x.meas,x.pred]);
  const lo=Math.min(...vs)/1.5, hi=Math.max(...vs)*1.5, X=axis(lo,hi,m.l,W-m.r,true), Y=axis(lo,hi,H-m.b,m.t,true), tk=logTicks(lo,hi,10);
  let s="";
  for(const v of tk) s+='<line class="gr" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+esc(fT(v))+'</text><text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+esc(fT(v))+'</text>';
  s+='<line class="ax" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+(H-m.b)+'" y2="'+(H-m.b)+'"/>';
  s+='<polygon points="'+[X(lo)+","+Y(lo*1.25),X(hi/1.25)+","+Y(hi),X(hi)+","+Y(hi),X(hi)+","+Y(hi*0.8),X(lo/0.8)+","+Y(lo),X(lo)+","+Y(lo)].join(" ")+'" style="fill:var(--soft)"/>';
  s+='<line x1="'+X(lo)+'" y1="'+Y(lo)+'" x2="'+X(hi)+'" y2="'+Y(hi)+'" style="stroke:var(--muted)" stroke-dasharray="4 4"/>';
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+t("calib.term.meas")+'</text><text x="12" y="'+(m.t-4)+'">'+t("calib.term.pred")+'</text>';
  for(const {c,rows} of res) for(const x of rows){ const on=c.set===calibSel;
    s+='<circle cx="'+X(x.meas).toFixed(1)+'" cy="'+Y(x.pred).toFixed(1)+'" r="'+(on?4.5:3.5)+'" style="fill:'+css(c.col)+';stroke:var(--panel);opacity:'+(on?1:0.55)+'" stroke-width="1.5" '+
      tipAttr(t("calib.tip.title",{name:c.name,label:x.label}),[{v:fT(x.meas),l:t("calib.term.meas")},{v:fT(x.pred),l:t("calib.term.pred"),c:c.col},{v:x.ratio.toFixed(2),l:t("calib.term.ratio")}])+'/>'; }
  svg.setAttribute("viewBox","0 0 "+W+" "+H); svg.innerHTML=s; bindTips(svg);
}
$("p-calib-sum").addEventListener("click",e=>{ const tr=e.target.closest("[data-calib]"); if(!tr) return; calibSel=tr.dataset.calib; renderCalib(); });
