/* ================= 架构页 · 机制示意图 =================
   每种模块一张图，画的是“决定 roofline 位置的那个量”：GQA 的分组、滑窗的窗口、MLA 的 latent、DSA 的两段、CSA 的压缩 + 滑窗两路和层间共享、线性注意力的固定状态、
   MoE 一批 token 碰到多少 expert、dense FFN 的权重复用。图随当前（可能改过的）结构参数变，旁边的滑块就是这些参数。
   思路借鉴 Attention Lab（divagr.com/attention-lab）的分步图解，这里把每一步落到字节和 FLOPs 上；代码是本项目自己写的。 */
const VZ = (function(){
const W = 680;
const r = (x,y,w,h,fill,extra) => '<rect x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+Math.max(0,w).toFixed(1)+'" height="'+Math.max(0,h).toFixed(1)+'" rx="2" style="fill:'+fill+'" '+(extra||"")+'/>';
const tx = (x,y,s,o) => { o=o||{}; return '<text x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'"'+(o.a?' text-anchor="'+o.a+'"':'')+(o.cls?' class="'+o.cls+'"':'')+(o.fill?' style="fill:'+o.fill+'"':'')+'>'+esc(s)+'</text>'; };
const ln = (x1,y1,x2,y2,stroke,w,dash) => '<line x1="'+x1.toFixed(1)+'" y1="'+y1.toFixed(1)+'" x2="'+x2.toFixed(1)+'" y2="'+y2.toFixed(1)+'" style="stroke:'+stroke+'" stroke-width="'+(w||1)+'"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
const PAL = ["var(--s1)","var(--s2)","var(--s3)","var(--s4)","var(--s5)","var(--s6)","var(--c-attn)","var(--c-lin)"];
/* 确定性的伪随机（同一组参数每次画出来一样） */
function rng(seed){ let s=seed>>>0||1; return () => { s^=s<<13; s>>>=0; s^=s>>17; s^=s<<5; s>>>=0; return s/4294967296; }; }
/* 横条对比：items [{label, v, col, hl}]，按最大值归一。左边留给标签，右边留给数值 */
function bars(x, y, w, items, fmt){
  const mx=Math.max(...items.map(i=>i.v),1e-30), LW=Math.min(250, w*0.42), VW=86; let s="";
  items.forEach((it,i)=>{ const yy=y+i*22, bw=Math.max(2,(w-LW-VW)*it.v/mx);
    s+=tx(x,yy+12,it.label,{cls:it.hl?"lbl":""})+r(x+LW,yy+2,bw,13,it.col,it.hl?"":'opacity=".45"')+tx(x+LW+bw+5,yy+12,fmt(it.v),{cls:it.hl?"lbl":""}); });
  return s;
}
const svgWrap = (h, body) => ({ h, svg: body });

/* GQA：Hq 个查询头分成 Hkv 组，每组共用一份 K / V */
function gqa(m, f){
  const Hq=Math.max(1,f.Hq|0), Hkv=Math.max(1,Math.min(Hq,f.Hkv|0)), g=Hq/Hkv, show=Math.min(Hq,48), kshow=Math.min(Hkv,Math.ceil(show/g));
  const x0=30, w=W-60, qx=i=>x0+(i+0.5)*w/show, qw=Math.min(12,w/show*0.7);
  const kx=j=>{ const a=Math.ceil(j*g), b=Math.min(show,Math.ceil((j+1)*g))-1; return (qx(a)+qx(Math.max(a,b)))/2; };
  let s=tx(x0,16,t("arch.viz.gqa.q",{n:Hq}),{cls:"lbl"});
  for(let i=0;i<show;i++){ const grp=Math.floor(i/g); if(grp>=kshow) break;
    s+=ln(qx(i),36,kx(grp),92,PAL[grp%PAL.length],1,null)+r(qx(i)-qw/2,24,qw,12,PAL[grp%PAL.length]); }
  if(Hq>show) s+=tx(W-30,16,t("arch.viz.more",{n:Hq-show}),{a:"end"});
  for(let j=0;j<kshow;j++){ const bw=Math.max(6,Math.min(40,g*w/show*0.8)); s+=r(kx(j)-bw/2,92,bw,26,PAL[j%PAL.length]); }
  s+=tx(x0,138,t("arch.viz.gqa.kv",{n:Hkv,g:fI(g)}),{cls:"lbl"});
  const per=h=>2*h*f.dh;
  s+=bars(x0,154,w,[{label:t("arch.viz.gqa.mha",{n:Hq}),v:per(Hq),col:"var(--c-kv)"},{label:t("arch.viz.gqa.cur",{n:Hkv}),v:per(Hkv),col:"var(--c-kv)",hl:true},{label:t("arch.viz.gqa.mqa"),v:per(1),col:"var(--c-kv)"}],v=>t("arch.viz.elemTok",{n:fInt(v)}));
  s+=tx(x0,236,t("arch.viz.gqa.note",{g:fI(g)}));
  return svgWrap(246, s);
}

/* 滑窗：每个 token 只看最近 W 个；全局层看整段 */
function swa(m, f, wl){
  const n=Math.max(1,wl.n), Wd=f.W||4096, cells=64, x0=30, w=W-60, cw=w/cells, keep=Math.max(1,Math.round(Math.min(n,Wd)/n*cells));
  let s=tx(x0,16,t("arch.viz.swa.ctx",{n:fInt(n)}),{cls:"lbl"});
  for(let i=0;i<cells;i++){ const on=i>=cells-keep; s+=r(x0+i*cw+1,26,cw-2,22,on?"var(--c-attn)":"var(--soft)"); }
  s+=r(x0+(cells-1)*cw+1,26,cw-2,22,"var(--ink)");
  s+=tx(x0,66,t("arch.viz.swa.win",{w:fInt(Wd),keys:fInt(Math.min(n,Wd))}));
  for(let i=0;i<cells;i++) s+=r(x0+i*cw+1,78,cw-2,22,"var(--c-attn)",'opacity=".55"');
  s+=r(x0+(cells-1)*cw+1,78,cw-2,22,"var(--ink)");
  s+=tx(x0,118,t("arch.viz.swa.glob",{keys:fInt(n)}));
  // 层的排布：滑窗 / 全局
  const L=f.n, gE=f.gEvery||4, gO=f.gOff||0, lw=Math.min(14,w/L);
  s+=tx(x0,146,t("arch.viz.swa.stack",{swa:f.swa||0,glob:L-(f.swa||0)}),{cls:"lbl"});
  for(let i=0;i<L;i++){ const isS=(f.swa||0)>0 && i%gE!==gO; s+=r(x0+i*lw,154,lw-2,20,"var(--c-attn)",isS?'opacity=".45"':""); }
  s+=bars(x0,186,w,[{label:t("arch.viz.swa.barSwa"),v:Math.min(n,Wd),col:"var(--c-kv)",hl:true},{label:t("arch.viz.swa.barGlob"),v:n,col:"var(--c-kv)"}],v=>t("arch.viz.tokens",{n:fInt(v)}));
  return svgWrap(236, s);
}

/* MLA：每个 token 只缓存一个 kv_lora + rope 维的 latent；decode 把 W_uk / W_uv 吸收进 q / o，所有头在 latent 上做 MQA */
function mla(m, f, wl, dsaMode){
  const H=f.H, dq=f.nope+f.rope, lat=f.kvLora+f.rope, x0=30;
  const exp=H*(dq+f.vh), ratio=exp/lat;
  let s=tx(x0,16,t("arch.viz.mla.cache"),{cls:"lbl"});
  s+=bars(x0,26,W-60,[{label:t("arch.viz.mla.expanded",{h:H}),v:exp,col:"var(--c-kv)"},{label:t("arch.viz.mla.latent",{c:f.kvLora,r:f.rope}),v:lat,col:"var(--c-kv)",hl:true}],v=>t("arch.viz.elemTok",{n:fInt(v)}));
  s+=tx(x0,84,t("arch.viz.mla.ratio",{x:fI(ratio)}));
  // decode：H 个头 → 一个 latent（MQA）
  const hx=i=>x0+20+i*(260/Math.min(H,32)), show=Math.min(H,32);
  s+=tx(x0,112,t("arch.viz.mla.decode"),{cls:"lbl"});
  for(let i=0;i<show;i++){ s+=ln(hx(i)+4,138,160,186,"var(--c-attn)",0.8); s+=r(hx(i),126,8,12,"var(--c-attn)"); }
  s+=r(120,186,80,22,"var(--c-kv)")+tx(160,201,t("arch.viz.mla.latBox",{n:lat}),{a:"middle",fill:"var(--panel)"});
  s+=tx(x0,228,t("arch.viz.mla.decNote",{h:H,i:fI(2*H*(lat+f.kvLora)/(lat*2))}));
  // prefill：kv_b 把 latent 展开成每头 K / V
  const px=360;
  s+=tx(px,112,t(dsaMode?"arch.viz.mla.prefillDsa":"arch.viz.mla.prefill"),{cls:"lbl"});
  s+=r(px,150,60,22,"var(--c-kv)")+tx(px+30,165,t("arch.viz.mla.tag"),{a:"middle",fill:"var(--panel)"});
  for(let i=0;i<8;i++){ s+=ln(px+60,161,px+120,128+i*10,"var(--muted)",0.8); s+=r(px+120,124+i*10,dsaMode?40:150,7,dsaMode?"var(--c-kv)":"var(--c-attn)",dsaMode?'opacity=".4"':""); }
  s+=tx(px,228,t(dsaMode?"arch.viz.mla.preNoteDsa":"arch.viz.mla.preNote",{h:H,d:dq,v:f.vh}));
  return svgWrap(240, s);
}

/* DSA：indexer（fp8、少量头）扫全部 key 打分，主注意力只读 top-k 个 latent */
function dsa(m, f, wl, x){
  const n=x.n, kp=f.kpool||1, pools=x.pools, topk=f.topk, cells=96, x0=30, w=W-60, cw=w/cells;
  const keepFrac=x.keep, rnd=rng(n*31+topk), scores=[...Array(cells)].map(()=>rnd()), cut=[...scores].sort((a,b)=>b-a)[Math.max(0,Math.round(keepFrac*cells)-1)];
  let s=tx(x0,16,t("arch.viz.dsa.scan",{n:fInt(n),p:fInt(pools),kp}),{cls:"lbl"});
  for(let i=0;i<cells;i++){ const h=6+scores[i]*30, on=scores[i]>=cut; s+=r(x0+i*cw+0.5,62-h,cw-1,h,on?"var(--c-attn)":"var(--line)"); }
  s+=tx(x0,80,t(n>topk?"arch.viz.dsa.pick":"arch.viz.dsa.all",{k:fInt(topk),pct:pc(keepFrac)}));
  const idxB=x.idxB, attB=x.attB, denseB=x.allB;
  s+=tx(x0,108,t("arch.viz.dsa.bytes"),{cls:"lbl"});
  s+=bars(x0,116,w,[{label:t("arch.viz.dsa.idx",{h:f.idxH,d:f.idxD}),v:idxB,col:"var(--c-lin)",hl:true},{label:t("arch.viz.dsa.sparse",{k:fInt(Math.min(n,topk))}),v:attB,col:"var(--c-kv)",hl:true},{label:t("arch.viz.dsa.dense"),v:denseB,col:"var(--c-kv)"}],fB);
  const idxF=x.idxF, attF=x.attF;
  s+=tx(x0,200,t("arch.viz.dsa.flops",{a:fF(idxF),b:fF(attF)}));
  s+=tx(x0,220,t("arch.viz.dsa.note",{k:fInt(topk)}));
  return svgWrap(232, s);
}

/* CSA（DeepSeek-V4.1）：每个 query 读 ≤ top-k 个压缩 latent（indexer 打分选出来）+ 最近 W 个滑窗 latent；K、V 是同一份 dh 维 latent，
   所有头共用（MQA）。数字是 Arch.explain 给的每条序列、每层平均 */
function csa(m, f, wl, x){
  const n=x.n, cells=96, x0=30, w=W-60, cw=w/cells, keep=x.keep, all=x.scored<=0;
  const rnd=rng(n*31+f.topk), scores=[...Array(cells)].map(()=>rnd()), cut=[...scores].sort((a,b)=>b-a)[Math.max(0,Math.round(keep*cells)-1)];
  let s=tx(x0,16,t("arch.viz.csa.scan",{n:fInt(n)}),{cls:"lbl"});
  for(let i=0;i<cells;i++){ const h=6+scores[i]*30, on=all||scores[i]>=cut; s+=r(x0+i*cw+0.5,62-h,cw-1,h,on?"var(--c-kv)":"var(--line)"); }
  s+=tx(x0,80,all?t("arch.viz.csa.all",{k:fInt(f.topk)}):t("arch.viz.csa.pick",{k:fInt(f.topk),pct:pc(keep),ni:x.nIdx,cap:fInt(x.cap)}));
  // 滑窗：原始 token 一行，最后 W 个是这一层自己的滑窗 latent
  const win=Math.max(1,Math.round(Math.min(n,f.W)/n*cells));
  for(let i=0;i<cells;i++) s+=r(x0+i*cw+1,90,cw-2,14,i>=cells-win?"var(--c-attn)":"var(--soft)");
  s+=r(x0+(cells-1)*cw+1,90,cw-2,14,"var(--ink)");
  s+=tx(x0,120,t("arch.viz.csa.win",{w:fInt(f.W),keys:fInt(x.ks)}));
  s+=tx(x0,148,t("arch.viz.csa.bytes"),{cls:"lbl"});
  s+=bars(x0,156,w,[{label:t("arch.viz.csa.idx",{h:f.idxH,d:f.idxD}),v:x.idxB,col:"var(--c-lin)",hl:true},{label:t("arch.viz.csa.cmp",{k:fInt(x.kc)}),v:x.kc*x.latB,col:"var(--c-kv)",hl:true},
    {label:t("arch.viz.csa.swa",{w:fInt(x.ks)}),v:x.ks*x.swaB,col:"var(--c-attn)",hl:true},{label:t("arch.viz.csa.dense",{n:fInt(n)}),v:n*x.latB,col:"var(--c-kv)"}],fB);
  s+=tx(x0,262,t("arch.viz.csa.flops",{a:fF(x.idxF),b:fF(x.attF)}));
  s+=tx(x0,282,t("arch.viz.csa.note",{h:f.H,d:f.dh,k:fInt(f.topk),w:fInt(f.W)}));
  return svgWrap(294, s);
}
/* CSA 的层：哪些层产生压缩 KV、哪些层自己打分、哪些层复用；prefill 的 bounded replay 从最后一个 kv source 层切开 */
function csaLayers(m, f, wl){
  const P=Common.csaPlan(f), L=P.length, x0=30, w=W-60, lw=w/L, kv=Common.csaKv(m, wl.kvDt);
  const col = q => q.m===0 ? "var(--c-cp)" : q.kv ? "var(--c-kv)" : q.idx ? "var(--c-lin)" : "var(--c-attn)";
  const nSrc=P.filter(q=>q.kv).length, nIdx=P.filter(q=>q.idx&&!q.kv).length, nRe=P.filter(q=>q.m>0&&!q.kv&&!q.idx).length, nSwa=P.filter(q=>q.m===0).length;
  let s=tx(x0,16,t("arch.viz.csa.stack",{L}),{cls:"lbl"});
  P.forEach((q,i)=>{ s+=r(x0+i*lw+0.5,26,lw-1,22,col(q),q.m>0&&!q.kv&&!q.idx?'opacity=".45"':""); if(q.m===2) s+=r(x0+i*lw+0.5,50,lw-1,3,"var(--muted)"); });
  // 每个 kv source 管到下一个 source 之前（同一个 ratio 的层复用它）
  const src=P.map((q,i)=>q.kv?i:-1).filter(i=>i>=0);
  src.forEach((i,j)=>{ const end=(src[j+1]??L)-1, a=x0+i*lw+lw/2, b=x0+end*lw+lw-1; s+=ln(a,60,b,60,"var(--c-kv)",1.5)+ln(a,56,a,64,"var(--c-kv)",1.5); });
  const last=src.length?src[src.length-1]:L, xc=x0+last*lw;
  s+=ln(xc,20,xc,72,"var(--ink)",1,"3 3");
  const lg=[["var(--c-kv)","",t("arch.viz.csa.lgSrc",{n:nSrc})],["var(--c-lin)","",t("arch.viz.csa.lgIdx",{n:nIdx})],["var(--c-attn)",'opacity=".45"',t("arch.viz.csa.lgRe",{n:nRe})],["var(--c-cp)","",t("arch.viz.csa.lgSwa",{n:nSwa})]];
  lg.forEach(([c,o,label],i)=>{ const lx=x0+i*w/4; s+=r(lx,79,10,10,c,o)+tx(lx+14,88,label); });
  s+=tx(x0,106,t("arch.viz.csa.lgNote"));
  s+=tx(x0,122,t("arch.viz.csa.replay",{l:last,w:fInt(f.W)}));
  // 每 token 的压缩 KV：共享后 vs 每个压缩层都自己存一份
  const own=P.filter(q=>q.m>0).reduce((a,q)=>a+(kv.lat+kv.idx)/q.m,0);
  s+=tx(x0,148,t("arch.viz.csa.share",{dt:wl.kvDt.toUpperCase()}),{cls:"lbl"});
  s+=bars(x0,156,w,[{label:t("arch.viz.csa.shared",{n:nSrc}),v:kv.main+kv.index,col:"var(--c-kv)",hl:true},{label:t("arch.viz.csa.own",{n:P.filter(q=>q.m>0).length}),v:own,col:"var(--c-kv)"}],fB);
  s+=tx(x0,214,t("arch.viz.csa.swaTok",{w:fInt(f.W),n:L,b:fB(kv.swaTok*f.W)}));
  return svgWrap(226, s);
}
/* CSA 模型里只有滑窗的层：每个 query 只读这一层最近 W 个 latent，没有压缩 KV 和 indexer */
function csaSwa(m, f, wl, x){
  const n=Math.max(1,wl.n), cells=64, x0=30, w=W-60, cw=w/cells, keep=Math.max(1,Math.round(Math.min(n,f.W)/n*cells));
  const P=Common.csaPlan(f), only=P.filter(q=>q.m===0).map(q=>q.i);
  let s=tx(x0,16,t("arch.viz.swa.ctx",{n:fInt(n)}),{cls:"lbl"});
  for(let i=0;i<cells;i++) s+=r(x0+i*cw+1,26,cw-2,22,i>=cells-keep?"var(--c-attn)":"var(--soft)");
  s+=r(x0+(cells-1)*cw+1,26,cw-2,22,"var(--ink)");
  s+=tx(x0,66,t("arch.viz.csa.swaOnly",{w:fInt(f.W),keys:fInt(x.ks),l:only.join(", ")}));
  const L=P.length, lw=w/L;
  P.forEach((q,i)=>{ s+=r(x0+i*lw+0.5,80,lw-1,20,q.m===0?"var(--c-cp)":"var(--c-attn)",q.m===0?"":'opacity=".35"'); });
  s+=tx(x0,118,t("arch.viz.csa.swaRest",{n:L-only.length}));
  s+=bars(x0,130,w,[{label:t("arch.viz.csa.swaBar",{w:fInt(x.ks)}),v:x.attB,col:"var(--c-attn)",hl:true},{label:t("arch.viz.csa.swaAll",{n:fInt(n)}),v:n*x.swaB,col:"var(--c-kv)"}],fB);
  s+=tx(x0,190,t("arch.viz.csa.swaNote",{h:f.H,d:f.dh,b:fB(x.swaB)}));
  return svgWrap(202, s);
}

/* 线性注意力：每个头一个 d_k × d_v 的状态，大小与上下文无关；GDN 一个头一个衰减门，KDA 每个通道一个 */
function lin(m, l, wl, kind, x){
  const x0=30, n=x.n, sb=x.sb, st=x.st, kvEq=x.kvEq;
  let s=tx(x0,16,t("arch.viz.lin.state",{h:l.Hv,k:l.dk,v:l.dv}),{cls:"lbl"});
  const gx=x0, gy=28, rows=Math.max(1,Math.min(16,Math.round(l.dk/8))), cols=Math.max(1,Math.min(16,Math.round(l.dv/8)));
  for(let i=0;i<rows;i++) for(let j=0;j<cols;j++) s+=r(gx+j*9,gy+i*9,8,8,"var(--c-st)",'opacity="'+(0.35+0.6*((i*7+j*3)%10)/10).toFixed(2)+'"');
  s+=tx(gx,gy+rows*9+14,t("arch.viz.lin.oneHead",{k:l.dk,v:l.dv}));
  // 门：GDN 一个旋钮，KDA d_k 个
  const kx=gx+cols*9+28;
  s+=tx(kx,gy+8,t(kind==="kda"?"arch.viz.lin.gateKda":"arch.viz.lin.gateGdn",{k:l.dk}),{cls:"lbl"});
  const dials = kind==="kda" ? Math.min(16,rows) : 1;
  for(let i=0;i<dials;i++){ const cy=gy+18+i*9; s+='<circle cx="'+(kx+6)+'" cy="'+(cy+4)+'" r="3.6" style="fill:var(--c-lin)"/>'+ln(kx+12,cy+4,kx+40,cy+4,"var(--c-lin)",1,"2 2"); }
  // 对比：同样 n 个 token，全注意力的 KV（放在状态网格下面）
  const yb=Math.max(182, gy+rows*9+40);
  s+=tx(x0,yb,t("arch.viz.lin.vs",{n:fInt(n)}),{cls:"lbl"});
  s+=bars(x0,yb+8,W-60,[{label:t("arch.viz.lin.barState"),v:st,col:"var(--c-st)",hl:true},{label:t("arch.viz.lin.barKv",{n:fInt(n)}),v:kvEq,col:"var(--c-kv)"}],fB);
  // prefill：C=64 的块
  const px=400, C=64;
  s+=tx(px,16,t("arch.viz.lin.chunk",{c:C}),{cls:"lbl"});
  for(let b=0;b<4;b++){ const bx=px+b*66, by=30; s+='<polygon points="'+bx+','+by+' '+bx+','+(by+48)+' '+(bx+48)+','+(by+48)+'" style="fill:var(--c-lin);opacity:.35"/>';
    if(b<3) s+=ln(bx+50,by+28,bx+60,by+28,"var(--c-st)",2)+'<polygon points="'+(bx+60)+','+(by+24)+' '+(bx+60)+','+(by+32)+' '+(bx+64)+','+(by+28)+'" style="fill:var(--c-st)"/>'; }
  s+=tx(px,106,t("arch.viz.lin.chunkNote1"));
  s+=tx(px,124,t("arch.viz.lin.chunkNote2"));
  s+=tx(px,150,t("arch.viz.lin.decNote",{b:fB(2*st)}));
  return svgWrap(yb+60, s);
}

/* MoE：B 个 token、每个选 k 个 expert，一步碰到 E·(1−(1−k/E)^B) 个；碰到的每个 expert 读一遍权重 */
function moe(m, e, wl, B, x){
  const E=Math.max(1,e.E), k=Math.max(1,e.k), cov=x.cov, hit=Math.round(x.hit), x0=30, w=W-60;
  const cols=Math.min(64,E), rowsN=Math.ceil(E/cols), cs=Math.min(9.5,(w)/cols), rnd=rng(E*7+k*131+B), order=[...Array(E).keys()];
  for(let i=E-1;i>0;i--){ const j=Math.floor(rnd()*(i+1)); [order[i],order[j]]=[order[j],order[i]]; }
  const on=new Set(order.slice(0,hit));
  let s=tx(x0,16,t("arch.viz.moe.grid",{e:E,k,b:fInt(B),hit,pct:pc(cov)}),{cls:"lbl"});
  const rowsShow=Math.min(rowsN,16);
  for(let i=0;i<Math.min(E,cols*rowsShow);i++){ const cx=x0+(i%cols)*cs, cy=26+Math.floor(i/cols)*cs; s+=r(cx,cy,cs-1.5,cs-1.5,on.has(i)?"var(--c-moe)":"var(--soft)"); }
  let y=30+rowsShow*cs;
  if(rowsN>rowsShow) { s+=tx(x0,y+8,t("arch.viz.more",{n:E-cols*rowsShow})); y+=12; }
  const rowsPer = x.rows;
  s+=tx(x0,y+14,t("arch.viz.moe.rows",{r:fI(rowsPer),f:e.F,d:e.Dlat||m.D}));
  const per=x.per;
  s+=bars(x0,y+24,w,[{label:t("arch.viz.moe.read",{hit}),v:hit*per,col:"var(--c-moe)",hl:true},{label:t("arch.viz.moe.all",{e:E}),v:E*per,col:"var(--c-moe)"},{label:t("arch.viz.moe.act",{b:fInt(B)}),v:B*(m.D*2*2),col:"var(--c-act)"}],fB);
  s+=tx(x0,y+100,t(e.Fs>0?(e.Dlat?"arch.viz.moe.extraBoth":"arch.viz.moe.extraShared"):(e.Dlat?"arch.viz.moe.extraLat":"arch.viz.moe.extraNone"),{fs:e.Fs,dl:e.Dlat,d:m.D}));
  return svgWrap(y+112, s);
}

/* dense FFN：一份 3·D·F 的权重被 B 行复用，强度 ≈ B（BF16） */
function dense(m, F, wl, B, x){
  const x0=30, I=x.I, ridge=x.ridge, Bst=x.Bstar;
  let s=tx(x0,16,t("arch.viz.dense.w",{d:m.D,f:F}),{cls:"lbl"});
  s+=r(x0,26,300,90,"var(--c-lm)",'opacity=".55"')+tx(x0+150,76,t("arch.viz.dense.wBox",{b:fB(x.wB)}),{a:"middle",cls:"lbl"});
  const rows=Math.min(32,B); for(let i=0;i<rows;i++) s+=r(x0+330,26+i*90/rows,120,Math.max(1,90/rows-1),"var(--c-act)");
  s+=tx(x0+330,132,t("arch.viz.dense.rows",{b:fInt(B)}));
  s+=tx(x0,160,t("arch.viz.dense.I",{i:fI(I),r:fI(ridge),bs:fInt(Bst)}),{cls:"lbl"});
  s+=tx(x0,180,t("arch.viz.dense.note"));
  return svgWrap(192, s);
}

/* 入口：按模块画；返回 {h, svg} */
function draw(mod, m, wl, hw, g){
  const f=m.full, x=Arch.explain(m,hw,g,wl,mod.id), B=x.B;
  if(mod.id==="full") return f.kind==="gqa" ? gqa(m,f) : f.kind==="mla" ? mla(m,f,wl,false) : f.kind==="csa" ? csa(m,f,wl,x) : dsa(m,f,wl,x);
  if(mod.id==="swa") return f.kind==="csa" ? csaSwa(m,f,wl,x) : swa(m,f,wl);
  if(mod.id==="lin") return lin(m,m.lin,wl,mod.kind,x);
  if(mod.id==="moe") return moe(m,m.moe,wl,B,x);
  return dense(m,x.F,wl,B,x);
}
/* 第二张图：DSA 的 latent 部分沿用 MLA 的图；CSA 是层的分工（KV / top-k 怎么在层间共享） */
const extra = (mod, m, wl) => mod.id!=="full" ? null : m.full.kind==="dsa" ? mla(m,m.full,wl,true) : m.full.kind==="csa" ? csaLayers(m,m.full,wl) : null;

return { draw, extra };
})();
