/* ================= 集群：网络结构（cluster/fabric）和对训练、推理的影响（cluster/impact） =================
   数都来自 Fabric（src/engine/fabric.js），它把这里的配置折算成 hw / topo 再交给训练和推理的同一套模型；这里只画 */
const CL = {
  sel: { scale:Fabric.SCALES, domain:Fabric.ENUMS.domain, nvlFrac:Fabric.ENUMS.nvlFrac, pcie:Fabric.ENUMS.pcie, nic:Fabric.ENUMS.nic, ports:Fabric.ENUMS.ports, over:Fabric.ENUMS.over, qps:Fabric.ENUMS.qps },
  nvsRow: { H100:"dgx-h100-system", H200:"dgx-h100-system", B200:"hgx-blackwell", B300:"hgx-blackwell", GB200:"gb200-nvl72-nvlink", GB300:"gb200-nvl72-nvlink" },
  col: { hbm:"mem", nvl:"accent", p2pPcie:"accent", host:"c-oth", pcie:"s4", nicRaw:"s2", nic:"s2", ring:"s3", a2a:"s5", spine:"s6" },
  pub: () => DB.set("cluster_public"),
  row: id => DB.set("cluster_public").find(p => p.id === "cluster_public:"+id),
  gbs: v => (v>=100 ? fInt(v) : v>=10 ? v.toFixed(1) : v.toFixed(2))+" GB/s",
  pc: d => (d>=0 ? "+" : "")+(d*100).toFixed(Math.abs(d)<0.1 ? 1 : 0)+"%",
  link: p => p.url ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.src)+'</a>' : esc(p.src)
};
function clCtx(){ const {m,hw,topo,g}=ctx(); return { m, hw, topo, g, x:Fabric.resolve(hw,topo,S.net) }; }

/* 机架：下拉框的选项文字跟着硬件变（“跟硬件（400 Gb/s）”这种），每次画之前重填；不起作用的项灰掉 */
function syncNetRack(){
  const { hw, topo, g } = ctx(), auto = Fabric.resolve(hw, topo, Object.assign({}, S.net, { domain:0, pcie:0, nic:0 }));
  const lab = {
    scale: v => t("cluster.rack.opt.scale", { k:v, n:fInt(topo.gpus*v) }),
    domain: v => +v ? (+v===72 ? t("cluster.rack.opt.nvl72") : String(v)) : t("cluster.rack.opt.auto", { v:auto.domain }),
    nvlFrac: v => +v===1 ? "100%" : Math.abs(v-17/18)<1e-9 ? t("cluster.rack.opt.chipDown") : Math.round(v*100)+"%",
    pcie: v => +v ? t("cluster.rack.opt.gen", { g:v, bw:Fabric.PCIE[v] }) : t("cluster.rack.opt.auto", { v:"Gen"+auto.pcie }),
    nic: v => +v ? fInt(v)+" Gb/s" : t("cluster.rack.opt.auto", { v:fInt(auto.nic)+" Gb/s" }),
    ports: v => t("cluster.rack.opt.ports", { p:v, seg:fInt(v*auto.node) }),
    over: v => +v===1 ? t("cluster.rack.opt.over1") : v+":1",
    qps: v => String(v) };
  for(const [k, vals] of Object.entries(CL.sel)){
    const el = document.querySelector('[data-rack] [data-bind="net.'+k+'"]'); if(!el) continue;
    el.innerHTML = vals.map(v => '<option value="'+String(v)+'">'+esc(lab[k](v))+'</option>').join("");
    el.value = String(S.net[k]); if(el.selectedIndex<0){ S.net[k] = Fabric.DEF[k]; el.value = String(Fabric.DEF[k]); }
  }
  markInert("net", Fabric.inert(hw, topo, S.net, g, S.inf));
}
document.addEventListener("click", e => { if(e.target && e.target.id==="cl-reset"){ S.net = clone(Fabric.DEF); syncInputs(); render(); } });

/* 横向条形图（对数轴）：带宽阶梯 */
function clBars(svgId, rows){
  const svg=$(svgId), W=680, lw=250, rw=86, rh=27, H=18+rows.length*rh;
  svg.setAttribute("viewBox", "0 0 "+W+" "+H);
  const vs=rows.map(r=>r.v).filter(v=>v>0), lo=Math.min(...vs)/2, hi=Math.max(...vs)*1.15, X=axis(lo, hi, lw, W-rw, true);
  let s="";
  for(const v of logTicks(lo, hi, 10)) if(v>=lo && v<=hi) s+='<line class="gr" x1="'+X(v)+'" x2="'+X(v)+'" y1="6" y2="'+(H-4)+'"/>';
  rows.forEach((r, i) => { const y=10+i*rh, w=Math.max(2, X(r.v)-lw);
    s+='<g '+tipAttr(r.name, [{v:CL.gbs(r.v), l:r.tip||"", c:r.col}])+'><rect x="0" y="'+(y-2)+'" width="'+W+'" height="'+rh+'" style="fill:transparent"/>'+
      '<text x="'+(lw-8)+'" y="'+(y+13)+'" text-anchor="end">'+esc(r.name)+'</text>'+
      '<rect x="'+lw+'" y="'+(y+3)+'" width="'+w+'" height="14" rx="4" style="fill:'+css(r.col)+'"/>'+
      '<text x="'+(lw+w+6)+'" y="'+(y+14)+'" class="lbl">'+esc(CL.gbs(r.v))+'</text></g>'; });
  svg.innerHTML=s; bindTips(svg);
}
/* NVSwitch 示意：上排 GPU，下排交换芯片，每张卡连到每颗芯片 */
function clNvs(svgId, x, row){
  const svg=$(svgId), W=680, H=230;
  if(x.intra==="pcie"){ svg.innerHTML='<text x="'+W/2+'" y="'+H/2+'" text-anchor="middle" class="lbl">'+esc(t("cluster.nvs.none"))+'</text>'; return; }
  const G=Math.min(x.domain, 8), K=row&&row.chips ? Math.min(row.chips, 18) : 1, gx=i=>60+i*(W-120)/Math.max(1,G-1), kx=j=>K>1 ? 40+j*(W-80)/(K-1) : W/2;
  let s="";
  for(let i=0;i<G;i++) for(let j=0;j<K;j++) s+='<line x1="'+gx(i)+'" y1="62" x2="'+kx(j)+'" y2="160" style="stroke:var(--accent);opacity:'+(K>4?0.28:0.55)+'" stroke-width="1.2"/>';
  for(let i=0;i<G;i++) s+='<rect x="'+(gx(i)-24)+'" y="34" width="48" height="28" rx="5" style="fill:var(--panel);stroke:var(--ink)"/><text x="'+gx(i)+'" y="53" text-anchor="middle">'+(x.domain>8 && i===G-1 ? "…" : "GPU"+i)+'</text>';
  for(let j=0;j<K;j++){ const w=K>9?26:K>1?60:150; s+='<rect x="'+(kx(j)-w/2)+'" y="160" width="'+w+'" height="24" rx="5" style="fill:var(--accent-soft);stroke:var(--accent)"/>'+(K<=9?'<text x="'+kx(j)+'" y="177" text-anchor="middle">'+(K>1?"NVS"+j:"NVSwitch")+'</text>':''); }
  s+='<text x="12" y="20">'+esc(t("cluster.nvs.top", { n:x.domain }))+'</text><text x="12" y="214">'+esc(K>1 ? t("cluster.nvs.chips", { k:row.chips }) : t("cluster.nvs.chipsUnknown"))+'</text>';
  svg.innerHTML=s;
}
/* 机内 PCIe 示意：两颗 CPU、四个 PCIe 交换机，每个下面两张 GPU 两张网卡；高亮当前配置下 GPU → 网卡（和 GPU → GPU）走的路 */
function clPcie(svgId, x){
  const svg=$(svgId), W=680, cpuX=[170,510], swX=[90,250,430,590], yC=40, yS=118, yD=198;
  const box=(cx,cy,w,h,label,st)=>'<rect x="'+(cx-w/2)+'" y="'+(cy-h/2)+'" width="'+w+'" height="'+h+'" rx="5" style="'+st+'"/><text x="'+cx+'" y="'+(cy+4)+'" text-anchor="middle">'+esc(label)+'</text>';
  const ln=(a,b,c,d,st)=>'<line x1="'+a+'" y1="'+b+'" x2="'+c+'" y2="'+d+'" style="'+st+'" stroke-width="1.4"/>';
  const path=(pts,col,dash)=>'<polyline points="'+pts.map(p=>p.join(",")).join(" ")+'" fill="none" style="stroke:'+css(col)+'" stroke-width="3.2" stroke-linejoin="round"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
  const dev=k=>[-54,-18,18,54].map(d=>swX[k]+d);           // 每个交换机下：GPU、GPU、NIC、NIC
  let s=ln(cpuX[0],yC,cpuX[1],yC,"stroke:var(--muted)");
  for(let k=0;k<4;k++){ s+=ln(cpuX[k>>1],yC,swX[k],yS,"stroke:var(--muted)"); for(const xx of dev(k)) s+=ln(swX[k],yS,xx,yD,"stroke:var(--muted)"); }
  const g0=dev(0)[0], g1=dev(0)[1], n0=dev(0)[2], g4=dev(2)[0];
  if(x.gdr) s+=path([[g0,yD],[swX[0],yS],[n0,yD]],"accent");
  else s+=path([[g0,yD],[swX[0],yS],[cpuX[0],yC],[swX[0]+8,yS],[n0,yD]],"warn","6 4");
  if(x.intra==="pcie") s+=path([[g0-4,yD],[swX[0]-4,yS],[g1,yD]],"s3")+path([[g0+4,yD],[swX[0]+4,yS],[cpuX[0],yC+4],[cpuX[1],yC+4],[swX[2],yS],[g4,yD]],"s3","3 3");
  s+=box(cpuX[0],yC,120,30,"CPU0","fill:var(--panel);stroke:var(--ink)")+box(cpuX[1],yC,120,30,"CPU1","fill:var(--panel);stroke:var(--ink)");
  s+='<text x="'+((cpuX[0]+cpuX[1])/2)+'" y="'+(yC-8)+'" text-anchor="middle" class="lbl">'+esc(t("cluster.pcie.socket"))+'</text>';
  for(let k=0;k<4;k++){ s+=box(swX[k],yS,96,26,t("cluster.pcie.sw"),"fill:var(--accent-soft);stroke:var(--accent)");
    dev(k).forEach((xx,i)=>{ s+=box(xx,yD,32,26,i<2?"G"+(2*k+i):"N"+(2*k+i-2),i<2?"fill:var(--panel);stroke:var(--ink)":"fill:var(--panel);stroke:var(--s2)"); }); }
  s+='<text x="12" y="'+(yD+38)+'" class="lbl">'+esc(t("cluster.pcie.legend"))+'</text>';
  svg.innerHTML=s;
}

function renderClusterFabric(){
  syncNetRack();
  const { m, hw, topo, g, x } = clCtx(), net=S.net;
  const L = Fabric.ladder(hw, topo, net, g), get = k => (L.find(r=>r.key===k)||{}).GBs;
  const nvl = get("nvl") ?? get("p2pPcie"), nic = get("nic"), multi = x.N > x.domain;
  $("cl-ladder-lede").innerHTML = t(x.intra==="pcie" ? "cluster.fab.ladder.ledePcie" : "cluster.fab.ladder.lede", { hw:hw.name, n:V(fInt(x.N)), dom:V(x.domain), intra:t("cluster.intra."+x.intra), nic:V(fInt(x.nic)) });
  kpis($("cl-kpis"), [
    [t("cluster.fab.kpi.nvl"), CL.gbs(nvl), t(x.intra==="pcie" ? "cluster.fab.kpi.nvlFootPcie" : "cluster.fab.kpi.nvlFoot", { n:x.domain })],
    [t("cluster.fab.kpi.nic"), CL.gbs(nic), nic < x.nic/8 ? t("cluster.fab.kpi.nicCap", { raw:fInt(x.nic) }) : t("cluster.fab.kpi.nicFoot", { raw:fInt(x.nic) }), nic < x.nic/8],
    [t("cluster.fab.kpi.ratio"), (nvl/nic).toFixed(nvl/nic>=10?0:1)+"×", t("cluster.fab.kpi.ratioFoot")],
    [t(x.intra==="pcie" ? "cluster.fab.kpi.machines" : "cluster.fab.kpi.domains"), fInt(Math.ceil(x.N/x.domain)), t(x.intra==="pcie" ? "cluster.fab.kpi.machinesFoot" : "cluster.fab.kpi.domainsFoot", { n:fInt(x.N), d:x.domain })]]);
  clBars("cl-ladder", L.map(r => ({ v:r.GBs, col:CL.col[r.key], name:t("cluster.ladder."+r.key, { gen:r.gen, d:r.d, n:fInt(r.n||0), over:r.over }), tip:t("cluster.ladder.tip."+r.key) })));
  const ring = L.find(r=>r.key==="ring");
  $("cl-ladder-read").innerHTML = ul([
    t("cluster.fab.read.gap", { r:V((hw.hbm/nic).toFixed(0)+"×") }),
    multi && ring ? t("cluster.fab.read.ring", { d:ring.d, bw:V(CL.gbs(ring.GBs)), nic:CL.gbs(nic) }) : t("cluster.fab.read.oneDomain", { n:fInt(x.N), dom:x.domain }),
    multi ? t("cluster.fab.read.a2a", { bw:V(CL.gbs(get("a2a"))) }) : "" ]);

  // ---- NVSwitch ----
  const row = CL.row(CL.nvsRow[S.hw]);
  clNvs("cl-nvs", x, row);
  $("cl-nvs-cap").innerHTML = x.intra==="pcie" ? t("cluster.nvs.capPcie") : row ? t("cluster.nvs.cap", { p:esc(row.platform), v:esc(row.value), src:CL.link(row) }) + (row.chips && row.links ? " "+t("cluster.nvs.check", { k:row.chips, p:row.ports, g:row.gpus, l:row.links, tot:fInt(row.chips*row.ports) }) : "") : "";
  const ar = Fabric.arSweep(hw, topo, net, g);
  $("cl-ar-lede").innerHTML = t("cluster.fab.ar.lede", { n:V(ar.n), bw:CL.gbs(nvl) });
  const arSer = [["ring","s3","",t("cluster.ar.ring")],["oneshot","s2","5 4",t("cluster.ar.oneshot")],["twoshot","s4","5 4",t("cluster.ar.twoshot")],["nvls","accent","",t("cluster.ar.nvls")]]
    .filter(([k]) => k!=="nvls" || x.intra!=="pcie").map(([k,col,dash,name]) => ({ name, col, dash, pts:ar.rows.map(r=>[r.bytes, r[k]]) }));
  lineChart("cl-ar", { series:arSer, xLog:true, yLog:true, xBase:4, xFmt:fB, yFmt:fT, tipFmt:fT, xLab:t("cluster.ar.x"), yLab:t("cluster.ar.y") });
  legendInto("cl-ar-legend", arSer.map(s=>({ name:s.name, col:s.col, line:true })));
  const big = ar.rows[ar.rows.length-1], small = ar.rows[0];
  $("cl-nvs-read").innerHTML = ul([
    t("cluster.fab.ar.readSmall", { b:fB(small.bytes), one:fT(small.oneshot), ring:fT(small.ring) }),
    x.intra!=="pcie" ? t("cluster.fab.ar.readBig", { b:fB(big.bytes), nv:fT(big.nvls), ring:fT(big.ring), x:V((big.ring/big.nvls).toFixed(1)+"×") }) : "",
    x.intra!=="pcie" ? t("cluster.fab.ar.readScope") : "",
    x.nvlFrac<1 && x.intra!=="pcie" ? t("cluster.fab.ar.readFrac", { f:Math.round(x.nvlFrac*100)+"%" }) : "" ]);

  // ---- 机内 PCIe ----
  clPcie("cl-pcie", x);
  $("cl-pcie-cap").innerHTML = t(x.gdr ? "cluster.pcie.capGdr" : "cluster.pcie.capNoGdr", { host:fInt(hw.host) }) + (x.intra==="pcie" ? " "+t("cluster.pcie.capP2p") : "");
  const fit = Fabric.pcieFit(hw, topo, net);
  table($("cl-gen"), [t("cluster.pcie.gen.head"), t("cluster.pcie.gen.bw")].concat(fit[0].fits.map(f=>fInt(f.nic)+" Gb/s")),
    fit.map(r => ({ cls:r.cur?"hl":"", cells:["Gen"+r.gen, fInt(r.GBs)].concat(r.fits.map(f => f.ok ? "✓" : '<span style="color:var(--warn)">✗</span>')) })), [1]);
  const tp = Fabric.tpPcie(m, hw, topo, S.inf, net, g), tpMax = tp[tp.length-1];
  $("cl-tp-lede").innerHTML = t("cluster.fab.tp.lede", { model:esc(m.short), engine:ENGN[S.inf.engine]||S.inf.engine, B:S.inf.B, ctx:fInt(S.inf.n), gen:x.pcie, bw:fInt(Fabric.PCIE[x.pcie]) });
  table($("cl-tp"), [t("cluster.fab.tp.head.tp"), t("cluster.fab.tp.head.nvs"), t("cluster.fab.tp.head.pcie"), t("cluster.fab.tp.head.ratio"), t("cluster.fab.tp.head.gain")],
    tp.map((r, i) => [String(r.tp), fT(r.nvswitch), fT(r.pcie), (r.pcie/r.nvswitch).toFixed(2)+"×", i ? t("cluster.fab.tp.gain", { nv:CL.pc(r.nvswitch/tp[i-1].nvswitch-1), pc:CL.pc(r.pcie/tp[i-1].pcie-1) }) : "—"]), [1,2,3]);
  const best = tp.reduce((a,b) => b.pcie < a.pcie ? b : a);
  $("cl-pcie-read").innerHTML = ul([
    t(best.tp<tpMax.tp ? "cluster.fab.tp.readEat" : "cluster.fab.tp.readStill", { best:best.tp, max:tpMax.tp, slow:V((tpMax.pcie/tpMax.nvswitch).toFixed(2)+"×") }),
    t("cluster.fab.pcie.readNuma"), t("cluster.fab.pcie.readGdr") ]);

  // ---- scale-out ----
  const so = Fabric.scaleOut(hw, topo, net, g);
  kpis($("cl-out-kpis"), [
    [t("cluster.out.kpi.seg"), fInt(so.segment), t("cluster.out.kpi.segFoot", { p:so.node, ports:x.ports })],
    [t("cluster.out.kpi.segs"), fInt(so.segs), t("cluster.out.kpi.segsFoot", { n:fInt(so.N) })],
    [t("cluster.out.kpi.leaves"), fInt(so.leaves), t("cluster.out.kpi.leavesFoot", { up:so.up, ports:x.ports })],
    [t("cluster.out.kpi.cross"), CL.gbs(so.crossGBs), so.segs>1 ? t("cluster.out.kpi.crossFoot", { over:so.over }) : t("cluster.out.kpi.crossOne")]]);
  const ov = Fabric.overSweep(hw, topo, net, g), nOv = Math.max(x.N, 2*x.segment);
  $("cl-over-lede").innerHTML = t("cluster.fab.over.lede", { n:V(fInt(nOv)), seg:fInt(x.segment), q:x.qps, two:x.N < 2*x.segment ? t("cluster.fab.over.two") : "" });
  const kinds = [["ring","s3"],["p2p","s2"],["a2a","s5"]], ovSer = [];
  for(const [k,col] of kinds) for(const route of ["adaptive","ecmp"]) ovSer.push({ name:t("cluster.over."+k)+" · "+t("cluster.route."+route), short:route==="adaptive"?t("cluster.over."+k):"", col, dash:route==="ecmp"?"5 4":"", pts:ov.map(r=>[r.over, r[route][k]]) });
  lineChart("cl-over", { series:ovSer, xLog:true, xBase:2, yMin:0, yMax:1.05, xFmt:v=>v+":1", yFmt:v=>Math.round(v*100)+"%", labelEnds:false, xLab:t("cluster.over.x"), yLab:t("cluster.over.y"),
    vlines:[{ x:x.over, label:t("cluster.over.now") }] });
  legendInto("cl-over-legend", ovSer.map(s=>({ name:s.name, col:s.col, line:true })));
  const qp = Fabric.qpSweep(hw, topo, net, g);
  $("cl-qp-lede").innerHTML = t("cluster.fab.qp.lede", { ports:x.ports, up:qp[0].up });
  lineChart("cl-qp", { series:[{ name:t("cluster.qp.eff"), col:"s2", pts:qp.map(r=>[r.qps, r.eff]) }], xLog:true, xBase:2, yMin:0, yMax:1, xFmt:String, yFmt:v=>Math.round(v*100)+"%",
    xLab:t("cluster.qp.x"), yLab:t("cluster.qp.y"), vlines:[{ x:x.qps, label:t("cluster.over.now") }] });
  const at = (r, k) => (r[k]*100).toFixed(0)+"%", o4 = ov.find(r=>r.over===4)||ov[ov.length-1];
  $("cl-out-read").innerHTML = ul([
    t("cluster.fab.out.readRing"),
    t("cluster.fab.out.readA2a", { p2p:V(at(o4.adaptive,"p2p")), a2a:V(at(o4.adaptive,"a2a")) }),
    t("cluster.fab.out.readEcmp", { e1:V((qp[0].eff*100).toFixed(0)+"%"), e16:V((qp.find(r=>r.qps===16).eff*100).toFixed(0)+"%") }),
    t("cluster.fab.out.readHpn") ]);

  // ---- 公开数字 ----
  const P = CL.pub();
  table($("cl-spec"), [t("cluster.pub.head.platform"), t("cluster.pub.head.what"), t("cluster.pub.head.value"), t("cluster.pub.head.src")],
    P.filter(p=>p.kind==="spec").map(p => [esc(p.platform||"—"), esc(p.what), esc(p.value), CL.link(p)]));
  table($("cl-pub"), [t("cluster.pub.head.what"), t("cluster.pub.head.value"), t("cluster.pub.head.src")],
    P.filter(p=>p.kind!=="spec").map(p => [esc(p.what)+(p.platform?'<br><span class="tag">'+esc(p.platform)+'</span>':''),
      esc(p.value)+(p.caveat?'<br><span style="color:var(--warn)">'+esc(p.caveat)+'</span>':''), CL.link(p)+'<br><span class="tag">'+esc(p.date)+'</span>']));
}

function renderClusterImpact(){
  syncNetRack();
  const { m, hw, topo, g, x } = clCtx(), net=S.net, ci=S.inf, ct=S.trn;
  const sT = Fabric.split("train", m, hw, topo, ct, net, g), sP = Fabric.split("prefill", m, hw, topo, ci, net, g), sD = Fabric.split("decode", m, hw, topo, ci, net, g);
  // 配置跑不起来（计算核心标成 invalid 的警告）时照样算时间，但在顶上说清楚，和训练页、推理页一样
  const bad = r => r.warn.filter(w => w && w.bad), badT = bad(Fabric.train(m, hw, topo, ct, net, g)), badI = bad(Fabric.decode(m, hw, topo, ci, net, g));
  setWarn(badT.map(w => t("cluster.warn.train", { w:String(w) })).concat(badI.map(w => t("cluster.warn.infer", { w:String(w) }))));
  const moe = ci.moePar && m.moe.E ? ci.moePar : "tp";
  $("cl-imp-lede").innerHTML = t("cluster.imp.lede", { model:esc(m.short), hw:hw.name, n:V(fInt(x.N)), k:x.scale, dom:x.domain, intra:t("cluster.intra."+x.intra),
    fw:ct.fw, tp:ct.tp, cp:ct.cp, ep:ct.ep, pp:ct.pp, engine:ENGN[ci.engine]||ci.engine, ni:V(ci.N), dp:ci.dpA, moe, B:ci.B });
  const share = s => s.t>0 ? (s.nic+s.nvl)/s.t : 0;
  kpis($("cl-imp-kpis"), [
    [t("cluster.imp.kpi.train"), fT(sT.t), t("cluster.imp.kpi.netFoot", { nic:pct(sT.nic,sT.t), nvl:pct(sT.nvl,sT.t) })],
    [t("cluster.imp.kpi.prefill"), fT(sP.t), t("cluster.imp.kpi.netFoot", { nic:pct(sP.nic,sP.t), nvl:pct(sP.nvl,sP.t) })],
    [t("cluster.imp.kpi.decode"), fT(sD.t), t("cluster.imp.kpi.netFoot", { nic:pct(sD.nic,sD.t), nvl:pct(sD.nvl,sD.t) })]]);
  const bars = [["train",sT],["prefill",sP],["decode",sD]];
  $("cl-bars").innerHTML = bars.map(([k]) => '<h3>'+esc(t("cluster.imp.bar."+k))+'</h3><div class="bar" id="cl-bar-'+k+'"></div><div class="barlist" id="cl-bl-'+k+'"></div>').join("");
  for(const [k,s] of bars) stackBar($("cl-bar-"+k), $("cl-bl-"+k), [
    { label:t("cluster.imp.seg.rest"), v:s.rest, color:css("c-oth") }, { label:t("cluster.imp.seg.nvl"), v:s.nvl, color:css("accent") }, { label:t("cluster.imp.seg.nic"), v:s.nic, color:css("s2") }], s.t, fT);
  const G = Fabric.groups(m, hw, topo, ct, net, g), cross = G.filter(r=>!r.intra);
  $("cl-grp-lede").innerHTML = t("cluster.imp.grp.lede");
  const kind = r => r.intra ? (x.intra==="pcie" ? t("cluster.imp.grp.pcie") : "NVLink") : t("cluster.imp.grp.cross", { d:r.d, segs:r.m<r.n ? t("cluster.imp.grp.segs") : "" });
  table($("cl-groups"), [t("cluster.imp.grp.head.dim"), t("cluster.imp.grp.head.n"), t("cluster.imp.grp.head.stride"), t("cluster.imp.grp.head.link"), t("cluster.imp.grp.head.ring"), t("cluster.imp.grp.head.p2p"), t("cluster.imp.grp.head.a2a")],
    G.map(r => ({ cls:r.intra?"":"hl", cells:[t("cluster.dim."+r.dim), fInt(r.n), fInt(r.stride), kind(r)+(r.inNet?" · "+t(r.intra?"cluster.imp.grp.nvls":"cluster.imp.grp.sharp"):""), CL.gbs(r.bw/1e9), CL.gbs(r.p2p/1e9), CL.gbs(r.a2a/1e9)] })), [1,2,4,5,6]);
  const top = bars.map(([k,s]) => ({ k, s, f:share(s) })).sort((a,b)=>b.f-a.f)[0];
  $("cl-imp-read").innerHTML = ul([
    t("cluster.imp.read.method"),
    cross.length ? t("cluster.imp.read.cross", { dims:cross.map(r=>t("cluster.dim."+r.dim)).join(t("cluster.listSep")) }) : t("cluster.imp.read.noCross", { n:fInt(x.N), dom:x.domain }),
    top.f>0.005 ? t("cluster.imp.read.top", { what:t("cluster.imp.bar."+top.k), f:V(pct(top.f,1)) }) : "",
    x.scale>1 ? t("cluster.imp.read.weak", { k:x.scale }) : "" ]);

  // ---- 带宽还是延迟 ----
  const K = Fabric.KNOBS.map(k=>k.key), sens = [["train",ct],["prefill",ci],["decode",ci]].map(([k,c]) => ({ k, r:Fabric.sens(k, m, hw, topo, c, net, g) }));
  table($("cl-sens"), [t("cluster.sens.head.what"), t("cluster.sens.head.t0")].concat(K.map(k=>t("cluster.sens.knob."+k))),
    sens.map(({k,r}) => { const mx = Math.max(...r.rows.map(x=>Math.abs(x.d)));
      return [t("cluster.imp.bar."+k), fT(r.t0)].concat(r.rows.map(x => Math.abs(x.d)<0.0005 ? "0" : (Math.abs(x.d)===mx && mx>0.01 ? "<b>"+CL.pc(x.d)+"</b>" : CL.pc(x.d)))); }), [1,2,3,4,5,6,7]);
  const d = (k, key) => sens.find(s=>s.k===k).r.rows.find(x=>x.key===key).d;
  $("cl-sens-read").innerHTML = ul([
    t("cluster.sens.read.lat", { tr:V(CL.pc(d("train","lat2"))), pf:V(CL.pc(d("prefill","lat2"))), dc:V(CL.pc(d("decode","lat2"))) }),
    t("cluster.sens.read.bw", { tr:V(CL.pc(d("train","nicBwHalf"))), pf:V(CL.pc(d("prefill","nicBwHalf"))), dc:V(CL.pc(d("decode","nicBwHalf"))) }),
    t("cluster.sens.read.cliff"), t("cluster.sens.read.lat0") ]);

  // ---- NVLink 域多大 ----
  const dT = Fabric.domainSweep("train", m, hw, topo, ct, net, g).filter(L => L.pts.length>1), dD = Fabric.domainSweep("decode", m, hw, topo, ci, net, g);
  $("cl-dom-lede").innerHTML = [t("cluster.imp.dom.lede"), dT.length ? t("cluster.imp.dom.ledeTrain", { n:V(fInt(x.N)) }) : "", dD.length ? t("cluster.imp.dom.ledeDec", { B:ci.B }) : ""].filter(Boolean).join(" ");
  const cols = ["s1","s2","s3","s5"], vl = [{ x:x.domain, label:t("cluster.over.now") }];
  $("cl-dom-t-fig").hidden = !dT.length;
  if(dT.length) lineChart("cl-dom-t", { series:dT.map((L,i)=>({ name:t("cluster.dom.ep", { ep:L.key }), col:cols[i%4], pts:L.pts.map(p=>[p.domain, p.t]) })), xLog:true, xBase:2, xFmt:String, yFmt:fT, tipFmt:fT, yMin:0,
    xLab:t("cluster.dom.x"), yLab:t("cluster.dom.yTrain"), vlines:vl });
  $("cl-dom-d-fig").hidden = !dD.length;
  if(dD.length) lineChart("cl-dom-d", { series:dD.map((L,i)=>({ name:t("cluster.dom.inst", { n:L.key }), col:cols[i%4], pts:L.pts.map(p=>[p.domain, p.t]) })), xLog:true, xBase:2, xFmt:String, yFmt:fT, tipFmt:fT, yMin:0,
    xLab:t("cluster.dom.x"), yLab:t("cluster.dom.yDec"), vlines:vl });
  const span = L => { const a=L.pts[0], b=L.pts[L.pts.length-1]; return a && b ? a.t/b.t : 1; }, wT = dT.reduce((a,b)=>span(b)>span(a)?b:a, dT[0]);
  $("cl-dom-read").innerHTML = ul([
    wT ? t("cluster.imp.dom.readTrain", { ep:wT.key, lo:wT.pts[0].domain, hi:wT.pts[wT.pts.length-1].domain, x:V(span(wT).toFixed(1)+"×") }) : badT.length ? t("cluster.imp.dom.readBad") : t("cluster.imp.dom.readSmall", { n:fInt(x.N), d:Fabric.DOMAINS[0] }),
    dD.length ? t("cluster.imp.dom.readDec", { x:V(span(dD[dD.length-1]).toFixed(2)+"×"), n:dD[dD.length-1].key }) : t("cluster.imp.dom.readDense"),
    t("cluster.imp.dom.readWhy") ]);

  // ---- 大块传输 ----
  const X = Fabric.transfers(m, hw, topo, ci, net, g);
  table($("cl-xfer"), [t("cluster.xfer.head.what"), t("cluster.xfer.head.bytes"), t("cluster.xfer.head.nic"), t("cluster.xfer.head.other"), t("cluster.xfer.head.vs")], [
    [t("cluster.xfer.kv", { p:fInt(X.P) }), fB(X.kv), fT(X.kvNic), t("cluster.xfer.kvNvl", { t:fT(X.kvNvl) }), t("cluster.xfer.kvVs", { r:pct(X.kvNic, X.ttft), ttft:fT(X.ttft) })],
    [t("cluster.xfer.w"), fB(X.w), fT(X.wOne), t("cluster.xfer.wShard", { n:X.Ninf, t:fT(X.wShard) }), "—"]], [1,2]);
  $("cl-xfer-read").innerHTML = ul([ t("cluster.xfer.readKv"), t("cluster.xfer.readW") ]);
}
