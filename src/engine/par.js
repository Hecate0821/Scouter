/* 并行原语：CP 三种取 K / V 的办法、EP 五个后端、FSDP 每层的字节与时间。训练核心和后训练的并行一节都只调这里，
   两边看到的是同一组公式。所有函数按“一层、最忙那张卡”算，调用方再乘层数和文档数。 */
const Par = (function(){
const { link } = Common;

const binomRatio = (Z, z, K) => { let r=1; for(let i=0;i<K;i++) r *= Math.max(0, Z-z-i)/(Z-i); return r; };   // C(Z−z,K)/C(Z,K)
const pDest = (Z, E, K) => E<=1 ? 1 : 1-binomRatio(Z, Z/E, K);   // 一个 token 命中某个 rank 的概率（不放回地选 K 个 expert）
const pow2Dom = topo => Math.pow(2, Math.floor(Math.log2(topo.domain)));   // 2 的幂的组能用满的 NVLink 域：HGX 8，NVL72 64
const lk = (hw, topo, n, stride) => { const l=link(hw,topo,n,stride); return Object.assign({ n, kind: n<=1 ? "—" : l.intra ? "NVLink" : "IB" }, l); };
/* 一段线上传输：calls 次调用的固定延迟 + 字节 / 有效带宽 */
const wireT = (bytes, l, eff, calls, g) => (l.n<=1||bytes<=0) ? 0 : (calls||1)*(l.intra?g.alphaNv:g.alphaIb)*1e-6 + bytes/(l.bw*eff);
/* 一次集合通信：pass（fwd / rc / bwd）、op（AG / RS / AR / A2A / P2P）、张量、每卡线上字节、时间。调用方的总字节和总时间都由这些加出来 */
const xc = (pass, op, what, bytes, t) => ({ pass, op, what, bytes, t:t||0 });

const EP = {
  standard:{ name:Msg.of("engine_par.ep.name.standard"), short:"a2a", col:"s5", fused:false },
  ag:      { name:Msg.of("engine_par.ep.name.ag"), short:"AG-RS", col:"s1", fused:false },
  deepep:  { name:Msg.of("engine_par.ep.name.deepep"), short:"DeepEP", col:"s2", fused:false },
  distmoe: { name:Msg.of("engine_par.ep.name.distmoe"), short:"Dist-MoE", col:"s3", fused:true },
  mok:     { name:Msg.of("engine_par.ep.name.mok"), short:"MoK", col:"s4", fused:true } };
const CP = {
  ring:      { name:"Ring", col:"s1" },
  allgather: { name:"AllGather", col:"s2" },
  ulysses:   { name:"Ulysses", col:"s3" } };

/* ---------- CP：一个 attention 层、一篇文档 ----------
   x：S 文档长度、C、tp、m、d、hw、topo、g、ac（full 重算）、tCore（这一层这一篇的核心 F+B 时间）、kind：full / swa */
function cpLayer(meth, x){
  const { S, C, tp, m, d, hw, topo, g, ac, tCore, kind } = x, f=m.full;
  const o = { meth, wire:0, calls:0, tw:0, exposed:0, lk:lk(hw,topo,C,tp), ok:true, why:"", xs:[] };
  if(C<=1) return o;
  const l = o.lk, gqa = f.kind==="gqa";
  const kvTok = (gqa ? d.kvTokElems/Math.min(tp, f.Hkv) : d.kvTokElems) + (f.kind==="dsa" ? f.idxD : 0);   // 每 token 的 K+V（或 latent + indexer key）元素
  const Lb = 2*(S/C)*kvTok;                                     // 本卡一份 K + V 的字节
  const kvK = gqa ? "gqa" : f.kind==="dsa" ? "dsa" : "mla", kv = Msg.of("engine_par.cp.kv."+kvK);
  const split = det => { o.wire = det.reduce((s,x)=>s+x.bytes,0); for(const x of det) x.t = o.wire>0 ? o.tw*x.bytes/o.wire : 0; o.xs = det; };   // 每次调用按字节分摊总时间
  if(kind==="swa" && meth!=="ulysses"){                         // 滑窗：只和邻居交换最后 W 个 token
    const w1 = 2*Math.min(f.W, S/C)*kvTok, det = [xc("fwd","P2P",Msg.of("engine_par.cp.swa.fwd", { W: f.W }),w1)];
    if(ac) det.push(xc("rc","P2P",Msg.of("engine_par.cp.swa.rc"),w1));
    det.push(xc("bwd","P2P",Msg.of("engine_par.cp.swa.bwd"),w1), xc("bwd","P2P",Msg.of("engine_par.cp.swa.bwdGrad"),w1));
    o.calls = det.length; o.tw = wireT(det.length*w1, l, g.etaAg, o.calls, g); split(det); o.exposed = o.tw; return o;
  }
  if(meth==="allgather"){
    const chunks = gqa ? Math.min(f.Hkv/Math.min(tp,f.Hkv), 2) : 1, w1 = (C-1)*Lb, det = [xc("fwd","AG",Msg.of("engine_par.cp.ag.fwd", { kv }),w1)];
    if(ac) det.push(xc("rc","AG",Msg.of("engine_par.cp.ag.rc", { kv }),w1));
    det.push(xc("bwd","AG",Msg.of("engine_par.cp.ag.bwd", { kv }),w1), xc("bwd","RS",Msg.of("engine_par.cp.dkv."+kvK),w1));
    o.calls = (ac?8:6)*chunks; o.tw = wireT(det.length*w1, l, g.etaAg, o.calls, g); split(det);
  } else if(meth==="ring"){
    const det = [xc("fwd","P2P",Msg.of("engine_par.cp.ring.fwd", { kv, n: C-1 }),(C-1)*Lb)];
    if(ac) det.push(xc("rc","P2P",Msg.of("engine_par.cp.ring.rc", { kv }),(C-1)*Lb));
    det.push(xc("bwd","P2P",Msg.of("engine_par.cp.ring.bwd", { kv }),(C-1)*Lb), xc("bwd","P2P",Msg.of("engine_par.cp.ring.bwdGrad", { n: C }),2*C*Lb));
    o.calls = 2*(C-1)+C + (ac?C-1:0); o.tw = wireT(det.reduce((s,x)=>s+x.bytes,0), l, g.etaAg, o.calls, g); split(det);
  } else {
    const H = d.Hattn/tp;
    // 一次 a2a 发出 (C−1)/C：Q/K/V 序列 → 头，输出 O 头 → 序列。MLA 的 latent 不能按头切，按 C 份复制
    const uQKV = gqa ? H*f.dh + 2*Math.max(f.Hkv/Math.min(tp,f.Hkv), C)*f.dh : H*d.dqk + C*kvTok, uO = gqa ? H*f.dh : H*d.dv;
    const a = u => 2*(S/C)*u*(C-1)/C;
    const det = [xc("fwd","A2A",Msg.of("engine_par.cp.uly.qkv"),a(uQKV)), xc("fwd","A2A",Msg.of("engine_par.cp.uly.o"),a(uO))];
    if(ac) det.push(xc("rc","A2A",Msg.of("engine_par.cp.uly.rcQkv"),a(uQKV)), xc("rc","A2A",Msg.of("engine_par.cp.uly.rcO"),a(uO)));
    det.push(xc("bwd","A2A",Msg.of("engine_par.cp.uly.dO"),a(uO)), xc("bwd","A2A",Msg.of("engine_par.cp.uly.dQkv"),a(uQKV)));
    o.calls = ac?12:8; o.tw = wireT(det.reduce((s,x)=>s+x.bytes,0), l, g.etaN, o.calls, g); split(det);
    if(H % C){ o.ok=false; o.why = Msg.of("engine_par.cp.uly.why", { H, C }).invalid(); }
  }
  // Ring / AllGather 可以和分块计算重叠，只露出第一块和盖不住的部分；Ulysses 的两次 all-to-all 全部阻塞
  o.exposed = meth==="ulysses" ? o.tw : o.tw/C + Math.max(0, o.tw - tCore*(C-1)/C);
  return o;
}

/* ---------- EP：一层 MoE 的通信 + expert GEMM（F+B，含重算） ----------
   x：n 本卡 token、K、Z、E、De、stride、hw、topo、g、ac、prec、imb、tExpF / tExpB（这一层 expert 计算的前向 / 反向含重算）、syncUs、mokF */
function epLayer(be, x){
  const { n, K, Z, E, De, hw, topo, g, ac, imb } = x, tExpF=x.tExpF||0, tExpB=x.tExpB||0, tExp=tExpF+tExpB;
  const o = { be, ok:true, why:"", wire:0, ib:0, tw:0, tAlpha:0, tSync:0, t:tExp, extra:0, exch:0, syncs:0, buf:0, lk:lk(hw,topo,E,x.stride||1), parts:null, det:be!=="deepep", xs:[] };
  if(E<=1){ o.parts = { f:[["sm",Msg.of("engine_par.ep.seg.exp"),tExpF,"moe"]], b:[["sm",Msg.of("engine_par.ep.seg.bwdExp", { rc: "" }),tExpB,"moe"]] }; o.buf = n*K*imb*De*2; return o; }
  const l = o.lk, mx = /^mx|fp8/.test(x.prec||"");
  const bd = (mx && (be==="deepep"||be==="distmoe"||be==="mok")) ? 1+1/32 : 2;          // MXFP8 量化融合进 dispatch
  const slots = n*K*imb*(1-1/E), dE = (E-1)*pDest(Z,E,K);
  let disp, comb;
  if(be==="ag"){ disp=(E-1)*n*De*bd; comb=(E-1)*n*De*2; }
  else if(be==="deepep"){ disp=n*imb*dE*De*bd; comb=n*imb*dE*De*2; }
  else { disp=slots*De*bd; comb=slots*De*2; }
  const dispB = disp/bd*2, combB = comb;
  const replay = ac && (be==="standard"||be==="ag");                                      // 重算时 NCCL 路径要重放 dispatch / combine
  o.exch = 4 + (replay?2:0);
  o.wire = disp + comb + dispB + combB + (replay ? disp+comb : 0);
  const effN = be==="ag" ? g.etaAg : g.etaN, dom = pow2Dom(topo);
  /* 线上时间：跨域时 NCCL a2a 按均匀路由的远端比例；DeepEP 每个远端节点只发一次再 NVLink 扇出；AG 环全程按最慢链路 */
  const tWire = w => {
    if(l.intra) return w/(hw.nvl*1e9*effN);
    if(be==="ag"){ o.ib += w; return w/(hw.ib*1e9*g.etaAg); }
    if(be==="deepep"){ const nodes=E/dom, rn=(nodes-1)*(1-binomRatio(Z, Z/nodes, K)); const ibw = w*rn/Math.max(dE,1e-9); o.ib += ibw; return Math.max(ibw/(hw.ib*1e9*g.etaN), w/(hw.nvl*1e9*effN)); }
    const rf=(E-dom)/(E-1); o.ib += w*rf; return Math.max(w*rf/(hw.ib*1e9*g.etaN), w*(1-rf)/(hw.nvl*1e9*effN));
  };
  const a = (l.intra ? g.alphaNv : g.alphaIb)*1e-6;
  const callsPer = be==="standard"||be==="ag" ? 2 : be==="deepep" ? 1 : 0;
  const aOne = be==="deepep" ? g.deepepUs*1e-6 + (l.intra?0:g.alphaIb*1e-6) : a;
  o.tAlpha = o.exch*callsPer*aOne + (be==="distmoe" ? 4*g.barrierUs*1e-6 + 2e-5 : 0);
  o.syncs = be==="standard" ? (replay?2:1) : be==="deepep" ? 1 : 0;
  o.tSync = o.syncs*(x.syncUs||0)*1e-6;
  const tD = tWire(disp), tC = tWire(comb), tDB = tWire(dispB), tCB = tWire(combB);
  o.tw = tD + tC + tDB + tCB + (replay ? tD + tC : 0);
  const al = o.tAlpha/o.exch||0, lane = l.intra ? "nvl" : "ib";
  {  // 逐次调用：AG-RS 后端是 all-gather / reduce-scatter，其余是点对点的 all-to-all（融合后端在 GEMM kernel 里发）
    const [oD,oC] = be==="ag" ? ["AG","RS"] : ["A2A","A2A"], fz = EP[be].fused ? Msg.of("engine_par.ep.call.fused") : "", q = bd<2 ? Msg.of("engine_par.ep.call.mx") : "";
    o.xs = [xc("fwd",oD,Msg.of("engine_par.ep.call.disp", { mx: q, fused: fz }),disp,tD+al), xc("fwd",oC,Msg.of("engine_par.ep.call.comb", { fused: fz }),comb,tC+al)];
    if(replay) o.xs.push(xc("rc",oD,Msg.of("engine_par.ep.call.rcDisp", { mx: q }),disp,tD+al), xc("rc",oC,Msg.of("engine_par.ep.call.rcComb"),comb,tC+al));
    o.xs.push(xc("bwd",oD==="AG"?"AG":"A2A",Msg.of("engine_par.ep.call.bwdComb", { fused: fz }),combB,tCB+al), xc("bwd",oD==="AG"?"RS":"A2A",Msg.of("engine_par.ep.call.bwdDisp", { fused: fz }),dispB,tDB+al));
  }
  if(!EP[be].fused){
    o.t = tExp + o.tw + o.tAlpha + o.tSync;
    const nm = be==="ag" ? [Msg.of("engine_par.ep.seg.agDisp"),Msg.of("engine_par.ep.seg.agComb")] : be==="deepep" ? [Msg.of("engine_par.ep.seg.deepepDisp"),Msg.of("engine_par.ep.seg.deepepComb")] : [Msg.of("engine_par.ep.seg.a2aDisp"),Msg.of("engine_par.ep.seg.a2aComb")];
    const f = [];
    if(o.syncs) f.push(["cpu", be==="deepep"?Msg.of("engine_par.ep.seg.syncDeepep"):Msg.of("engine_par.ep.seg.syncD2h"), o.tSync/o.syncs, "sync"]);
    f.push([lane, nm[0], tD+al, "ep"], ["sm",Msg.of("engine_par.ep.seg.exp"),tExpF,"moe"], [lane, nm[1], tC+al, "ep"]);
    const b = [];
    if(replay) b.push([lane,Msg.of("engine_par.ep.seg.rcDisp"),tD+al,"ep"], [lane,Msg.of("engine_par.ep.seg.rcComb"),tC+al,"ep"]);
    if(replay && o.syncs>1) b.unshift(["cpu",Msg.of("engine_par.ep.seg.rcSync"),o.tSync/o.syncs,"sync"]);
    b.push([lane,Msg.of("engine_par.ep.seg.bwdComb"),tCB+al,"ep"], ["sm",Msg.of("engine_par.ep.seg.bwdExp", { rc: ac ? Msg.of("engine_par.ep.seg.withRc") : "" }),tExpB,"moe"], [lane,Msg.of("engine_par.ep.seg.bwdDisp"),tDB+al,"ep"]);
    o.parts = { f, b };
  } else if(be==="distmoe"){
    o.t = Math.max(tExp/0.92, o.tw) + o.tAlpha;
    const sF = Math.max(tExpF/0.92, tD+tC), sB = Math.max(tExpB/0.92, tDB+tCB), bar = g.barrierUs*1e-6;
    o.parts = { f:[["sm",Msg.of("engine_par.ep.dist.fBarrier"),bar+1e-5,"sync"],["sm",Msg.of("engine_par.ep.dist.fGemm"),sF,"moe"],["nvl",Msg.of("engine_par.ep.dist.fWire"),tD+tC,"ep",true],["sm",Msg.of("engine_par.ep.dist.fSum"),bar,"sync"]],
                b:[["sm",Msg.of("engine_par.ep.dist.bBarrier"),bar+1e-5,"sync"],["sm",Msg.of("engine_par.ep.dist.bGemm"),sB,"moe"],["nvl",Msg.of("engine_par.ep.dist.bWire"),tDB+tCB,"ep",true],["sm",Msg.of("engine_par.ep.dist.bSum"),bar,"sync"]] };
    if(!l.intra){ o.ok=false; o.why=Msg.of("engine_par.ep.dist.whyDomain").invalid(); }
    if(hw.sm<100){ o.ok=false; o.why=Msg.of("engine_par.ep.dist.whySm").invalid(); }
  } else {
    const fr = Math.min(0.6, Math.max(0.05, x.mokF||0.25)), wEff = Math.min(1, fr/0.3);   // 通信 SM 比例 f：f ≥ 0.3 时打满 NVLink
    o.t = Math.max(tExp/(1-fr), o.tw/wEff) + 0.03*tExp;
    const sF = Math.max(tExpF/(1-fr), (tD+tC)/wEff), sB = Math.max(tExpB/(1-fr), (tDB+tCB)/wEff);
    o.parts = { f:[["sm",Msg.of("engine_par.ep.mok.fGemm"),sF,"moe"],["nvl",Msg.of("engine_par.ep.mok.fWire"),(tD+tC)/wEff,"ep",true]],
                b:[["sm",Msg.of("engine_par.ep.mok.bGemm"),sB,"moe"],["nvl",Msg.of("engine_par.ep.mok.bWire"),(tDB+tCB)/wEff,"ep",true]] };
    if(topo.domain<64 || !hw.grace){ o.ok=false; o.why=Msg.of("engine_par.ep.mok.whyHw").invalid(); }
    else if(!l.intra){ o.ok=false; o.why=Msg.of("engine_par.ep.mok.whyDomain").invalid(); }
  }
  o.extra = o.t - tExp;                                            // 关键路径上比纯 GEMM 多出来的部分（含 host sync）
  o.buf = be==="standard" ? 2*n*K*imb*De*2 : be==="ag" ? 2*E*n*De*2 : be==="deepep" ? n*K*imb*De*2 + 1e9 : be==="distmoe" ? 2*n*K*De*2 + n*K*imb*De*2 : 0.4e9;
  return o;
}

/* ---------- FSDP：一层参数 P 在 F 张卡上切分 ---------- */
const fsdpBytes = (P, F, aG, gB) => F>1 ? P*(F-1)/F*(aG*2+gB) : 0;
/* 一层的三次（reshard 关掉时两次）：前向 gather BF16 参数、反向再 gather、反向末尾 reduce-scatter 梯度 */
const fsdpDet = (P, F, l, g, aG, gB) => { if(F<=1||P<=0) return [];
  const ag = P*(F-1)/F*2, rs = P*(F-1)/F*gB, wt = b => wireT(b, l, g.etaAg, 1, g);
  return [xc("fwd","AG",Msg.of("engine_par.fsdp.fwd"),ag,wt(ag))].concat(aG===2?[xc("bwd","AG",Msg.of("engine_par.fsdp.bwdAg"),ag,wt(ag))]:[], [xc("bwd","RS",Msg.of("engine_par.fsdp.grad", { dt: gB===4?"FP32":"BF16" }),rs,wt(rs))]); };
const fsdpT = (P, F, l, g, aG, gB) => fsdpDet(P,F,l,g,aG,gB).reduce((s,x)=>s+x.t,0);

return { EP, CP, xc, pDest, binomRatio, pow2Dom, lk, wireT, cpLayer, epLayer, fsdpBytes, fsdpDet, fsdpT };
})();
