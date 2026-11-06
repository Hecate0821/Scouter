/* 集群网络：把机内互联（NVSwitch / PCIe）和 scale-out（网卡、叶交换机、收敛比、路由）的配置折算成 hw / topo，
   再交给训练和推理的同一套模型算（Train.run、Infer.decode / prefill、Common.link / agT / arT / a2aT）。
   默认配置就是主页面的硬件和拓扑：不改任何一项时，这里的每个数都和其他页一致（tests/engine.js 检查）。
   这里只做两件事：配置 → hw / topo；对同一个模型做反事实（把网卡或 NVLink 换成无限快）拆出时间花在哪。 */
const Fabric = (function(){
const { link, agT, arT, a2aT, spineX, ecmpMax, derive } = Common;

const PCIE = Common.freeze({ 4:32, 5:64, 6:128 });     // x16 每方向 GB/s（规范速率；编码和协议开销进 η）
const DOMAINS = Common.freeze([8, 16, 32, 64, 72, 128, 256, 512]);
const SCALES = Common.freeze([1, 2, 4, 8, 16, 32, 64]);
/* 配置项。0 = 跟着硬件走：domain 取主页面拓扑的 NVLink 域；pcie 在网卡超过 Gen5 x16 时取 Gen6（ConnectX-8 的 800G 要 Gen6 x16）；
   nic 取硬件表的每卡网卡。ports 是叶交换机的下行口：51.2 Tb/s（双向）的 400G 交换机 64 口，一半向下是 32 */
const DEF = Common.freeze({ scale:1, domain:0, intra:"nvswitch", nvlFrac:1, nvls:false, gdr:true, pcie:0, nic:0, ports:32, over:1, route:"adaptive", qps:1, sharp:false });
const ENUMS = Common.freeze({ intra:["nvswitch","pcie"], route:["adaptive","ecmp"], pcie:[0,4,5,6], scale:SCALES, domain:[0].concat(DOMAINS),
  nvlFrac:[1, 17/18, 0.75, 0.5], nic:[0,200,400,800,1600], ports:[16,32,64,128], over:[1,2,4,8,15], qps:[1,2,4,8,16] });
const nodeOf = hw => hw.grace ? 4 : 8;                  // 一个计算节点（托盘）的 GPU 数，也是网卡数 = rail 数

const NUM = ["scale","domain","nvlFrac","pcie","nic","ports","over","qps"];
function resolve(hw, topo, net){
  const x = Object.assign({}, DEF, net||{});
  for(const k of NUM) x[k] = +x[k] || 0;                 // 界面的下拉框给的是字符串
  for(const k of ["nvls","gdr","sharp"]) x[k] = x[k]===true || x[k]==="true";
  x.scale = x.scale || 1; x.nvlFrac = x.nvlFrac || 1; x.over = x.over || 1; x.qps = x.qps || 1;
  x.nic = x.nic || hw.ib*8;
  x.pcie = x.pcie || (x.nic/8 > PCIE[5] ? 6 : 5);
  x.ports = x.ports || 32;
  if(topo.pcieOnly) x.intra = "pcie";                   // 没有 NVLink 的卡（RTX PRO 6000）只能走 PCIe
  x.node = nodeOf(hw);
  x.domain = x.intra==="pcie" ? x.node : (x.domain || topo.domain);
  x.N = topo.gpus*x.scale;
  x.segment = x.ports*x.node;                           // 一层叶交换机（每个 rail 一台）能连满的 GPU 数
  return x;
}
/* 配置 → 模型用的 hw / topo。理想下限（g.lb）按自适应路由：哈希碰撞是调度造成的，不是硬件上限 */
function apply(hw, topo, net, g){
  const x = resolve(hw, topo, net), pcie = PCIE[x.pcie];
  const nvl = x.intra==="pcie" ? pcie : hw.nvl*x.nvlFrac;
  const ib = Math.min(x.nic/8, x.gdr ? pcie : hw.host);  // GPUDirect RDMA：网卡直接读写显存，受 PCIe x16 限制；关掉时经主存中转，受主机链路限制
  const hw2 = Object.assign({}, hw, { nvl, ib });
  const topo2 = Object.assign({}, topo, { gpus:x.N, domain:x.domain, pcieOnly:x.intra==="pcie",
    net:{ segment:x.segment, ports:x.ports, rails:x.node, over:x.over, ecmp:x.route==="ecmp" && !(g&&g.lb), qps:x.qps, nvls:x.nvls && x.intra!=="pcie", sharp:x.sharp } });
  return { hw:hw2, topo:topo2, x };
}
/* 哪些配置项在当前配置下不起作用（界面把它们标灰，tests/fabric.js 检查标了的确实不改任何结果）。
   inf：推理实例的卡数（训练的卡数是 x.N）。理由的文字在 shell.inert.<why> */
function inert(hw, topo, net, g, inf){
  const x = resolve(hw, topo, net), o = {}, put = (why, ...ks) => { for(const k of ks) if(!o[k]) o[k] = why; };
  const span = Math.max(x.N, (inf&&inf.N)||1);
  if(topo.pcieOnly) put("pcieCard", "intra");
  if(x.intra==="pcie") put("pcieNoNvl", "domain", "nvlFrac", "nvls");
  if(span<=x.domain) put("oneDomain", "ports", "over", "route", "qps", "sharp");   // 网卡和 GPUDirect 还管 PD 分离和权重同步的传输
  if(span<=x.segment) put("oneSegment", "over", "route", "qps");
  if(x.route!=="ecmp" || (g&&g.lb)) put(g&&g.lb ? "ideal" : "notEcmp", "qps");
  return o;
}
/* 当前训练配置的每个并行组：几张卡、间隔多少个 rank、走哪条链路，以及环 / 点对点 / all-to-all 各拿到多少带宽（每卡每方向） */
function groups(m, hw, topo, c, net, g){
  const a = apply(hw, topo, net, g), r = Train.run(m, a.hw, a.topo, trnCfg(c, a.x), g), mg = c.fw==="megatron", fs = r.par.fsdp;
  const L = [["tp", r.tp, 1], ["cp", r.cp, r.tp], ["ep", r.ep, mg ? r.etp : 1], ["pp", r.pp>1 ? 2 : 1, r.tp*r.cp*r.dp]];
  if(mg) L.push(["dp", r.dp*r.cp, r.tp], ["edp", r.edp, r.etp*r.ep]);
  else if(fs){ L.push(["fsdp", fs.fs, r.tp]); if(m.moe.E) L.push(["efsdp", fs.fe, r.ep]); if(fs.R>1) L.push(["hsdp", fs.R, fs.fs*r.tp]); }
  return L.filter(([,n]) => n>1).map(([dim, n, stride]) => { const l = link(a.hw, a.topo, n, stride);
    return { dim, n, stride, intra:l.intra, d:l.d, m:l.m, bw:l.bw, p2p:l.p2p, a2a:l.a2a, inNet:!!l.inNet }; });
}
/* 推理实例可以比训练规模大：topo 的卡数至少放得下这个实例 */
const infTopo = (topo, c) => Object.assign({}, topo, { gpus:Math.max(topo.gpus, c.N||1) });
/* 训练按弱扩展：卡数乘 scale，全局 batch 也乘 scale，每卡的活不变 */
const trnCfg = (c, x) => Object.assign({}, c, { N:x.N, Gtok:c.Gtok*x.scale });

const RUN = {
  train:  (m, hw, topo, c, g, x) => Train.run(m, hw, topo, trnCfg(c, x), g).stepT,
  decode: (m, hw, topo, c, g) => Infer.decode(m, hw, infTopo(topo, c), Object.assign({}, c, { phase:"decode" }), g).stepT,
  prefill:(m, hw, topo, c, g) => Infer.prefill(m, hw, infTopo(topo, c), Object.assign({}, c, { phase:"prefill" }), g).ttft };

function train(m, hw, topo, c, net, g){ const a = apply(hw, topo, net, g); return Object.assign(Train.run(m, a.hw, a.topo, trnCfg(c, a.x), g), { fab:a }); }
function decode(m, hw, topo, c, net, g){ const a = apply(hw, topo, net, g); return Object.assign(Infer.decode(m, a.hw, infTopo(a.topo, c), Object.assign({}, c, { phase:"decode" }), g), { fab:a }); }
function prefill(m, hw, topo, c, net, g){ const a = apply(hw, topo, net, g); return Object.assign(Infer.prefill(m, a.hw, infTopo(a.topo, c), Object.assign({}, c, { phase:"prefill" }), g), { fab:a }); }

/* 时间花在哪（反事实）：网卡换成无限快、零延迟少掉的时间记给网卡；再把 NVLink 也换掉，少掉的记给 NVLink；剩下的是计算和其他。
   和逐项加通信时间不同，这样算的是“去掉它能省多少”，已经扣掉了能藏在计算后面的部分 */
const FAST = 1e30;
function split(kind, m, hw, topo, c, net, g){
  const a = apply(hw, topo, net, g), run = RUN[kind];
  const t = run(m, a.hw, a.topo, c, g, a.x);
  const hwN = Object.assign({}, a.hw, { ib:FAST }), gN = Object.assign({}, g, { alphaIb:0 });
  const tN = run(m, hwN, a.topo, c, gN, a.x);
  const hwV = Object.assign({}, hwN, { nvl:FAST }), gV = Object.assign({}, gN, { alphaNv:0 });
  const tV = run(m, hwV, a.topo, c, gV, a.x);
  return { t, nic:Math.max(0, t-tN), nvl:Math.max(0, tN-tV), rest:tV };
}
/* 带宽还是延迟：每一项单独改，看时间变多少（相对）。延迟一项把每次通信调用的固定开销一起改：
   NCCL 的 α（NVLink / 网卡）、DeepEP 低延迟 all-to-all 和普通模式的每次调用开销、Dist-MoE 的 barrier */
const LAT = Common.freeze(["alphaNv","alphaIb","a2aLLus","deepepUs","barrierUs"]);
const lat = f => LAT.reduce((o,k) => (o[k]=f, o), {});
const KNOBS = Common.freeze([
  { key:"nicBw2", hw:{ ib:2 } }, { key:"nicBwHalf", hw:{ ib:0.5 } }, { key:"nvlBw2", hw:{ nvl:2 } }, { key:"nvlBwHalf", hw:{ nvl:0.5 } },
  { key:"latHalf", g:lat(0.5) }, { key:"lat2", g:lat(2) } ]);
function sens(kind, m, hw, topo, c, net, g){
  const a = apply(hw, topo, net, g), run = RUN[kind], t0 = run(m, a.hw, a.topo, c, g, a.x);
  const mul = (o, f) => { const r = Object.assign({}, o); for(const k in f||{}) r[k] = o[k]*f[k]; return r; };
  return { t0, rows:KNOBS.map(k => { const t = run(m, mul(a.hw, k.hw), a.topo, c, mul(g, k.g), a.x); return { key:k.key, t, d:t/t0-1 }; }) };
}
/* NVLink 域多大：卡数不变，只改域的大小。训练每条线是一个 EP 度（当前值和它的 2、4、8 倍里跑得起来的），
   推理每条线是一个宽 EP 实例（DP attention + DeepEP，N = 8…64）；dense 模型没有推理这张图 */
function domainSweep(kind, m, hw, topo, c, net, g){
  const x0 = resolve(hw, topo, net), E = m.moe.E||0;
  const lines = kind==="train"
    ? (E ? [1,2,4,8].map(k => c.ep*k).filter(ep => ep<=x0.N && E%ep===0) : [c.ep]).map(ep => ({ key:ep, c:Object.assign({}, c, { ep }) }))
    : (E ? [8,16,32,64].map(N => ({ key:N, c:Object.assign({}, c, { N, dpA:N, moePar:"deepep" }) })) : []);
  const top = kind==="train" ? x0.N : 64;
  const doms = DOMAINS.filter(D => D<=top && (D!==72 || hw.grace));
  return lines.map(L => ({ key:L.key, pts:doms.map(D => {
    const a = apply(hw, topo, Object.assign({}, net, { domain:D, intra:"nvswitch" }), g);
    const r = kind==="train" ? Train.run(m, a.hw, a.topo, trnCfg(L.c, a.x), g) : Infer.decode(m, a.hw, infTopo(a.topo, L.c), Object.assign({}, L.c, { phase:"decode" }), g);
    return { domain:D, t:kind==="train" ? r.stepT : r.stepT, ok:!r.warn.some(w => w && w.bad) }; }).filter(p => p.ok) })).filter(L => L.pts.length);
}
/* PCIe 机器上的 TP：同一个推理配置，TP 从 1 到一台机器的卡数，NVSwitch 和 PCIe 各算一遍 */
function tpPcie(m, hw, topo, c, net, g){
  const node = nodeOf(hw), out = [];
  for(let tp=1; tp<=node; tp*=2){
    const ci = Object.assign({}, c, { N:tp, dpA:1, moePar: m.moe.E ? "tp" : c.moePar });
    const row = { tp };
    for(const intra of ["nvswitch","pcie"]){ const a = apply(hw, topo, Object.assign({}, net, { intra }), g); row[intra] = RUN.decode(m, a.hw, a.topo, ci, g); }
    out.push(row);
  }
  return out;
}

/* 单个集合通信（不带模型）：n 张卡、每卡 bytes，各种算法的时间。NVSwitch 一节和 scale-out 一节都用它 */
function coll(op, bytes, n, hw, topo, net, g, algo){
  const a = apply(hw, topo, net, g), lk = link(a.hw, a.topo, n, 1);
  if(op==="ag") return agT(bytes, n, lk, g);
  if(op==="a2a") return a2aT(bytes, n, lk, g);
  if(op==="p2p") return n<=1 ? 0 : (lk.intra?g.alphaNv:g.alphaIb)*1e-6 + bytes/(lk.p2p*g.etaN);
  if(algo==="nvls") return arT(bytes, n, Object.assign({}, lk, { inNet:true }), g);
  return arT(bytes, n, Object.assign({}, lk, { inNet:false }), g, algo);
}
const MSGS = Common.freeze([4e3, 16e3, 64e3, 256e3, 1e6, 4e6, 16e6, 64e6, 256e6, 1e9]);
/* NVLink 域内 all-reduce 随消息大小：ring / one-shot / two-shot / NVLS */
function arSweep(hw, topo, net, g){
  const x = resolve(hw, topo, net), n = Math.min(x.domain, x.N, 64);
  return { n, rows:MSGS.map(b => ({ bytes:b, ring:coll("ar",b,n,hw,topo,net,g,"ring"), oneshot:coll("ar",b,n,hw,topo,net,g,"oneshot"), twoshot:coll("ar",b,n,hw,topo,net,g,"twoshot"), nvls:coll("ar",b,n,hw,topo,net,g,"nvls") })) };
}

/* 带宽阶梯：每卡每方向 GB/s */
function ladder(hw, topo, net, g){
  const a = apply(hw, topo, net, g), x = a.x, N = x.N;
  const rows = [ { key:"hbm", GBs:hw.hbm }, { key:x.intra==="pcie" ? "p2pPcie" : "nvl", GBs:a.hw.nvl }, { key:"host", GBs:hw.host },
    { key:"pcie", GBs:PCIE[x.pcie], gen:x.pcie }, { key:"nicRaw", GBs:x.nic/8 }, { key:"nic", GBs:a.hw.ib } ];
  if(N > x.domain){ const l = link(a.hw, a.topo, N, 1); rows.push({ key:"ring", GBs:l.bw/1e9, d:l.d }, { key:"a2a", GBs:l.a2a/1e9, n:N }); }
  if(N > x.segment) rows.push({ key:"spine", GBs:a.hw.ib/x.over, over:x.over });
  return rows;
}
/* PCIe 代数和网卡：x16 每方向能不能装下这张网卡（GPUDirect RDMA 时网卡直接读写显存，走的就是这条 x16） */
function pcieFit(hw, topo, net){
  const x = resolve(hw, topo, net);
  return Object.keys(PCIE).map(Number).map(gen => ({ gen, GBs:PCIE[gen], cur:gen===x.pcie,
    fits:ENUMS.nic.filter(v => v>0).map(nic => ({ nic, ok:nic/8<=PCIE[gen] })) }));
}
/* 叶 / spine 两层：每个 rail 一台叶交换机，segment = 下行口 × rail 数 */
function scaleOut(hw, topo, net, g){
  const x = resolve(hw, topo, net), segs = Math.ceil(x.N/x.segment), up = Math.max(1, Math.round(x.ports/x.over));
  const leaves = segs*x.node, nodes = Math.ceil(x.N/x.node);
  return { N:x.N, node:x.node, nodes, segment:x.segment, segs, leaves, up, spinePorts: segs>1 ? leaves*up : 0, nic:x.nic, over:x.over,
    crossGBs: Math.min(x.nic/8, x.gdr ? PCIE[x.pcie] : hw.host)/(segs>1 ? x.over : 1) };
}
/* 跨 segment 的三种流量能拿到网卡带宽的几成：横轴收敛比，自适应路由和 ECMP（按当前 QP 数）各一条 */
function overSweep(hw, topo, net, g){
  const x = resolve(hw, topo, net), N = Math.max(x.N, 2*x.segment);       // 当前规模不到两个 segment 时按两个算（否则没有跨 segment 的流量）
  const big = Object.assign({}, topo, { gpus:N });
  return ENUMS.over.map(over => {
    const row = { over };
    for(const route of ["adaptive","ecmp"]){
      const a = apply(hw, big, Object.assign({}, net, { over, route, scale:1 }), g), l = link(a.hw, a.topo, N, 1), l2 = link(a.hw, a.topo, 2, x.segment);
      const ringN = Math.min(l.d*a.hw.ib*1e9, a.hw.nvl*1e9);
      row[route] = { ring:l.bw/ringN, p2p:l2.p2p/(a.hw.ib*1e9), a2a:l.a2a/link(a.hw, Object.assign({}, a.topo, { net:{} }), N, 1).a2a };
    }
    return row;
  });
}
/* ECMP 和 QP 数：一个叶交换机下所有网卡都往别的 segment 发（流水线相邻两段在不同 segment 时就是这样），拿到几成带宽 */
function qpSweep(hw, topo, net, g){
  const x = resolve(hw, topo, net), U = Math.max(1, Math.round(x.ports/x.over));
  return ENUMS.qps.concat([32, 64]).map(q => { const F = x.ports*q, mx = ecmpMax(F, U);
    return { qps:q, eff:(F/U)/mx, flows:F, up:U }; });
}

/* 不是集合通信的大块传输：PD 分离时把 prompt 的 KV 从 prefill 实例搬到 decode 实例；RL 每轮把新权重推给推理引擎 */
function transfers(m, hw, topo, cInf, net, g){
  const a = apply(hw, topo, net, g), c = Object.assign({}, cInf, { phase:"decode", B:1, n:cInf.P||cInf.n });
  const r = Infer.step(m, a.hw, infTopo(a.topo, c), c, g), mem = Infer.memory(m, a.hw, c, r);
  const kv = mem.perSeq, nic = a.hw.ib*1e9*g.etaN, nvl = a.hw.nvl*1e9*g.etaN, w = derive(m).total*2, Ninf = Math.max(1, cInf.N||1);
  const pf = Infer.prefill(m, a.hw, infTopo(a.topo, cInf), Object.assign({}, cInf, { phase:"prefill" }), g);
  return { P:c.n, kv, kvNic:kv/nic + g.alphaIb*1e-6, kvNvl:kv/nvl + g.alphaNv*1e-6, ttft:pf.ttft,
    w, wOne:w/nic, wShard:w/(Ninf*nic), Ninf };
}

/* 公开实测的对照（集群页"模型对实测"一节）：一行实测写成 spec，这里按同一套 link / arT / a2aT 算模型的数，不另调参数。
   spec：hw（HW 的 key）、op（ar = all-reduce、a2a = 每卡发 bytes 的 all-to-all、p2p = 两卡之间一次 RDMA 写）、bytes、n（卡数）、
   domain（一个 NVLink 域 / 分区几张卡）、nic（每卡网卡 Gb/s，缺省按硬件表）、nvls / sharp（交换机里做归约）、
   metric：busbw（nccl-tests 的 bus bandwidth，GB/s）、us（一次的时间）、gbps（点对点吞吐，Gb/s）。
   跨机按每个 segment 无限大、不收敛（公开的数字都没写 spine 收敛比） */
function bench(spec, g){
  const hw = Object.assign({}, Common.HW[spec.hw], spec.nic ? { ib:spec.nic/8 } : {});
  const n = spec.n, B = spec.bytes;
  const topo = { gpus:n, domain:spec.domain, net:{ nvls:!!spec.nvls, sharp:!!spec.sharp } };
  const lk = link(hw, topo, n, 1);
  const t = spec.op==="ar" ? arT(B, n, lk, g)
    : spec.op==="a2a" ? a2aT(B, n, lk, g)
    : (lk.intra ? g.alphaNv : g.alphaIb)*1e-6 + B/(lk.p2p*g.etaN);
  const v = spec.metric==="busbw" ? B/t*(spec.op==="ar" ? 2*(n-1)/n : (n-1)/n)/1e9 : spec.metric==="gbps" ? B*8/t/1e9 : t*1e6;
  return { model:v, t, intra:lk.intra, linkGBs:(spec.op==="p2p" ? lk.p2p : spec.op==="a2a" ? lk.a2a : lk.bw)/1e9, ratio:spec.meas>0 ? v/spec.meas : null };
}

return { PCIE, DOMAINS, SCALES, DEF, ENUMS, KNOBS, LAT, MSGS, nodeOf, resolve, inert, groups, apply, train, decode, prefill, split, sens, domainSweep, tpPcie, coll, arSweep, ladder, pcieFit, scaleOut, overSweep, qpSweep, transfers, bench };
})();
