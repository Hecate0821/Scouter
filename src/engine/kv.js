/* KV cache：一个请求、一个 agent 会话在每种层上要存多少字节；前缀缓存在多轮里能命中多少；分层存储（HBM → 主机内存 → 远端池）
   命中后 TTFT 是多少。所有字节都按每张卡（一个 attention rank）算，和 Infer.memory 的 KV 池同一口径。

   会话：sys 个 token 的共享前缀（系统提示 + 工具定义，所有会话共用一份），每轮追加 turnIn 个新 token（工具结果 / 用户消息）、
   生成 turnOut 个，一共 turns 轮；sessions 个会话同时在跑（一个副本上）。第 t 轮的输入 = 上一轮结束时的整段上下文 + turnIn。
   命中：上一轮结束时的上下文还在缓存里，就只算 turnIn 个新 token 的 prefill（attention 仍要看整个前缀）；
   不在就从共享前缀之后整段重算。

   驻留比例：稳态下的工作集 = sessions × 一个会话在两轮之间要留下的字节（按轮平均）。会话回来的时间随机（工具调用时长不一）时，
   LRU 的 Che 近似给出命中概率 = 容量 / 工作集（每个会话一样大时正好如此）；回来的间隔很规律时 LRU 会整体抖动（工作集一超过容量
   几乎全不中），按会话保留（优先级、pin）才能退回到容量 / 工作集。这里取前者。

   每种层留下的东西（引擎都只按"下一轮能用上"来留）：
   - 全注意力 / MLA / DSA：每个 token 一份 KV（DSA 再加 indexer 的 key），前缀多长就存多长；
   - 滑窗层：下一轮只需要前缀最后 W 个 token，窗口外的页可以释放（SGLang UnifiedRadixCache 的 SWA 组件、vLLM SlidingWindowManager、
     TRT-LLM VSWA 的 per_request / per_conversation）；
   - 线性注意力：没有逐 token 的 KV，只有固定大小的状态；要命中，就得在命中位置存过一份状态快照。 */
const KV = (function(){

/* 各引擎的前缀缓存画像（现实建模用；理想下限按 1 个 token 的粒度、装回全部重叠）
   page：一页多少 token（前缀按页匹配，最后一页平均空一半）；
   tail：在提示结尾 / 请求结束处额外发布页内的部分页，下一轮能精确命中到上一轮结束的位置（TokenSpeed）；
   linBlock：线性注意力状态只存在 block 边界上（vLLM 的 align 模式），混合模型的 block 要放大到一个注意力页不小于一个状态页；
   hostOverlap：从主机内存装回和 prefill 逐层重叠（SGLang HiCache、TokenSpeed L2），否则先装完再算（vLLM OffloadingConnector 的异步拷贝）。
   出处：SGLang --page-size 默认 1、--mamba-radix-cache-strategy auto = no_buffer（只留序列结尾的状态）；vLLM block_size 默认 16，
   docs/design/hybrid_kv_cache_manager.md；TokenSpeed docs/design/cache-concepts.md、kernel_page_sizes.py。 */
const ENGINES = {
  sglang:     { page:1,  tail:false, linBlock:false, hostOverlap:true },
  vllm:       { page:16, tail:false, linBlock:true,  hostOverlap:false },
  tokenspeed: { page:64, tail:true,  linBlock:false, hostOverlap:true },
  titan:      { page:1,  tail:false, linBlock:false, hostOverlap:true }
};

/* 默认的 agent 会话和缓存层级。hostGB / remoteGB 是每张卡分给 KV 的主机内存 / 远端池容量；inclusive = 下层包含上层
   （SGLang HiCache、TokenSpeed L2 默认 write_through） */
const BASE = Common.freeze({ sys:8192, turnIn:2048, turnOut:512, turns:40, sessions:64, hostGB:200, remoteGB:0, inclusive:true,
  replicas:1, routing:"aware" });

/* 每张卡上：每个 token 的 KV、滑窗层每个 token、一份线性注意力状态快照（字节） */
function perRank(m, hw, topo, c, g){
  const r = Infer.step(m, hw, topo, Object.assign({}, c, { phase:"decode", B:1, n:1024, specK:0 }), g);
  const f = m.full, l = m.lin, nSwa = f.n>0 ? (f.swa||0) : 0;
  return { r, full:f.n>0 ? (f.n-nSwa)*(r.kvTokRank+(r.idxTok||0)) : 0, swa:nSwa*r.kvTokRank, W:nSwa ? f.W : 0, snap:l.n>0 ? l.n*(r.Sl+r.Cl) : 0,
    nFull:f.n-nSwa, nSwa, nLin:l.n, kvTokRank:r.kvTokRank };
}

/* 一个请求在上下文 L 时缓存里的字节（每张卡）：全注意力按 L、滑窗按 min(L, W)、线性注意力一份快照 */
const bytesAt = (p, L) => p.full*L + p.swa*(p.W ? Math.min(L, p.W) : L) + p.snap;

/* prefill：从位置 p0 开始算 P 个新 token，按 chunk C 切（attention 看整个前缀）；返回总时间 */
function prefillFrom(m, hw, topo, c, g, p0, P){
  let t = 0;
  for(let x=0; x<P; x+=c.C){ const cc=Math.min(c.C, P-x); t += Infer.step(m, hw, topo, Object.assign({}, c, { phase:"prefill", C:cc, p:p0+x, reqPerChunk:1 }), g).stepT; }
  return t;
}

/* 整页：每种层的字节、会话工作集、各层级的驻留比例、每轮 TTFT */
function plan(m, hw, topo, c, g){
  const lb = !!g.lb, E = ENGINES[c.engine] || ENGINES.titan, warn = [];
  const p = perRank(m, hw, topo, c, g), mem = Infer.memory(m, hw, Object.assign({}, c, { B:1 }), p.r);
  // blk：物理块；gran：下一轮能命中的位置粒度。vLLM 混合模型把块放大到一个注意力页（blk × 每层每 token 的 KV）不小于一个状态页，取 16 的倍数
  const linBlk = p.nLin && p.kvTokRank>0 ? Math.ceil(p.snap/p.nLin/p.kvTokRank/16)*16 : 0;
  const blk = lb ? 1 : E.linBlock && linBlk ? Math.max(E.page, linBlk) : E.page;
  const gran = lb || E.tail ? 1 : blk;
  const step = c.turnIn + c.turnOut, T = Math.max(1, Math.round(c.turns));
  const endL = t => c.sys + t*step;                                   // 第 t 轮结束时的上下文
  const unique = L => bytesAt(p, L) - p.full*c.sys - (p.W ? 0 : p.swa*c.sys);   // 共享前缀只存一份
  const shared = bytesAt(p, c.sys);
  const frag = lb ? 0 : (blk/2)*(p.full + p.swa);                     // 最后一页平均空一半
  let ws1 = 0; for(let t=1;t<=T;t++) ws1 += unique(endL(t)) + frag; ws1 /= T;   // 一个会话在两轮之间留下的平均字节
  const dp = Math.max(1, p.r.dpA), ws = c.sessions/dp*ws1;          // DP attention：会话分到 dpA 组，每张卡只存自己那组的
  const C1 = Math.max(0, mem.avail - shared), C2 = c.hostGB*1e9, C3 = c.remoteGB*1e9;
  if(mem.avail<=0) warn.push(Msg.of("engine_kv.warn.noFit").invalid());
  const local = c.inclusive ? Math.max(C1, C2) : C1+C2, all = c.inclusive ? Math.max(local, C3) : local+C3;
  const route = c.routing==="random" ? 1/Math.max(1, c.replicas) : 1;      // 随机路由：下一轮落回存着它的副本的概率
  const res = (S) => {                                                // S 个会话时每个层级的驻留比例（远端池是所有副本共用的）
    const W = S/dp*ws1, r = C => W>0 ? Math.min(1, C/W) : 1;
    const hbm = route*r(C1), host = route*(r(local)-r(C1)), remote = C3>0 ? Math.max(0, r(all)-route*r(local)) : 0;
    return { hbm, host, remote, miss:Math.max(0, 1-hbm-host-remote) };
  };
  const eta = lb ? 1 : g.etaN, ovl = lb || E.hostOverlap;
  const load = (bytes, bw) => bytes/(bw*1e9*eta);
  // 抽样若干轮算 TTFT（每轮的几种情况和会话数无关，换会话数只换比例）
  const ts = T<=12 ? Array.from({length:T}, (_,i)=>i+1) : Array.from({length:12}, (_,i)=>1+Math.round(i*(T-1)/11));
  const turns = ts.map(t => {
    const Lprev = endL(t-1), Lin = Lprev + c.turnIn, h = t===1 ? c.sys : Math.floor(Lprev/gran)*gran;
    const hSys = Math.floor(c.sys/gran)*gran;
    const tHit = prefillFrom(m, hw, topo, c, g, h, Lin-h);
    const bytesH = Math.max(0, unique(h));
    const lh = load(bytesH, hw.host), lr = load(bytesH, hw.ib);
    const tHost = t===1 ? tHit : ovl ? Math.max(lh, tHit) : lh + tHit;
    const tRemote = t===1 ? tHit : ovl ? Math.max(lr, tHit) : lr + tHit;
    const tMiss = t===1 ? tHit : prefillFrom(m, hw, topo, c, g, hSys, Lin-hSys);
    const tNone = prefillFrom(m, hw, topo, c, g, 0, Lin);
    return { t, Lprev, Lin, hit:h, newTok:Lin-h, hbm:tHit, host:tHost, remote:tRemote, miss:tMiss, none:tNone, loadHost:lh, loadRemote:lr, bytes:bytesH };
  });
  const expect = (S) => { const f = res(S); let tt=0, hit=0, inTok=0;
    for(const x of turns){ const first = x.t===1;
      tt += first ? x.hbm : f.hbm*x.hbm + f.host*x.host + f.remote*x.remote + f.miss*x.miss;
      hit += first ? x.hit : (f.hbm+f.host+f.remote)*x.hit + f.miss*Math.floor(c.sys/gran)*gran; inTok += x.Lin; }
    return { frac:f, ttft:tt/turns.length, hitRate:hit/inTok };
  };
  const cur = expect(c.sessions);
  const none = turns.reduce((s,x)=>s+x.none,0)/turns.length, allHit = turns.reduce((s,x)=>s+x.hbm,0)/turns.length;
  const fit = (C) => ws1>0 ? C/ws1*dp : Infinity;                      // 一个副本能留住多少个会话
  // 空间：每个请求的缓存字节随上下文；"全换成全注意力"的对照（同样的 KV 每 token，所有注意力类层都按 L 存）
  const Ls = [1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576];
  const allFull = (p.nFull+p.nSwa+p.nLin)*(p.nFull ? p.full/p.nFull : p.kvTokRank);
  const space = Ls.map(L => ({ L, bytes:bytesAt(p, L), full:p.full*L, swa:p.swa*(p.W?Math.min(L,p.W):L), lin:p.snap, allFull:allFull*L,
    perGpu:mem.avail>0 ? Math.floor(mem.avail/bytesAt(p, L)) : 0 }));
  const dtypes = ["bf16","fp8","nvfp4"].map(dt => { const q = perRank(m, hw, topo, Object.assign({}, c, { kvDt:dt }), g);
    const mq = Infer.memory(m, hw, Object.assign({}, c, { kvDt:dt, B:1 }), q.r);
    return { kvDt:dt, full:q.full, swa:q.swa, snap:q.snap, at128k:bytesAt(q, 131072), perGpu128k:mq.avail>0 ? Math.floor(mq.avail/bytesAt(q, 131072)) : 0 }; });
  const sweep = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096].map(S => Object.assign({ sessions:S }, expect(S)));
  return { perTok:{ full:p.full, swa:p.swa, W:p.W, snap:p.snap, kvTokRank:p.kvTokRank, snapTok:p.full>0 ? p.snap/p.full : 0 },
    layers:{ full:p.nFull, swa:p.nSwa, lin:p.nLin }, page:blk, gran, overlap:ovl, avail:mem.avail, weights:mem.weights,
    shared, ws1, ws, cap:{ hbm:C1, host:C2, remote:C3 }, fit:{ hbm:fit(C1), host:fit(C2), remote:fit(C3) },
    frac:cur.frac, hitRate:cur.hitRate, ttft:cur.ttft, ttftNone:none, ttftHit:allHit, turns, sweep, space, dtypes, N:p.r.N, tpA:p.r.tpA, dpA:p.r.dpA, warn };
}

return { ENGINES, BASE, plan, perRank, bytesAt };
})();
