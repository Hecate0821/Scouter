/* ================= 一个 step 的显存时间线（torchtitan FSDP2，pp = 1） =================
   不另建模型：计算和通信时间、每层保存的激活字节、各类显存全部取 Train.run 的结果，这里只按 FSDP2 的生命周期把它们排到时间轴上。
   - FSDP2（torch/distributed/fsdp/_fully_shard）：unshard（下一块提前预取）→ 前向后 reshard → 反向再 unshard →
     post-backward 把未切分的梯度转成 reduce dtype 交给 reduce-scatter，结果累加进本卡的梯度分片。
   - torchtitan apply_fsdp_to_decoder：每个 transformer 块一个 FSDP 单元；dense 部分（注意力 / 线性注意力、shared、router 或 dense FFN）
     在 dp_shard × cp 组，routed experts 在 edp 组（同一个 FSDPState 的两个 param group）。tok_embeddings 反向不 unshard
     （set_unshard_in_backward(False)），[norm, lm_head] 前向后不 reshard。
   - 所有 FSDPState 共用 FSDPCommContext 的 all_gather_stream 和 reduce_scatter_stream：dense 和 expert 的 all-gather 排在同一条流上，
     reduce-scatter 在另一条。o.agMode = "split" 是假设的对照（两组各一条 AG 流），不是 FSDP2 的行为。
   - 梯度分片在第一个 micro-batch 的 reduce-scatter 落地时才分配（zero_grad 用 set_to_none），之后各 micro-batch 累加进去，
     optimizer.step 之后 zero_grad 释放。fp32 参数分片 + Adam m / v 整个 step 常驻（offload 时在主机）。
   - 只排第一个和最后一个 micro-batch（中间的和最后一个一样），再接优化器。HSDP 的副本 all-reduce 和 Train.run 一样每步一次，放在最后一个 micro-batch。
   输出只有数字、类别键和计算核心的 Msg，文字由界面翻译。 */
const MemTL = (function(){
const CATS = ["opt", "gsh", "work", "agD", "agE", "actA", "actL", "actM", "grad", "rcws", "qw", "buf", "logit"];
const CI = Object.fromEntries(CATS.map((k, i) => [k, i]));
const ACT = { full:"actA", swa:"actA", lin:"actL", moe:"actM", dense:"actM" };
const SECS = ["full", "swa", "lin", "moe", "dense", "head"];
const BLK = { attn:["full","swa"], full:["full"], swa:["swa"], lin:["lin"], moe:["moe"], dense:["dense"], head:["head"] };
const LANE = { a2a:"ep", cnt:"ep", etp:"ep", tp:"tp", tpidx:"tp", cpkv:"cp", cplin:"cp", cpswa:"cp" };

/* 每个小节（一种注意力 / 一种 MLP / 输出头）每层的前向、重算、反向计算时间，以及每层的逐次通信。
   只属于一种小节的算子按该小节的层数平均（CSA 各压缩比的算子各管一部分层）；注意力里全注意力和滑窗共用的算子两边都算一层一份 */
function sections(r, nL){
  const sec = {}; for(const k of SECS) sec[k] = { f:0, b:0, rc:0, comm:{ fwd:[], rc:[], bwd:[] } };
  const where = o => o.group==="attn" ? (o.mod ? [o.mod] : BLK.attn) : o.group==="lin" ? BLK.lin : o.group==="lm" ? BLK.head : o.key==="dense" ? BLK.dense : o.group==="moe" ? BLK.moe : [];
  let pend = null;
  for(const o of r.ops){
    if(o.key==="norm"){ pend = o; continue; }
    const ss = where(o).filter(s => nL[s]>0);
    for(const x of pend ? [pend, o] : [o]) for(const s of ss){
      const per = ss.length>1 ? 1/(x.layers||1) : 1/nL[s];
      sec[s].f += x.fwd.t*per; sec[s].b += (x.dgrad.t+x.wgrad.t)*per; if(x.rc) sec[s].rc += x.fwd.t*per;
    }
    pend = null;
  }
  for(const x of r.comm){
    const lane = LANE[x.key], ss = (BLK[x.blk]||[]).filter(s => nL[s]>0);
    if(!lane || !x.xs || !ss.length) continue;
    const ex = x.t>0 && x.exp!=null ? x.exp/x.t : 1, w = Math.min(1, (x.layers||1)/ss.reduce((a, s) => a+nL[s], 0));
    for(const s of ss) for(const y of x.xs){ const p = y.pass==="rc" ? "rc" : y.pass==="bwd" ? "bwd" : "fwd";
      sec[s].comm[p].push({ lane, t:y.t*w, ex, op:y.op, what:y.what, bytes:y.bytes*w }); }
  }
  return sec;
}

/* m：模型；r：Train.run(m, …) 的结果；g：口径；o.agMode："shared"（FSDP2）/ "split"（对照） */
function run(m, r, g, o){
  const F = r.par && r.par.fsdp; if(!F || r.pp>1) return null;
  o = o || {};
  const d = Common.derive(m), plan = Common.layerPlan(m), tp = r.tp, ep = r.ep, mem = r.mem, Tc = r.Tc;
  const nL = { head:1, moe:plan.filter(x => x.mlp==="moe").length }; nL.dense = plan.length-nL.moe;
  for(const k of ["full", "swa", "lin"]) nL[k] = plan.filter(x => x.attn===k).length;
  const sec = sections(r, nL);
  // full AC 每层只存块输入（D·2/TP 字节），记在注意力和 MLP 两半各一半
  const saved = s => (mem.rcFull ? m.D*2/tp/2 : mem.layerAct[s])*Tc, whole = s => mem.layerAct0[s]*Tc;
  const fs = F.fs, fe = F.fe, gB = F.gB, nM = r.microsPerStep;
  const wt = (bytes, l) => Par.wireT(bytes, l, g.etaAg, 1, g);
  const qPer = mem.qW>0 ? mem.qW/((d.pDenseAll-(d.pEngTab||0))/tp + d.pExpertsAll/ep) : 0;
  const cpBuf = s => s==="full" && r.par.cpFull ? r.par.cpFull.buf : s==="swa" && r.par.cpSwa ? r.par.cpSwa.buf : 0;
  const ev = [], segs = [], marks = [];
  const add = (k, t0, t1, dv) => { if(dv) ev.push({ c:CI[k], t0, t1:Math.max(t0, t1), dv }); };
  const seg = (lane, t0, t1, kind, x) => { if(t1>t0) segs.push(Object.assign({ lane, t0, t1, kind }, x||{})); };
  const onDev = mem.states>0;                                         // offload 时 fp32 分片、Adam 状态和梯度分片都在主机
  if(onDev) add("opt", 0, 0, r.Pshard*12);
  add("work", 0, 0, mem.work);

  /* FSDP 单元：embedding、每层一个、[norm, lm_head]。d = dense 组的本卡参数（已按 TP 切），e = routed experts（已按 EP 切） */
  const blocks = [{ kind:"emb", d:d.pEmb/tp, e:0 }];
  plan.forEach((x, i) => blocks.push({ kind:"layer", i, attn:x.attn, mlp:x.mlp, d:F.blocks[i].d, e:F.blocks[i].e }));
  blocks.push({ kind:"head", d:d.pHead/tp, e:0 });
  const gd = b => fs>1 ? b.d : 0, ge = b => fe>1 ? b.e : 0;          // 分片组只有 1 张卡时参数本来就是整份，不 gather

  let t = 0, agFree = 0, agFreeD = 0, agFreeE = 0, rsFree = 0, gsh = 0;
  const shared = o.agMode!=="split";
  function issueAG(b, tI, pass, mb){
    const bD = gd(b)*(fs-1)/fs*2, bE = ge(b)*(fe-1)/fe*2, tD = bD ? wt(bD, F.lkD) : 0, tE = bE ? wt(bE, F.lkE) : 0;
    let sD, sE;
    if(shared){ sD = Math.max(tI, agFree); sE = sD+tD; agFree = sE+tE; }
    else { sD = Math.max(tI, agFreeD); agFreeD = sD+tD; sE = Math.max(tI, agFreeE); agFreeE = sE+tE; }
    add("agD", tI, tI, gd(b)*2); add("agE", tI, tI, ge(b)*2);          // 输出缓冲在发起时就在 AG 流上分配
    seg(shared ? "ag" : "agD", sD, sD+tD, "agD", { b, pass, mb, bytes:bD, F:fs });
    seg(shared ? "ag" : "agE", sE, sE+tE, "agE", { b, pass, mb, bytes:bE, F:fe });
    return Math.max(sD+tD, bE ? sE+tE : 0);
  }
  const freeAG = (b, tF) => { add("agD", tF, tF, -gd(b)*2); add("agE", tF, tF, -ge(b)*2); };
  // 小节里的 TP / CP / EP 通信均匀摆在计算窗口里（露出的部分已经算进窗口长度）
  function commIn(s, pass, t0, W, b, mb){
    const xs = sec[s].comm[pass];
    xs.forEach((x, i) => { const st = Math.min(Math.max(t0, t0+W*(i+0.5)/xs.length-x.t/2), t0+Math.max(0, W-x.t));
      seg(x.lane, st, st+x.t, x.lane, { op:x.op, what:x.what, bytes:x.bytes, b, pass, mb }); });
  }
  const passT = (s, pass) => (pass==="fwd" ? sec[s].f : pass==="rc" ? sec[s].rc : sec[s].b) + sec[s].comm[pass].reduce((a, x) => a+x.t*x.ex, 0);
  const bufs = (s, t0, t1) => { const v = (s==="moe" ? mem.a2aBuf : 0) + cpBuf(s); if(v){ add("buf", t0, t0, v); add("buf", t1, t1, -v); } };
  const EPS = 1e-6;

  function microBatch(mb, first, last){
    const ready = new Array(blocks.length).fill(0), t0 = t;
    marks.push({ t, kind:"fwd", mb });
    ready[0] = issueAG(blocks[0], t, "fwd", mb);
    for(let j = 0; j<blocks.length; j++){
      const b = blocks[j], start = Math.max(t, ready[j]);
      if(j+1<blocks.length) ready[j+1] = issueAG(blocks[j+1], start, "fwd", mb);
      seg("comp", t, start, "wait", { b, pass:"fwd", mb }); t = start;
      if(b.kind==="emb"){ seg("comp", t, t+EPS, "f", { sec:"emb", b, pass:"fwd", mb }); t += EPS; }
      else if(b.kind==="head"){
        const W = passT("head", "fwd"); seg("comp", t, t+W, "f", { sec:"head", b, pass:"fwd", mb }); commIn("head", "fwd", t, W, b, mb);
        add("logit", t, t+W, mem.logits); t += W;
      } else {
        if(qPer>0) add("qw", t, t, (b.d+b.e)*qPer);
        for(const s of [b.attn, b.mlp]){ const W = passT(s, "fwd");
          seg("comp", t, t+W, "f", { sec:s, b, pass:"fwd", mb }); commIn(s, "fwd", t, W, b, mb);
          add(ACT[s], t, t+W, saved(s)); bufs(s, t, t+W); t += W; }
        if(qPer>0 && mem.rcFull) add("qw", t, t, -(b.d+b.e)*qPer);
      }
      if(b.kind!=="head" && F.reshard) freeAG(b, t);
    }
    const tB = t; marks.push({ t, kind:"bwd", mb });
    /* 反向：head、最后一层 … 第 0 层、embedding；下一块的 unshard 在这一块开始时预取 */
    const order = [...blocks].reverse(), readyB = new Map([[order[0], t]]);
    for(let j = 0; j<order.length; j++){
      const b = order[j], start = Math.max(t, readyB.get(b) ?? t), nx = order[j+1];
      if(nx) readyB.set(nx, F.reshard && nx.kind!=="emb" ? issueAG(nx, start, "bwd", mb) : start);
      seg("comp", t, start, "wait", { b, pass:"bwd", mb }); t = start;
      if(b.kind==="head"){
        const W = passT("head", "bwd"); seg("comp", t, t+W, "b", { sec:"head", b, pass:"bwd", mb }); commIn("head", "bwd", t, W, b, mb);
        add("grad", t, t+W, b.d*2); t += W; add("logit", t, t, -mem.logits);
      } else if(b.kind==="emb"){ add("grad", t, t+EPS, b.d*2); seg("comp", t, t+EPS, "b", { sec:"emb", b, pass:"bwd", mb }); t += EPS; }
      else {
        for(const s of [b.attn, b.mlp]){ const W = passT(s, "rc"); if(W<=0) continue;
          seg("comp", t, t+W, "rc", { sec:s, b, pass:"rc", mb }); commIn(s, "rc", t, W, b, mb);
          add("rcws", t, t+W, Math.max(0, whole(s)-saved(s))); bufs(s, t, t+W);
          if(qPer>0 && mem.rcFull && s===b.attn) add("qw", t, t, (b.d+b.e)*qPer);
          t += W; }
        const bT0 = t;
        for(const s of [b.mlp, b.attn]){ const W = passT(s, "bwd");
          seg("comp", t, t+W, "b", { sec:s, b, pass:"bwd", mb }); commIn(s, "bwd", t, W, b, mb);
          add(ACT[s], t, t+W, -saved(s));
          if(passT(s, "rc")>0) add("rcws", t, t+W, -Math.max(0, whole(s)-saved(s)));
          bufs(s, t, t+W); t += W; }
        add("grad", bT0, t, (b.d+b.e)*2);
        if(qPer>0) add("qw", t, t, -(b.d+b.e)*qPer);
      }
      // post-backward：reshard（embedding 已在前向后释放）；未切分的 BF16 梯度转成 reduce dtype 交给 RS 流，dense 组先、expert 组后
      if(b.kind!=="emb" || !F.reshard) freeAG(b, t);
      add("grad", t, t, -(b.d+b.e)*2); add("grad", t, t, (b.d+b.e)*gB);
      const bD = gd(b)*(fs-1)/fs*gB, bE = ge(b)*(fe-1)/fe*gB, rsD = bD ? wt(bD, F.lkD) : 0, rsE = bE ? wt(bE, F.lkE) : 0;
      let s0 = Math.max(t, rsFree);
      seg("rs", s0, s0+rsD, "rsD", { b, mb, bytes:bD, F:fs }); s0 += rsD;
      add("grad", s0, s0, -b.d*gB); if(first && onDev){ add("gsh", s0, s0, b.d/fs*4); gsh += b.d/fs*4; }
      if(b.e){ seg("rs", s0, s0+rsE, "rsE", { b, mb, bytes:bE, F:fe }); s0 += rsE;
        add("grad", s0, s0, -b.e*gB); if(first && onDev){ add("gsh", s0, s0, b.e/fe*4); gsh += b.e/fe*4; } }
      if(last && F.R>1){ const bA = 2*(F.R-1)/F.R*(b.d/fs+b.e/fe)*gB, tA = wt(bA, F.lkR);
        seg("rs", s0, s0+tA, "ar", { b, mb, bytes:bA, F:F.R }); s0 += tA; }
      rsFree = s0;
    }
    return { t0, tB, t1:t };
  }

  const mbs = [microBatch(1, true, nM===1)];
  if(nM>1) mbs.push(microBatch(nM, false, true));
  t = Math.max(t, rsFree);
  marks.push({ t, kind:"opt" });
  seg("comp", t, t+r.tOpt, "opt", {}); t += r.tOpt;
  add("gsh", t, t, -gsh);                                            // zero_grad(set_to_none=True)
  const tEnd = Math.max(t, ...segs.map(s => s.t1));

  /* 扫一遍：每类显存是分段线性的，在事件点上精确叠加 */
  const nc = CATS.length, pts = new Map();
  const at = T => { let p = pts.get(T); if(!p){ p = { sl:new Float64Array(nc), st:new Float64Array(nc), hasSt:false }; pts.set(T, p); } return p; };
  for(const e of ev){ if(e.t1<=e.t0){ const p = at(e.t0); p.st[e.c] += e.dv; p.hasSt = true; }
    else { const k = e.dv/(e.t1-e.t0); at(e.t0).sl[e.c] += k; at(e.t1).sl[e.c] -= k; } }
  at(tEnd);
  const times = [...pts.keys()].sort((a, b) => a-b), cur = new Float64Array(nc), slope = new Float64Array(nc), out = [];
  let prev = times[0];
  for(const T of times){ const p = pts.get(T), dt = T-prev;
    for(let i = 0; i<nc; i++) cur[i] += slope[i]*dt;
    out.push({ t:T, v:Float64Array.from(cur) });
    if(p.hasSt){ for(let i = 0; i<nc; i++) cur[i] += p.st[i]; out.push({ t:T, v:Float64Array.from(cur) }); }
    for(let i = 0; i<nc; i++) slope[i] += p.sl[i];
    prev = T; }
  let peak = 0, pkI = 0; out.forEach((x, i) => { const s = x.v.reduce((a, b) => a+b, 0); x.tot = s; if(s>peak+1){ peak = s; pkI = i; } });
  const peakT = out[pkI].t, peakSeg = segs.find(s => s.lane==="comp" && s.kind!=="wait" && s.t0<=peakT && peakT<=s.t1) || null;
  const inMB1 = s => s.mb===1;
  const agBusy = segs.filter(s => (s.kind==="agD" || s.kind==="agE") && inMB1(s)).reduce((a, s) => a+s.t1-s.t0, 0);
  const waitT = segs.filter(s => s.kind==="wait" && inMB1(s)).reduce((a, s) => a+s.t1-s.t0, 0);
  return { cats:CATS, out, segs, marks, tEnd, peak, peakT, peakV:out[pkI].v, peakSeg, mbs, mbT:mbs[0].t1-mbs[0].t0, agBusy, waitT, nM, shared, fs, fe, gB, R:F.R, reshard:F.reshard };
}

return { CATS, run, sections };
})();
