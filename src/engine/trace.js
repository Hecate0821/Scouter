/* 实测 trace 和本页的逐 kernel 估计对照。trace 由调用方传进来（这里不读文件）：
   - torch profiler / Chrome trace 导出的 JSON：{ traceEvents: [{ ph:"X", cat:"kernel", name, dur }] }（dur 是 µs，只取 GPU kernel）
   - 或者已经按 kernel 汇总好的列表：[{ name, us, calls }]（也可以包在 { kernels: [...] } 里；us 是这些调用的总时间）
   steps 是 trace 覆盖了几步（decode 步或 prefill chunk），结果按每步给。
   kernel 名按 CAT 的正则归到几个类别（按顺序取第一个匹配）；本页的算子按 OPCAT 归到同样的类别，两边逐类相比。
   正则只看各引擎公开 kernel 名里的常见片段，不是完整的名字表：对不上的进 other，结果里会列出最耗时的那些，方便补规则。
   Chrome trace 的几处口径（对照 fzyzcjy/torch_utils、tlrmchlsmth/combine_traces 和公开的多 rank trace 定的）：
   - 合并过的多 rank trace 按 rank 分开（combine_traces 的 args.source_idx、fzyzcjy 合并脚本的 "[TPxx]" pid，否则按 pid），结果按 rank 平均；
   - 开了 PDL 时同一条 stream 上后一个 kernel 会在前一个结束前启动（停在 griddepcontrol.wait），时间段有重叠：每条 stream 内把重叠
     算给前一个 kernel，只取独占时间；另给 kernel 时间的并集（busy）和首尾跨度（wall），差是 GPU 空闲；
   - 通信 kernel 的时间里含等最慢 rank 的自旋（yushengsu-thu/lora_perf_lora_profile 里 two-shot all-reduce 占到 GPU 时间的 75–87%）：
     每种通信 kernel 按单次最快的 10% 估一个下限（floor），实测减下限是等待（wait）。 */
const Trace = (function(){
/* 顺序有讲究：名字里带 combine / dispatch / gemm 的几类先认——mHC（hyper-connection）的 kernel 是逐元素的，
   attention 的 split-KV 合并、Triton decode attention 的两段、稀疏 MLA 和 indexer 的 Hadamard 归 attention，剩下的 dispatch / combine 才是 EP 通信 */
const CAT = [
  ["comm", /nccl|rccl|all_?reduce|allreduce|all_?gather|reduce_?scatter|all_?to_?all|alltoall|cross_?device|custom_?ar|deep_?ep|mnnvl|one_?shot|two_?shot|lamport/i],
  ["elem", /mhc_|hyper_?conn|\balloc_|memcpy|_set_k_and_s/i],
  ["attn", /splitkv_combine|mla_combine|_fwd(?:_grouped)?_kernel_stage[12]|sparse_?(?:decode|prefill|mla|attn)|flash_?mla|hadamard/i],
  ["comm", /dispatch|combine/i],
  ["moe", /moe|fused_experts|expert|grouped|m_grouped|routing|topk_?gating|permute|\bbmm_|finalize|activationDeepSeek/i],
  ["lin", /gated_delta|delta_rule|fused_recurrent|chunk_(?:fwd|gla|kda|gdn|scaled|local)|kda|gdn|causal_conv1d|conv1d|mamba|selective_scan|ssm|recompute_w_u/i],
  ["attn", /fmha|flash_?attn|flash_fwd|flash_bwd|attention|attn|mla|paged|xqa|mqa_?logits|indexer|batch_?decode|batch_?prefill|decode_?kernel|prefill_?kernel|merge_?state|splitkv/i],
  ["gemm", /gemm|gemv|cublas|cutlass|nvjet|matmul|bmm|xmma|sm\d{2,3}_|linear|dot_?product|\bCijk_/i],
  ["samp", /sampl|softmax|top_?p|top_?k|argmax|penalt|logit/i],
  ["elem", /norm|rope|rotary|silu|gelu|swiglu|act_and_mul|elementwise|vectorized|add|mul|copy|cast|quant|fill|index|gather|scatter|reduce|cat_|split|transpose|embedding/i],
];
/* 采样类的名字（softmax、top-k）每步调用超过这么多次的，是逐层的 router / indexer，不是采样，归 elementwise */
const SAMP_CALLS = 16;
/* 本页算子 → 类别（通信另算 comm） */
const OPCAT = { moe:"moe", moeq:"moe", acore:"attn", aidx:"attn", lconv:"lin", lcore:"lin", samp:"samp",
  norm:"elem", anorm:"elem", lnorm:"elem", aprol:"elem", agate:"elem", mhc:"elem", attnres:"elem" };
const CATS = ["gemm", "attn", "lin", "moe", "comm", "samp", "elem", "other"];
const catOf = name => { for(const [c, re] of CAT) if(re.test(name)) return c; return "other"; };
const opCat = key => OPCAT[key] || "gemm";
const rankOf = e => { if(e.args && e.args.source_idx != null) return "src" + e.args.source_idx; const m = /^\[TP(\d+)\]/.exec(String(e.pid)); return m ? "tp" + m[1] : "pid" + e.pid; };
const q10 = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) * 0.1)]; };

/* trace → 每个 kernel 名每个 rank 平均的独占时间（秒）、原始时间、调用次数，通信 kernel 另有下限；
   以及 rank 数和每个 rank 平均的 raw（各 kernel 时长之和）、busy、wall（Chrome trace 带 ts 时才有后两个） */
function scan(trace){
  const by = {}, evs = trace && (Array.isArray(trace) ? null : trace.traceEvents);
  const add = (name, s, raw, n, dur) => { const k = by[name] || (by[name] = { name, s: 0, raw: 0, calls: 0, durs: catOf(name) === "comm" ? [] : null });
    k.s += s; k.raw += raw; k.calls += n; if(k.durs && dur != null) k.durs.push(dur); };
  let ranks = 1, busy = null, wall = null;
  if(evs){
    const ks = evs.filter(e => e && e.ph === "X" && typeof e.dur === "number" && /kernel|gpu_op/i.test(String(e.cat || "")));
    const rk = new Set(ks.map(rankOf)); ranks = Math.max(1, rk.size);
    const timed = ks.length > 0 && ks.every(e => typeof e.ts === "number");
    if(timed){
      const streams = {}, perRank = {};
      for(const e of ks){ (streams[rankOf(e) + "|" + e.pid + "|" + e.tid] = streams[rankOf(e) + "|" + e.pid + "|" + e.tid] || []).push(e); (perRank[rankOf(e)] = perRank[rankOf(e)] || []).push(e); }
      for(const st of Object.values(streams)){ st.sort((a, b) => a.ts - b.ts); let end = -Infinity;
        for(const e of st){ const t1 = e.ts + e.dur, ex = Math.max(0, t1 - Math.max(e.ts, end)); end = Math.max(end, t1); add(String(e.name), ex * 1e-6, e.dur * 1e-6, 1, e.dur * 1e-6); } }
      busy = 0; wall = 0;
      for(const es of Object.values(perRank)){ es.sort((a, b) => a.ts - b.ts); let lo = es[0].ts, hi = -Infinity, end = -Infinity;
        for(const e of es){ const t1 = e.ts + e.dur; busy += Math.max(0, t1 - Math.max(e.ts, end)); end = Math.max(end, t1); hi = Math.max(hi, t1); }
        wall += hi - lo; }
      busy *= 1e-6 / ranks; wall *= 1e-6 / ranks;
    } else for(const e of ks) add(String(e.name), e.dur * 1e-6, e.dur * 1e-6, 1, e.dur * 1e-6);
  } else {
    const list = Array.isArray(trace) ? trace : trace && trace.kernels;
    if(!Array.isArray(list)) throw new Error("trace must be a Chrome trace ({traceEvents}) or a list of {name, us, calls}");
    for(const k of list){ if(!k || k.name == null || !(+k.us >= 0)) throw new Error("each kernel needs name and us"); add(String(k.name), +k.us * 1e-6, +k.us * 1e-6, +k.calls || 1, null); }
  }
  const ks = Object.values(by).map(k => ({ name: k.name, s: k.s / ranks, raw: k.raw / ranks, calls: k.calls / ranks,
    floor: k.durs && k.durs.length ? q10(k.durs) * k.calls / ranks : null }));
  return { ks, ranks, raw: ks.reduce((a, k) => a + k.raw, 0), busy, wall };
}
const kernelsOf = trace => scan(trace).ks;

/* 对照：r 是 Infer.step 的结果；返回每个类别的实测和本页（每步秒数），以及每类最耗时的 kernel。
   trace 里另一条 stream 上和别的算子重叠的 kernel 照样在，所以本页这边逐类也算上重叠的算子（MoE 里的 shared expert、重叠的通信）；
   一步的墙钟另给：本页的 stepT 对实测的 wall / 步数（没开 CUDA graph 时实测含 CPU 发射的空隙） */
function compare(trace, steps, r){
  const sc = scan(trace), ks = sc.ks, n = Math.max(1, +steps || 1);
  if(!ks.length) throw new Error("trace has no GPU kernels");
  const meas = Object.fromEntries(CATS.map(c => [c, 0])), pred = Object.fromEntries(CATS.map(c => [c, 0])), top = Object.fromEntries(CATS.map(c => [c, []]));
  let floor = 0, floored = false;
  for(const k of ks){ let c = catOf(k.name); if(c === "samp" && k.calls / n > SAMP_CALLS) c = "elem";
    meas[c] += k.s / n; top[c].push({ name: k.name, s: k.s / n, calls: k.calls / n });
    if(c === "comm"){ if(k.floor != null){ floor += Math.min(k.floor, k.s) / n; floored = true; } else floor += k.s / n; } }
  for(const o of r.ops) pred[opCat(o.key)] += o.t;
  for(const x of r.comm) pred.comm += x.t;
  const rows = CATS.filter(c => meas[c] > 0 || pred[c] > 0).map(c => ({ cat: c, meas: meas[c], pred: pred[c], ratio: meas[c] > 0 && pred[c] > 0 ? pred[c] / meas[c] : null,
    top: top[c].sort((a, b) => b.s - a.s).slice(0, 5) }));
  const tm = rows.reduce((s, x) => s + x.meas, 0), tp = rows.reduce((s, x) => s + x.pred, 0);
  return { steps: n, kernels: ks.length, ranks: sc.ranks, rows, total: { meas: tm, pred: tp, ratio: tm > 0 ? tp / tm : null },
    overlap: (sc.raw - ks.reduce((a, k) => a + k.s, 0)) / n,
    gpu: sc.wall == null ? null : { busy: sc.busy / n, wall: sc.wall / n, idle: (sc.wall - sc.busy) / n, step: r.stepT, ratio: sc.wall > 0 ? r.stepT * n / sc.wall : null },
    comm: floored ? { meas: meas.comm, floor, wait: meas.comm - floor, pred: pred.comm, ratio: floor > 0 && pred.comm > 0 ? pred.comm / floor : null } : null,
    unmatched: top.other.slice(0, 10) };
}

return { CAT, OPCAT, CATS, catOf, opCat, scan, kernelsOf, compare };
})();
