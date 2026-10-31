/* 实测 trace 和本页的逐 kernel 估计对照。trace 由调用方传进来（这里不读文件）：
   - torch profiler / Chrome trace 导出的 JSON：{ traceEvents: [{ ph:"X", cat:"kernel", name, dur }] }（dur 是 µs，只取 GPU kernel）
   - 或者已经按 kernel 汇总好的列表：[{ name, us, calls }]（也可以包在 { kernels: [...] } 里；us 是这些调用的总时间）
   steps 是 trace 覆盖了几步（decode 步或 prefill chunk），结果按每步给。
   kernel 名按 CAT 的正则归到几个类别（按顺序取第一个匹配）；本页的算子按 OPCAT 归到同样的类别，两边逐类相比。
   正则只看各引擎公开 kernel 名里的常见片段，不是完整的名字表：对不上的进 other，结果里会列出最耗时的那些，方便补规则。 */
const Trace = (function(){
const CAT = [
  ["comm", /nccl|all_?reduce|allreduce|all_?gather|reduce_?scatter|all_?to_?all|alltoall|cross_?device|custom_?ar|deep_?ep|mnnvl|one_?shot|two_?shot|lamport|dispatch|combine/i],
  ["moe", /moe|fused_experts|expert|grouped|m_grouped|routing|topk_?gating|permute/i],
  ["lin", /gated_delta|delta_rule|fused_recurrent|chunk_(?:fwd|gla|kda|gdn|scaled|local)|kda|gdn|causal_conv1d|conv1d|mamba|selective_scan|ssm/i],
  ["attn", /fmha|flash_?attn|flash_fwd|flash_bwd|attention|attn|mla|paged|xqa|mqa_?logits|indexer|batch_?decode|batch_?prefill|decode_?kernel|prefill_?kernel|merge_?state|splitkv/i],
  ["gemm", /gemm|gemv|cublas|cutlass|nvjet|matmul|bmm|xmma|sm\d{2,3}_|linear|dot_?product/i],
  ["samp", /sampl|softmax|top_?p|top_?k|argmax|penalt|logit/i],
  ["elem", /norm|rope|rotary|silu|gelu|swiglu|act_and_mul|elementwise|vectorized|add|mul|copy|cast|quant|fill|index|gather|scatter|reduce|cat_|split|transpose|embedding/i],
];
/* 本页算子 → 类别（通信另算 comm） */
const OPCAT = { moe:"moe", moeq:"moe", acore:"attn", aidx:"attn", lconv:"lin", lcore:"lin", samp:"samp",
  norm:"elem", anorm:"elem", lnorm:"elem", aprol:"elem", agate:"elem", mhc:"elem", attnres:"elem" };
const CATS = ["gemm", "attn", "lin", "moe", "comm", "samp", "elem", "other"];
const catOf = name => { for(const [c, re] of CAT) if(re.test(name)) return c; return "other"; };
const opCat = key => OPCAT[key] || "gemm";

/* trace → 每个 kernel 名的总时间（秒）和调用次数 */
function kernelsOf(trace){
  const by = {};
  const add = (name, s, n) => { const k = by[name] || (by[name] = { name, s: 0, calls: 0 }); k.s += s; k.calls += n; };
  const evs = trace && (Array.isArray(trace) ? null : trace.traceEvents);
  if(evs){
    for(const e of evs) if(e && e.ph === "X" && typeof e.dur === "number" && /kernel|gpu_op/i.test(String(e.cat || ""))) add(String(e.name), e.dur * 1e-6, 1);
  } else {
    const list = Array.isArray(trace) ? trace : trace && trace.kernels;
    if(!Array.isArray(list)) throw new Error("trace must be a Chrome trace ({traceEvents}) or a list of {name, us, calls}");
    for(const k of list){ if(!k || k.name == null || !(+k.us >= 0)) throw new Error("each kernel needs name and us"); add(String(k.name), +k.us * 1e-6, +k.calls || 1); }
  }
  return Object.values(by);
}

/* 对照：r 是 Infer.step 的结果；返回每个类别的实测和本页（每步秒数），以及每类最耗时的 kernel */
function compare(trace, steps, r){
  const ks = kernelsOf(trace), n = Math.max(1, +steps || 1);
  if(!ks.length) throw new Error("trace has no GPU kernels");
  const meas = Object.fromEntries(CATS.map(c => [c, 0])), pred = Object.fromEntries(CATS.map(c => [c, 0])), top = Object.fromEntries(CATS.map(c => [c, []]));
  for(const k of ks){ const c = catOf(k.name); meas[c] += k.s / n; top[c].push({ name: k.name, s: k.s / n, calls: k.calls / n }); }
  for(const o of r.ops) if(o.overlap !== "moe") pred[opCat(o.key)] += o.t;
  for(const x of r.comm) if(!x.overlap) pred.comm += x.t;
  const rows = CATS.filter(c => meas[c] > 0 || pred[c] > 0).map(c => ({ cat: c, meas: meas[c], pred: pred[c], ratio: meas[c] > 0 && pred[c] > 0 ? pred[c] / meas[c] : null,
    top: top[c].sort((a, b) => b.s - a.s).slice(0, 5) }));
  const tm = rows.reduce((s, x) => s + x.meas, 0), tp = rows.reduce((s, x) => s + x.pred, 0);
  return { steps: n, kernels: ks.length, rows, total: { meas: tm, pred: tp, ratio: tm > 0 ? tp / tm : null }, unmatched: top.other.slice(0, 10) };
}

return { CAT, OPCAT, CATS, catOf, opCat, kernelsOf, compare };
})();
