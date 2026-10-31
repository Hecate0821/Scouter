// FLOPs parity: every model's matmul work, written out again from the model config with textbook formulas, against the
// engine's per-op FLOPs. The formulas here do not call Common.derive or any engine helper, so a slip in either side shows up.
// Counted per token, then × rows; one GPU (no TP), batch 64 on the CUDA-graph ladder, imbalance 1, BF16, reference engine.
// - GQA: q/k/v (+ output gate) and o projections; core 4·Hq·dh per key (QK + PV), causal prefill sees ctx + T/2 keys, a
//   sliding-window layer at most W.
// - MLA / DSA, decode and DSA prefill absorbed: q_a + kv_a, q_b, W_uk, latent MQA over (kvLora + rope) for QK and kvLora for PV,
//   W_uv, o (+ gate). DSA adds the indexer projections and FP8 logits over ctx / pool keys and attends to at most top-k keys.
//   MLA prefill is the MHA form: kv_b expands the new tokens and re-expands the prefix latent, core 2·H·(dqk + vh) per key.
// - GDN / KDA: in_proj (q, k, v, z + β, α; KDA adds the forget gate, low rank ≈ Hk·dk), depthwise conv, o_proj; decode core
//   8·Hv·dk·dv per token (the same count as BBuf/AI-Infra-Auto-Driven-SKILLS' model-compute-simulation, written independently).
//   Prefill uses the chunked algorithm, whose FLOPs depend on the chunk size, so only its projections are compared.
// - MoE: router D·E, k routed experts of 3·De·F, shared 3·D·Fs, latent down / up D·Dlat each; dense FFN 3·D·F; lm head D·V.
// Elementwise work (norms, RoPE, gates, softmax, sampling) is left out on both sides. Tolerance 0.5% (the router's softmax
// term and DSA's top-k + 3 window).
// Usage: node flops.js
const load = require("../src/engine/node.js");
const E = load();
const { MODELS, HW, TOPO, G0 } = E.Common;

const GROUPS = {
  attn: ["aproj", "aout", "acore", "aqa", "aqb", "agp", "auk", "auv", "aidxp", "aidx", "akvb"],
  lin: ["lproj", "lconv", "lcore", "lout"], linProj: ["lproj", "lout"],
  moe: ["router", "moe", "shared", "latd", "latu", "dense"], lm: ["lm"],
};
const sumOf = (r, keys) => r.ops.filter(o => keys.includes(o.key)).reduce((s, o) => s + o.flops, 0);

function expected(m, ph, T, ctx){
  const D = m.D, f = m.full, l = m.lin, e = m.moe, dec = ph === "decode";
  const out = { attn: 0, lin: 0, linProj: 0, moe: 0, lm: 0 };
  // which layers are full / sliding-window / linear, and which MLPs are dense: count the same way the HF configs describe them
  const nSwa = f.swa || 0, nGlob = f.n - nSwa, nMoe = m.L - m.dense.n;
  if (f.n > 0 && f.kind === "gqa") {
    const gate = f.gate ? (f.gateHead ? f.Hq : f.Hq * f.dh) : 0;
    const proj = 2 * D * (f.Hq * f.dh + gate + 2 * f.Hkv * f.dh) + 2 * f.Hq * f.dh * D;
    const keys = cap => dec ? Math.min(ctx, cap) : Math.min(ctx + T / 2, cap);
    out.attn = f.n * T * proj + T * 4 * f.Hq * f.dh * (nGlob * keys(Infinity) + nSwa * keys(f.W || Infinity));
  } else if (f.n > 0) {
    const H = f.H, lat = f.kvLora + f.rope, dqk = f.nope + f.rope;
    let per = 2 * D * (f.qLora + lat) + 2 * f.qLora * H * dqk + 2 * H * f.vh * D + (f.gate ? 2 * D * H * f.vh : 0);
    if (f.kind === "dsa") per += 2 * (f.qLora * f.idxH * f.idxD + D * f.idxD + D * f.idxH);
    const seen = dec ? ctx : ctx + T / 2;
    let core = 0;
    if (f.kind === "dsa" && seen > f.topk) core += 2 * f.idxH * f.idxD * seen / (f.kpool || 1);          // indexer logits
    if (dec || f.kind === "dsa") {
      const kv = f.kind === "dsa" ? Math.min(seen, f.topk) : seen;
      per += 2 * H * f.nope * f.kvLora + 2 * H * f.kvLora * f.vh;                                            // W_uk, W_uv
      core += 2 * H * (lat + f.kvLora) * kv;
    } else core += 2 * H * (dqk + f.vh) * seen;
    // MHA-form prefill: kv_b expands the new tokens, and the prefix latent again each chunk (SGLang MHA_CHUNKED_KV,
    // vLLM _compute_prefill_context), so it is per layer, not per token
    const expand = dec || f.kind === "dsa" ? 0 : 2 * (T + ctx) * f.kvLora * H * (f.nope + f.vh);
    out.attn = f.n * (T * (per + core) + expand);
  }
  if (l.n > 0) {
    const inW = 2 * l.Hk * l.dk + 2 * l.Hv * l.dv + 2 * l.Hv + (l.kind === "KDA" ? l.Hk * l.dk : 0);
    const convDim = 2 * l.Hk * l.dk + l.Hv * l.dv;
    out.linProj = l.n * T * (2 * D * inW + 2 * l.Hv * l.dv * D);
    out.lin = out.linProj + (dec ? l.n * T * (2 * l.conv * convDim + 8 * l.Hv * l.dk * l.dv) : NaN);
  }
  const De = e.Dlat || D;
  out.moe = nMoe * T * (2 * D * e.E + 2 * e.k * 3 * De * e.F + 2 * 3 * D * e.Fs + (e.Dlat ? 4 * D * e.Dlat : 0)) + m.dense.n * T * 2 * 3 * D * m.dense.F;
  out.lm = (dec ? T : 1) * 2 * D * m.V;
  return out;
}

const problems = [];
let n = 0;
for (const key of Object.keys(MODELS)) for (const ph of ["decode", "prefill"]) {
  const m = MODELS[key], T = 64, ctx = ph === "decode" ? 4096 : 8192;
  const cfg = Object.assign({}, E.Infer.BASE, { engine: "ref", phase: ph, N: 1, B: T, C: T, n: ctx, p: ctx, imb: 1, wDt: "bf16", dDt: "bf16", kvDt: "bf16" });
  const r = E.Infer.step(m, HW.B200, TOPO.hgx8, cfg, G0), want = expected(m, ph, T, ctx);
  if (r.rowsG !== T || r.rowsR !== T) { problems.push(`${key} ${ph}: rows ${r.rowsG} / ${r.rowsR}, expected ${T}`); continue; }
  for (const g of Object.keys(want)) {
    if (g === "lin" && ph === "prefill") continue;
    const got = sumOf(r, GROUPS[g]), exp = want[g]; n++;
    if (!exp && !got) continue;
    const rel = Math.abs(got - exp) / Math.max(exp, 1);
    if (!(rel <= 0.005)) problems.push(`${key} ${ph} ${g}: engine ${got.toExponential(4)} FLOPs, closed form ${exp.toExponential(4)} (${(rel * 100).toFixed(2)}% apart)`);
  }
}
console.log(`flops: ${n} group comparisons over ${Object.keys(MODELS).length} models × decode / prefill`);
if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
console.log("flops ok");
