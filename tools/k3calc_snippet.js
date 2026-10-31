// Run in the browser console on https://byronhsu.github.io/k3-inference-calculator/ (the page runs its own Python estimator
// in a Pyodide worker; this snippet only calls the page's k3CalculatorFetch). It prints one line per scenario:
//   phase hw batch len total_s sum_s link_bytes model_batch | group:flops,hbm_bytes,seconds ...
// Save the printed text to a file and convert it with:  uv run --no-project tools/import_k3calc.py lines.txt --ref <commit>
// Groups follow Scouter's modules: MLA projections / core, KDA projections / core, routed experts, the rest of the MoE
// (merged front GEMM with router, latent down and shared gate/up; latent up; shared down), dense FFN, lm head,
// communication and everything else. Sums are over all layers, per rank.
(async () => {
  const S = [["decode","b300",1,8192],["decode","b300",16,8192],["decode","b300",64,8192],["decode","b300",256,8192],["decode","b300",64,65536],
    ["decode","gb300",64,8192],["decode","h200",64,8192],["prefill","b300",1,2048],["prefill","b300",1,8192],["prefill","b300",1,16384],["prefill","h200",1,8192]];
  const G = { mla_proj:"mla_a_proj mla_q_b mla_q_absorb mla_v_deabsorb mla_output_gate mla_kv_b mla_o_proj", mla_core:"mla_absorbed_core mla_mha_core",
    kda_proj:"kda_qkvg kda_bfa kda_forget kda_o_proj", kda_core:"kda_fused_decode kda_short_convs kda_recurrence kda_gated_rmsnorm", moe_exp:"moe_routed_experts",
    moe_rest:"moe_fused_front moe_latent_up_replicated moe_shared_down moe_gemm_all_gather_tail moe_routed_input_contiguous moe_routed_input_quant_pack moe_route_topk moe_topk moe_predispatch_quant moe_shared_situ moe_latent_norm",
    dense:"dense_gate_up dense_down dense_situ", lm:"lm_head_gemm lm_head_logit_scale" };
  const of = {}; for (const g in G) for (const id of G[g].split(" ")) of[id] = g;
  const f = x => x ? (+x.toPrecision(4)).toExponential().replace("e+", "e") : "0";
  const out = [];
  for (const [phase, hw, B, L] of S) {
    const p = { phase, hardware:[hw], tp_size:8, moe_sharding:"tp", batch_size:B };
    if (phase === "prefill") p.sequence_length = L; else p.context_length = L;
    const r = await (await k3CalculatorFetch("./api/calculate", { method:"POST", body:JSON.stringify(p) })).json();
    if (r.error) { out.push("error " + JSON.stringify(p) + " " + JSON.stringify(r.error)); continue; }
    const one = r.results[0], acc = {}; let link = 0, sum = 0;
    for (const l of one.layers) for (const v of l.operations) {
      const g = of[v.id] || (v.category === "communication" ? "comm" : "other"), a = acc[g] || (acc[g] = [0, 0, 0]);
      a[0] += v.flops_per_rank; a[1] += v.hbm_bytes_per_rank; a[2] += v.duration_seconds; link += v.link_bytes_per_rank; sum += v.duration_seconds;
    }
    out.push([phase, hw, B, L, f(one.total_seconds), f(sum), f(link), one.workload.model_batch_size].join(" ") + " | " +
      Object.keys(acc).sort().map(g => g + ":" + acc[g].map(f).join(",")).join(" "));
  }
  console.log(out.join("\n"));
})();
