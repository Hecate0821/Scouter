/* engine_common: text from src/engine/common.js (model notes, official parameter counts, hardware / topology names) */
I18N.add("en", "engine_common", {
  "model.qwen35.official": "35B total / 3B active",
  "model.qwen35.note": "40 layers = 10 × (3 × Gated DeltaNet + 1 × GQA with output gate). Every layer is MoE: 256 routed experts, top-8, plus 1 shared expert with a sigmoid gate.",
  "model.kimik3.official": "2.8T total / 104B active",
  "model.kimik3.note": "69 KDA layers + 24 MLA layers with a sigmoid output gate (NoPE, but KV still stores 512+64 dims). Routed experts compute in a 3584-dim latent space (router, latent down-projection and shared gate_up are one bf16 GEMM); the 2 shared experts are merged into F=6144 and computed at 7168 dims (vLLM per-GPU constant SI=768@TP8). Layer 0 is a dense FFN. There is also an AttnRes block residual (up to 8 snapshots). In the official checkpoint only the routed experts are MXFP4; the rest is BF16. The active parameters computed from these fields are a few B above the official 104B; the difference comes from the basis used for the shared experts.",
  "model.glm53f.official": "320B total / 18B active",
  "model.glm53f.note": "34 KDA layers + 11 DSA layers (i%4==3). DSA = NoPE MLA (qk_nope = v = 256, KV 512 dims) + KPool lightning indexer: every 4 tokens are pooled into one key, and the top-512 pools (about 2048 tokens) are selected. The first 3 layers are dense FFN. 4-way mHC residual stream. The main repo releases FP8 block weights.",
  "model.musespark.name": "Muse Spark Open (816B-A42B)",
  "model.musespark.official": "816B total / 42B active",
  "model.musespark.note": "All 62 layers are softmax GQA (128 q heads, 16 KV heads, head dim 64, one sigmoid gate per head): 46 layers use a 2048 sliding window + RoPE, and 16 layers (i%4==1) see the full context with no positional encoding; only these 16 layers have KV that grows with context. Every layer is MoE: top-8 of 256, no shared expert; the router reads fp32 at 8192 dims, and experts compute in a 4096-dim latent (pre / post_expert_proj stay BF16). The official NVFP4 checkpoint quantizes only the experts (533 GB); the BF16 version is 1.64 TB. The active parameters computed from these fields are about 40B; the official basis gives 42B.",
  "model.qwen38_27b.name": "Qwen3.8-27B (dense)",
  "model.qwen38_27b.official": "27B dense",
  "model.qwen38_27b.note": "The only dense comparison point: 64 layers = 16 × (3 × Gated DeltaNet + 1 × GQA with output gate), the same attention structure as Qwen3.5-35B-A3B, with the FFN replaced by one 17408-wide SwiGLU per layer. The config is identical to Qwen3.5 / 3.6-27B. There are no experts, so no a2a and no expert load imbalance; every decode step reads all 27B weights.",
  "hw.B300": "B300 (HGX)",
  "topo.hgx8": "8 GPUs: a single HGX (NVLink domain 8)",
  "topo.nvl64": "64 GPUs: within one NVL72 rack (NVLink domain 72)"
});
