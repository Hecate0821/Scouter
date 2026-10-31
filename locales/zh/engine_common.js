/* engine_common：src/engine/common.js 里的文字（模型说明、官方参数量、硬件 / 拓扑的名字） */
I18N.add("zh", "engine_common", {
  "model.qwen35.official": "35B 总 / 3B 激活",
  "model.qwen35.note": "40 层 = 10 ×（3 × Gated DeltaNet + 1 × 带输出门的 GQA）。每层都是 MoE：256 个 routed expert 选 8，外加 1 个带 sigmoid 门的 shared expert。",
  "model.kimik3.official": "2.8T 总 / 104B 激活",
  "model.kimik3.note": "69 层 KDA + 24 层带 sigmoid 输出门的 MLA（NoPE，但 KV 仍存 512+64 维）。routed expert 在 3584 维 latent 空间里算（router、latent 下投影、shared gate_up 是同一个 bf16 GEMM）；2 个 shared expert 合并成 F=6144，在 7168 维上算（vLLM 每卡常数 SI=768@TP8）。第 0 层是 dense FFN。另有 AttnRes 块残差（最多 8 个快照）。官方 checkpoint 只有 routed expert 是 MXFP4，其余 BF16。按这些字段算出的激活参数比官方的 104B 多几 B，差别来自 shared expert 的口径。",
  "model.glm53f.official": "320B 总 / 18B 激活",
  "model.glm53f.note": "34 层 KDA + 11 层 DSA（i%4==3）。DSA = NoPE MLA（qk_nope = v = 256，KV 512 维）+ KPool lightning indexer：每 4 个 token 池化成一个 key，选 top-512 个池（约 2048 个 token）。前 3 层 dense FFN。4 路 mHC 残差流。主仓库发 FP8 block 权重。",
  "model.musespark.name": "Muse Spark Open（816B-A42B）",
  "model.musespark.official": "816B 总 / 42B 激活",
  "model.musespark.note": "62 层全是 softmax GQA（128 个 q 头、16 个 KV 头、head dim 64，每头一个 sigmoid 门）：46 层滑窗 2048 + RoPE，16 层（i%4==1）看全上下文、不加位置编码，只有这 16 层的 KV 随上下文增长。每层都是 MoE：256 选 8，没有 shared expert；router 在 8192 维上读 fp32，expert 在 4096 维 latent 里算（pre / post_expert_proj 保持 BF16）。官方 NVFP4 checkpoint 只量化 expert（533 GB），BF16 版 1.64 TB。按这些字段算出的激活参数约 40B，官方口径 42B。",
  "model.qwen38_27b.name": "Qwen3.8-27B（dense）",
  "model.qwen38_27b.note": "唯一的 dense 对照：64 层 = 16 ×（3 × Gated DeltaNet + 1 × 带输出门的 GQA），和 Qwen3.5-35B-A3B 同一套注意力结构，FFN 换成每层一个 17408 宽的 SwiGLU。config 与 Qwen3.5 / 3.6-27B 完全相同。没有 expert，所以没有 a2a、没有 expert 负载不均，decode 每步要读全部 27B 权重。",
  "hw.B300": "B300（HGX）",
  "topo.hgx8": "8 卡：单台 HGX（NVLink 域 8）",
  "topo.nvl64": "64 卡：同一 NVL72 机柜（NVLink 域 72）"
});
