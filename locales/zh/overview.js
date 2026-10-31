/* overview：index.html 里的静态文字（由迁移脚本从 index.html 抽出） */
I18N.add("zh", "overview", {
  "rack.hint": "总览照搬推理页和后训练页的当前配置；要改配置，去对应的页。",
  "b1.h2": "五页各管什么",
  "b1.lede": "后训练、推理和 RL 用的是同一个模型，但发出的 kernel、并行方式和精度选择都不一样，所以分开建模，只共用模型结构、硬件表和效率系数。并行和低精度跟着场景走：训练的 FSDP / CP / EP 和推理的 TP / DP-attention / EP 是两回事，放在各自的页里。",
  "card.train_step.h4": "后训练",
  "card.train_step.p": "Megatron-Core / torchtitan 的一个 optimizer step：fwd / dgrad / wgrad、重算、TP / CP / EP / PP / FSDP 通信、优化器和显存。并行一节把 FSDP、CP、EP 一维一维拆开，低精度一节看训练三遍 GEMM，最后和匿名实测对照。",
  "card.infer_step.h4": "推理",
  "card.infer_step.p": "SGLang / vLLM / TokenSpeed 一步 decode 或 prefill 的 kernel 序列；部署一节比较 TP、DP-attention、EP 和 PD 分离的取舍；低精度一节看一个 GEMM 和一层 MoE；投机解码单独一节。",
  "card.rl_mismatch.p": "rollout 和训练是两套 kernel，log-prob 对不上：差异从哪来、怎么修（TIS / MIS、routing replay、逐位对齐）；确定性 rollout；低精度 rollout 和权重同步；低精度训练与 QAT。",
  "card.appendix_data.h4": "附录",
  "card.appendix_data.p": "公开数据点，过去一年 LMSYS 博客和 SGLang / vLLM / slime / Miles / AReaL 发布的摘要；量化基础；模型结构；建模口径和参考的开源实现。",
  "b2.h2": "当前配置：推理一步和训练一步",
  "b3.h2": "时间花在哪里",
  "b3.lede": "两边按同样的类别归并：全注意力、线性注意力、MoE / FFN、lm_head（训练含 CE）、没藏住的通信、其他（norm、launch、优化器以外的杂项）。推理是一次 decode，训练是每个 PP stage 的一个 micro-batch（fwd + bwd）。",
  "b3.h3": "推理：一步 decode",
  "b3.h3_2": "后训练：一个 micro-batch",
  "b4.h2": "读数",
});
