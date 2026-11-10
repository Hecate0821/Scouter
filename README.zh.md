# Scouter：大模型训推 Roofline 实验台

选一个模型（MoE 或 dense）和一种 GPU，Scouter 按 kernel 推算一步 decode、一次 chunked prefill 或一个后训练 step，再把每个 attention、MLP 模块放到 roofline 上。

**在线：** https://hecate0821.github.io/Scouter/ · English: [README.md](README.md)

有公开权重、HuggingArch 有 spec 的五个模型（Qwen3.5-35B-A3B、Kimi K3、GLM-5.3-Flash、DeepSeek-V4.1-Flash、Qwen3.8-27B），现实一侧的数字来自 [HuggingArch](https://github.com/shenh10/HuggingArch)（MIT，作者 Han Shen）的逐算子 roofline 仿真，用 `vendor/HuggingArch` 里的 fork 离线跑出来；Scouter 把它和自己的理想下限放在一起，逐项拆开差距（[docs/hga.md](docs/hga.md)）。

> 所有数字都基于开源实现和公开数据推算，仅供学习参考，不要拿来做容量规划。

## 快速开始

```bash
git clone --recursive https://github.com/Hecate0821/Scouter.git   # HuggingArch 的 fork 是 submodule；只看页面不需要它
python3 -m http.server 8000      # 或直接打开 index.html，不需要构建
```

```bash
node tools/scouter.js infer --model kimik3 --hw B200 --basis ideal --B 128 --pick stepT,bound
```

```bash
node tools/scouter.js hga --model qwen35 --engine sglang --sig tp8 --hw H200 --isl 8192 --b 64 --axis b
```

```bash
cd tools/hga && uv run python export.py --model qwen35      # 重新导出 HuggingArch 的仿真（需要 submodule）
```

```bash
cd tests && npm install && npm test
```

## 三种口径

顶栏开关（命令行是 `--basis`）决定数字从哪来：

| 口径 | 含义 |
|---|---|
| **理想下限** | 这个配置不可能更快的时间：按峰值、没有固定开销、只算必须搬的流量，一步不短于关键路径和每种资源的总量。所有校准实测点都在它之上。约定参考 [k3-inference-calculator](#致谢)，推广到所有模型。 |
| **HuggingArch**（默认，真实模型） | [HuggingArch](https://github.com/shenh10/HuggingArch) 的仿真：每个算子 = max(计算, HBM, 通信, launch 下限)，算子之间串行相加，集合通信按 α-β 和引擎的算法。按引擎签名（每个引擎 5 个，所有模型共用：SGLang、vLLM、Megatron-Core、torchtitan）× GPU 在上下文 × batch 网格上预先算好，页面实时插值。推理和训练的“一步”页第一块是它和理想下限的对照（瀑布、按类别的时间、资源、曲线）；HuggingArch 不覆盖的页面显示理想下限。 |
| **现实建模**（默认，改过的 / 自定义模型） | Scouter 自己的估算，只用在 HuggingArch 没有 spec 的地方：按 kernel 类别的可达效率、launch 和同步开销、期望碰到的 expert、负载不均、各引擎实际用的集合通信。 |
| **实测** | 只看数据点：公开数字（带出处链接）和匿名实测。 |

## 页面

| 页面 | 内容 |
|---|---|
| 总览 | 同一个模型的推理一步和后训练一步并排。 |
| 模型架构 | 层排布、单个模块的 roofline 和旋钮、机制示意图、所有模型对比。 |
| 后训练 | 一个 step（fwd / dgrad / wgrad、重算、通信、优化器、显存）、并行、低精度配方。 |
| 推理 | decode / prefill、部署与并行（attention TP × DP + 宽 EP、DCP、mixed chunked prefill 这几类配方做成预设，用算出来的吞吐、并发上限和 decode 停顿对比）、低精度、投机解码、KV cache（每种层存多少、多轮 agent 会话在 HBM / 主机 / 远端三层里的前缀命中率、SGLang / vLLM / TRT-LLM / TokenSpeed 各怎么管），以及和 SGLang cookbook、自己的 profiler trace 对照。 |
| RL | 训推不一致、确定性 rollout、低精度 rollout 与 QAT。 |
| 其他 | **网络**：每张卡从 HBM 到 spine 每一层的带宽，NVSwitch 和 NVLS，机内 PCIe 和 GPUDirect，只有 PCIe 的机器，rail、收敛比和 ECMP；在一套组网上训练和推理的时间花在哪、怕带宽还是怕延迟、NVLink 域多大合适、PD 分离和权重同步的传输，以及模型对公开的 nccl-tests 结果。**Profile**：推理一步按线上 trace 的分法拆开（kern 的 compute / comm / idle 和 CUPTI HES 那篇文章的几类）、DP 和 EP 负载不均、常开 trace 的开销，以及各个工具。 |
| 附录 | 实测 kernel 对照屋顶、效率表、公开数据点、建模方法。 |

**模型：** Qwen3.5-35B-A3B、Kimi K3、GLM-5.3-Flash、Muse Spark Open、DeepSeek-V4.1-Flash、Qwen3.8-27B（dense 基线）。**GPU：** H100、H200、B200、B300、GB200、GB300、RTX PRO 6000 Server。**拓扑：** HGX 单机 8 卡、NVL72 机柜（64 卡）、只有 PCIe 的 8 卡服务器。

地址带着上下文，比如 `?model=kimik3&hw=B200&src=ideal#infer/step`。界面文字都在 `locales/`（[docs/i18n.md](docs/i18n.md)）。

## 目录

```
index.html, css/, src/ui/   页面：只画数据平面返回的东西，不算物理量
src/engine/                 计算核心：纯函数，不碰 DOM、不出文字（docs/engine.md）
src/plane/scouter.js        Scouter.*：页面、命令行、测试共用的唯一接口（docs/api.md）
src/engine/hga.js           引擎签名、网格插值、理想 vs HuggingArch 对照（docs/hga.md）
src/data/hga/<model>.js     HuggingArch 的导出，按需加载（tools/hga/export.py 生成）
vendor/HuggingArch/         git submodule：shenh10/HuggingArch（MIT）的 fork，加了训练步
data/{public,anon}/*.json   数据点 → tools/build_db.py → data/db.json、src/data/db.js
locales/<lang>/             所有界面和计算核心的文字
tools/                      scouter.js（命令行）、build_db.py、导入脚本、reduce_trace.py、hga/（HuggingArch 导出，uv 项目）
sim/                        PyTorch CPU 集合通信模拟器（sim/README.md）
tests/                      Node 检查：计算核心、数据平面、FLOPs、sim、配置生效、i18n、控制台、回归
```

## 文档

- [AGENTS.md](AGENTS.md)：怎么运行、模拟、改计算（给人和编码 agent）。
- [docs/api.md](docs/api.md)：每个 `Scouter.*` 调用、选项和返回字段。
- [docs/hga.md](docs/hga.md)：HuggingArch 后端：签名、导出、数据格式、对照怎么拆（英文）。
- [docs/modeling.md](docs/modeling.md)：每类 kernel 参考的实现、roofline 公式、校准、数据点格式（英文）。
- [docs/engine.md](docs/engine.md)、[docs/i18n.md](docs/i18n.md)：计算核心和文字的约定。

## 致谢

Scouter 建立在这些开源项目公开的代码、PR、文档和基准之上；公开博客和论文则标在它们支撑的数字旁边。每个模块页和数据点都链接到具体出处。列在这里不代表这些项目审阅或认可过 Scouter 的数字。

| 方向 | 项目 |
|---|---|
| 推理引擎 | [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [TokenSpeed](https://github.com/lightseekorg/tokenspeed), [DeepInfer](https://github.com/deepseek-ai/DeepInfer) |
| 训练和 RL | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [torchtitan](https://github.com/pytorch/torchtitan), [TorchRL](https://github.com/pytorch/rl), [slime](https://github.com/THUDM/slime), [Miles](https://github.com/radixark/miles), [AReaL](https://github.com/areal-project/AReaL), [PyTorch](https://github.com/pytorch/pytorch) |
| Kernel | [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [CUTLASS](https://github.com/NVIDIA/cutlass), [TransformerEngine](https://github.com/NVIDIA/TransformerEngine), [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) |
| MoE 并行 | [DeepEP](https://github.com/deepseek-ai/DeepEP), [dist_moe](https://github.com/meta-pytorch/dist_moe), [mixture-of-kittens](https://github.com/cursor/mixture-of-kittens) |
| KV cache | [Dynamo](https://github.com/ai-dynamo/dynamo), [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [llm-d](https://github.com/llm-d/llm-d) |
| 低精度、投机解码 | [torchao](https://github.com/pytorch/ao), [SpecForge](https://github.com/sgl-project/SpecForge) |
| 基准和 profiling | [sgl-cookbook](https://github.com/sgl-project/sgl-cookbook), [vLLM recipes](https://github.com/vllm-project/recipes), [sglang-amd-cookbook](https://github.com/jhinpan/sglang-amd-cookbook), [lora_perf_lora_profile](https://github.com/yushengsu-thu/lora_perf_lora_profile), [llm_reproducibility](https://github.com/nanomaoli/llm_reproducibility), [torch_utils](https://github.com/fzyzcjy/torch_utils), [combine_traces](https://github.com/tlrmchlsmth/combine_traces)，[kern](https://github.com/pegainfer-project/kern)（Apache-2.0；常开的 HES trace、它的指标和性能图谱），[CUPTI](https://docs.nvidia.com/cupti/13.4.1/main/main.html) 和 [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 文档 |
| 模型结构 | [open-weights-carousel](https://github.com/NickLucche/open-weights-carousel), [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| 仿真后端 | [HuggingArch](https://github.com/shenh10/HuggingArch)（MIT，见下） |
| 理想下限 | [k3-inference-calculator](https://github.com/ByronHsu/k3-inference-calculator) (Apache-2.0) |
| 机制示意图 | [Attention Lab](https://divagr.com/attention-lab/) |
| 网络 | [NCCL](https://github.com/NVIDIA/nccl)（NVLS、CollNet / SHARP 和 rail 组网的文档），[zartbot/micro_arch](https://github.com/zartbot/micro_arch)（MIT；只有 PCIe 的机器的测量，整套报告每一页都注明由 AI 生成、未经作者审阅），[coreweave/nccl-tests](https://github.com/coreweave/nccl-tests) 和 [Crusoe 的 InfiniBand 验证文档](https://docs.crusoe.ai/networking/infiniband/validating-infiniband-performance)（公开的 nccl-tests 结果），[zartbot/gfd](https://github.com/zartbot/gfd)（MIT；每次拷贝调用的固定开销） |
| 论文和文章 | [CUPTI HES：Blackwell 上用硬件采集 kernel 时间戳](https://susun-blog.com/blog/cupti-hes/)（susun；Profile 页的拆法和开销数字），[Sarathi-Serve](https://arxiv.org/abs/2403.02310) 和 [DistServe](https://arxiv.org/abs/2401.09670)（chunked prefill，以及它和 decode 的互相干扰），[Helix](https://arxiv.org/abs/2507.07120)（KV 按序列切），[R3](https://arxiv.org/abs/2510.11370)（路由回放），[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)（Keep Routing、Keep Sampling Mask），[模型页](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)上的 DeepSeek-V4.1 技术报告，[fabric-lib](https://arxiv.org/abs/2510.27656)，[De Sensi 等，SC24](https://arxiv.org/abs/2408.14090) |
| 硬件规格 | [RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)，[RTX Blackwell 架构白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（稠密和稀疏峰值） |
| 延伸阅读 | [zartbot/blog](https://github.com/zartbot/blog)（没有许可证，这里只用思路和注明出处的数字） |

### HuggingArch

真实模型现实一侧的数字来自 [HuggingArch](https://github.com/shenh10/HuggingArch)，作者 Han Shen，按 MIT 许可使用（Copyright (c) 2026 shenh10，全文见 [LICENSE](https://github.com/shenh10/HuggingArch/blob/main/LICENSE)，`vendor/HuggingArch/LICENSE` 里也有一份）。Scouter 不重写它：`vendor/HuggingArch` 是固定在某个提交的 fork（git submodule），`tools/hga/export.py` 离线运行它，`src/data/hga/` 存它的输出。fork 只加 Scouter 需要的部分，代码里逐处标明：在 forward walk 上搭的训练步（`backend/training/`，见它的 [docs/training.md](https://github.com/Hecate0821/HuggingArch/blob/main/docs/training.md)）、让 serving 显存放不下的 forward 也保留算子行的选项（`keep_infeasible`、`walk_unloadable`）、权重放不下时 sweep 报出原因而不是出错。所以训练的数字是这个 fork 的，不是上游 HuggingArch 的。用在这里不代表 HuggingArch 的作者审阅过 Scouter 的数字。如果这个对照对你有帮助，请引用 HuggingArch：

```bibtex
@misc{shen2026huggingarch,
  author       = {Shen, Han},
  title        = {HuggingArch: Automating Model Architecture Analysis},
  year         = {2026},
  howpublished = {\url{https://shenhan.cc/blog/huggingarch}},
  note         = {Code: \url{https://github.com/shenh10/HuggingArch}}
}
```
