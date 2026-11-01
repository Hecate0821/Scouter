# Scouter：MoE 训推 Roofline 实验台

选一个模型和一种 GPU，Scouter 按 kernel 推算一步 decode、一次 chunked prefill 或一个后训练 step，再把每个 attention、MLP 模块放到 roofline 上。

**在线：** https://hecate0821.github.io/Scouter/ · English: [README.md](README.md)

> 所有数字都基于开源实现和公开数据推算，仅供学习参考，不要拿来做容量规划。

## 快速开始

```bash
python3 -m http.server 8000      # 或直接打开 index.html，不需要构建
```

```bash
node tools/scouter.js infer --model kimik3 --hw B200 --basis ideal --B 128 --pick stepT,bound
```

```bash
cd tests && npm install && npm test
```

## 三种口径

顶栏开关（命令行是 `--basis`）决定数字从哪来：

| 口径 | 含义 |
|---|---|
| **理想下限** | 这个配置不可能更快的时间：按峰值、没有固定开销、只算必须搬的流量，一步不短于关键路径和每种资源的总量。所有校准实测点都在它之上。约定参考 [k3-inference-calculator](https://github.com/ByronHsu/k3-inference-calculator)（Apache-2.0），推广到所有模型。 |
| **现实建模**（默认） | 尽量贴近现实：按 kernel 类别的可达效率、launch 和同步开销、期望碰到的 expert、负载不均、各引擎实际用的集合通信。 |
| **实测** | 只看数据点：公开数字（带出处链接）和匿名实测。 |

## 页面

| 页面 | 内容 |
|---|---|
| 总览 | 同一个模型的推理一步和后训练一步并排。 |
| 模型架构 | 层排布、单个模块的 roofline 和旋钮、机制示意图、所有模型对比。 |
| 后训练 | 一个 step（fwd / dgrad / wgrad、重算、通信、优化器、显存）、并行、低精度配方。 |
| 推理 | decode / prefill、部署与并行、低精度、投机解码、KV cache（每种层存多少、多轮 agent 会话在 HBM / 主机 / 远端三层里的前缀命中率、SGLang / vLLM / TRT-LLM / TokenSpeed 各怎么管），以及和 SGLang cookbook、自己的 profiler trace 对照。 |
| RL | 训推不一致、确定性 rollout、低精度 rollout 与 QAT。 |
| 附录 | 实测 kernel 对照屋顶、效率表、公开数据点、建模方法。 |

**模型：** Qwen3.5-35B-A3B、Kimi K3、GLM-5.3-Flash、Muse Spark Open、Qwen3.8-27B（dense 基线）。**GPU：** H100、H200、B200、B300、GB200、GB300。**拓扑：** HGX 单机 8 卡、NVL72 机柜（64 卡）。

地址带着上下文，比如 `?model=kimik3&hw=B200&src=ideal#infer/step`。界面文字都在 `locales/`（[docs/i18n.md](docs/i18n.md)）。

## 目录

```
index.html, css/, src/ui/   页面：只画数据平面返回的东西，不算物理量
src/engine/                 计算核心：纯函数，不碰 DOM、不出文字（docs/engine.md）
src/plane/scouter.js        Scouter.*：页面、命令行、测试共用的唯一接口（docs/api.md）
data/{public,anon}/*.json   数据点 → tools/build_db.py → data/db.json、src/data/db.js
locales/<lang>/             所有界面和计算核心的文字
tools/                      scouter.js（命令行）、build_db.py、导入脚本、reduce_trace.py
sim/                        PyTorch CPU 集合通信模拟器（sim/README.md）
tests/                      Node 检查：计算核心、数据平面、FLOPs、sim、i18n、控制台、回归
```

## 文档

- [AGENTS.md](AGENTS.md)：怎么运行、模拟、改计算（给人和编码 agent）。
- [docs/api.md](docs/api.md)：每个 `Scouter.*` 调用、选项和返回字段。
- [docs/modeling.md](docs/modeling.md)：每类 kernel 参考的实现、roofline 公式、校准、数据点格式（英文）。
- [docs/engine.md](docs/engine.md)、[docs/i18n.md](docs/i18n.md)：计算核心和文字的约定。

## 来源

以 SGLang、vLLM、TensorRT-LLM、Megatron-LM、torchtitan、TorchRL、slime、Miles、AReaL 的公开代码、PR、文档，以及公开博客和论文为准；每个模块页都链接到它参考的代码。机制示意图复现了 [Attention Lab](https://divagr.com/attention-lab/) 的可视化思路。
