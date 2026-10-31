# Scouter：MoE 训推 Roofline 实验台

选一个模型和一种 GPU，Scouter 按 kernel 推算一步 decode、一次 chunked prefill 或一个后训练 step，再把每个 attention 模块、MLP 模块放到 roofline 上，看改架构会把它推到哪里。

**在线：** https://hecate0821.github.io/Scouter/ · English: [README.md](README.md)

> 所有数字都基于开源实现和公开数据推算，和实际可能有较大出入，仅供学习参考。

界面目前是中文；文字都在 `locales/` 里，加语言不用改代码，见 [docs/i18n.md](docs/i18n.md)。

## 页面

浏览器直接打开 `index.html`（`file://` 也能用，不需要构建），也可以用任意静态服务器托管这个目录。

界面按开发者控制台的样子排：

- **左侧栏**：按页分组，按 `/` 搜索。
- **顶栏**：模型、GPU、拓扑，以及**口径**开关：
  - **理想屋顶**：硬件 roofline，按峰值、没有固定开销；
  - **建模**：可达效率加固定开销，按开源引擎的默认做法（默认）；
  - **实测**：只看公开来源和匿名实测的数据点。
- **右侧参数栏**：这一页的控件。
- **命令行 / JSON 按钮**：显示这一页背后的 `Scouter.*` 调用，分别是 `tools/scouter.js` 命令和它的输出。
- **复制链接**：复制地址，地址里带着上下文，比如 `?model=kimik3&hw=B200&src=ideal#infer/step`。
- **浅色 / 深色**：切换主题。主题设置和配色跟 [hecate0821.github.io](https://hecate0821.github.io/) 一致；字体用系统无衬线字体，中文用黑体。

| 页面 | 内容 |
|---|---|
| 总览 | 同一个模型的推理一步 decode 和后训练一个 step 并排对比。 |
| 模型架构 | **模型总览**：层的排布（全注意力 / 滑窗 / 线性注意力，MoE / dense MLP）、每个模块一张卡片、参数量、KV 与状态缓存随上下文的变化。**一个模块**：机制示意图，架构旋钮（滑块），这个模块随 batch、上下文或序列长度变化在 roofline 上的轨迹（改过参数时原预设作为虚影），吞吐曲线，算子表，以及每条假设对应的开源代码。**跨模型对比**：所有模型的模块画在同一张 roofline 上，attention 和 MLP 的曲线，每层时间拆分。 |
| 后训练 | 一个 step（fwd / dgrad / wgrad、重算、通信、优化器、显存）；并行（一两层样例里每次通信的顺序和字节，FSDP、CP、EP 一维一维拆开）；低精度训练（只列 Megatron / TE / torchtitan 的默认配方）；和匿名实测对照。 |
| 推理 | 一步 decode / prefill；部署与并行（TP、DP-attention、EP、PD 分离的取舍）；低精度推理；投机解码（MTP、EAGLE-3、DFlash、DSpark）。 |
| RL | 训推不一致；确定性 rollout（batch invariance、可复现、训推逐位一致）；低精度 rollout 和权重同步；低精度训练与 QAT。 |
| 附录 | 实测 H200 kernel 和理想 / 建模屋顶的对照，公开数据与过去一年的发布；量化基础；建模口径和参考的开源实现。 |

地址是 `#页/子页/锚点`，比如 `#arch/module/lin`、`#infer/spec`、`#train/par/cp`；旧地址（`#par`、`#quant`、`#public`、`#appendix/models` 等）会跳到新位置。

**模型：**

| 模型 | 注意力 | MLP |
|---|---|---|
| Qwen3.5-35B-A3B | Gated DeltaNet + 带门 GQA，3 : 1 | MoE，带 shared expert |
| Kimi K3 | KDA + MLA | MoE，带 shared expert；第一层 dense |
| GLM-5.3-Flash | KDA + DSA（DeepSeek 稀疏注意力） | MoE，带 shared expert；前 3 层 dense |
| Muse Spark Open（816B-A42B） | GQA：46 层滑窗（2048）+ 16 层全局 | MoE |
| Qwen3.8-27B | Gated DeltaNet + 带门 GQA，3 : 1 | dense SwiGLU（dense 对照） |

**GPU：** H100、H200、B200、B300、GB200、GB300。

**拓扑：** 单机 8 卡（HGX），或 NVL72 里的 64 卡；所有并行组都在一个 NVLink 域内。

## 数字从哪来

以 SGLang、vLLM、TensorRT-LLM、Megatron-LM、torchtitan、TorchRL、slime、Miles、AReaL 的代码、PR 和文档为准，辅以公开的博客和论文。附录写了建模口径，每个模块页列出它的假设对应的源文件。

架构页没有自己的公式：它用推理页、训练页同一套逐 kernel 的模型，把 kernel 归到 attention / MLP 模块上，再除以该 kernel 覆盖的层数。

- **推理**用一个不偏向某个引擎的参考画像：只算各引擎都默认开的优化，单卡，不补齐 CUDA graph。
- **训练**用 Megatron-Core 的默认配置，单卡，不重算。

| 部分 | 默认假设 | 依据 |
|---|---|---|
| 全注意力（GQA、滑窗） | FlashAttention | flash-attention；SGLang `flashattention_backend.py`；vLLM v1 `flash_attn.py`；TensorRT-LLM `attention_backend` |
| MLA | decode 把上投影吸收进 query 和输出（FlashMLA）；prefill 把前缀的 KV 展开回每头的 K/V | FlashMLA；SGLang `deepseek_v2.py`；vLLM `mla_attention.py` |
| DSA | DeepSeek 开源的做法：FP8 lightning indexer（DeepGEMM）选出 top-k 个 key，再用 FlashMLA 稀疏注意力 | DeepSeek-V3.2-Exp、DeepGEMM、FlashMLA；SGLang `dsa_backend.py`；vLLM `flashmla_sparse.py`、`indexer.py` |
| 线性注意力（Gated DeltaNet、KDA） | flash-linear-attention：decode 用 fused recurrent kernel，prefill 和训练用 chunk kernel（chunk 64） | fla `ops/gated_delta_rule`、`ops/kda`；SGLang `hybrid_linear_attn_backend.py`；vLLM `gdn_attn.py` |
| MoE | 对路由到的 expert 做 fused grouped GEMM | SGLang `fused_moe_triton`；vLLM `fused_moe`；Megatron `moe/experts.py` |
| Prefill | chunked prefill：显存小于 160 GB 的卡每块 8192 个 token，否则 16384 | SGLang 启动参数；vLLM `arg_utils.py` |

roofline 模型：

- **每个 kernel 的时间：** max(FLOPs / (峰值 × 0.7), 字节 / (HBM 带宽 × 0.8), kernel 下限)。
- **ridge 点：** 峰值 × 0.7 / (HBM × 0.8)。
- **用哪个峰值：** 注意力核心按 BF16 峰值；GEMM 按权重精度的峰值，FP8、MXFP4 / NVFP4 分别对应 FP8、FP4 峰值。

机制示意图借鉴并重新实现了 [Attention Lab](https://divagr.com/attention-lab/) 的画法，每张图下方注明出处。

## 数据平面与 agent

所有数字来自同一个数据平面：`src/engine/` 里的模型、`src/data/` 里的数据点，以及唯一的入口 `src/plane/scouter.js`。页面只负责把它的输出画出来。同样的调用在 Node 和命令行里也能跑，agent 不开浏览器就能拿 Scouter 当参考计算器：

```
node tools/scouter.js infer --model kimik3 --hw B200 --basis ideal --B 128 --pick stepT,tComp
node tools/scouter.js sweep --run train --model qwen35 --over.ep 4,8 --pick stepT,mfu
node tools/scouter.js points --set inference_public --where.engine SGLang
```

每个调用、选项和返回字段见 [AGENTS.md](AGENTS.md)。

## 目录

```
index.html              页面骨架；按顺序加载下面的脚本（都是普通 <script>，共享全局作用域）
css/roofline.css
src/i18n/i18n.js        t(key, 参数)、data-i18n 属性、语言选择（?lang=、上次的选择、浏览器语言）
locales/zh/*.js         全部界面文字，一个命名空间一个文件；data.js 放数据文字的译文（docs/i18n.md）
src/engine/             计算模型：数据表加纯函数，不碰 DOM、不出文字、不读数据点库（docs/engine.md）
  msg.js                Msg.of(key, 参数)：计算核心返回消息，由调用方渲染成文字
  common.js             模型 / 硬件 / 拓扑表，derive()，roofline 公式，链路与集合通信时间
  infer.js              推理一步：SGLang / vLLM / TokenSpeed 的 kernel 序列、引擎中立的参考画像，
                        投机解码的 draft、验证、提交
  train.js              训练一步：Megatron-Core / torchtitan
  par.js                FSDP / CP / EP 的通信与重叠
  quant.js              低精度 GEMM、一层 MoE、训练三遍 GEMM
  arch.js               架构页：把一步里的 kernel 按层归到 attention / MLP 模块，以及示意图用的机制量
  calib.js              和实测对照：每个数据集复现哪些实测点、预测怎么算
  node.js               在 Node 里加载计算核心（一个 vm 上下文），给测试用
src/data/
  db.js                 生成文件：数据点库（const DB_RAW）
  query.js              DB.set / DB.where / DB.one / DB.meta
src/plane/
  scouter.js            数据平面的入口（Scouter.*）：计算核心 + 数据点库，JSON 进、JSON 出（AGENTS.md）
  node.js               在 Node 里加载数据平面，给测试和 tools/scouter.js 用
src/ui/                 各页的渲染；app.js 最后加载，负责路由和首次渲染
  theme.js shell.js     首次绘制前定主题；侧栏、顶栏、口径开关、命令行 / JSON、地址里的状态
  base.js state.js config.js   格式化与画图底座、全局状态、预设
  arch.js archviz.js    架构页；机制示意图
  overview.js train.js par.js calib.js comm.js infer.js tradeoff.js quant.js spec.js rl.js
  releases.js public.js dbview.js charts.js
data/
  anon/*.json           匿名数据点，每个文件一个数据集
  public/*.json         公开数据点（手工整理，带 src / url）
  db.json               生成文件：规范版本，每个点一行
tools/build_db.py       生成 data/db.json 和 src/data/db.js
tools/scouter.js        命令行调用数据平面，输出 JSON
sim/                    独立的 Python 项目：在 CPU gloo 进程上用 PyTorch FSDP2 / TP / EP 跑训练布局，
                        把每一次集合通信记成 JSON trace（sim/README.md）
tests/                  Node 检查：计算核心、数据平面、sim、i18n、外壳、回归和快照（jsdom）、渲染时间
AGENTS.md               给 agent 的数据平面说明：调用、选项、返回字段
docs/engine.md          计算核心的接口和规则
docs/i18n.md            界面文字与多语言
```

## 数据点库

```
{ schema: 1,
  sets:   { <id>: { title, origin: "anon" | "public", stage, hw, gpus, topo, model, conditions, dims: [...], metrics: { <名字>: { unit, label } } } },
  points: [ { id, set, ...维度, ...指标 } ] }
```

- **匿名数据点**（`origin: "anon"`）：自己测的数字。只保留测量条件（`conditions`）和数字，不带出处。每个数据集的点有同一组维度和指标，`id` 由维度拼成。
- **公开数据点**（`origin: "public"`）：公开的 PR、博客、论文、发布说明和文档里的数字，点上带 `src` / `url`。量化证据的权重 `w` 在构建时按“时效 × 硬件匹配 × 来源类型”算出，参照日期记在 `sets.quant_evidence.as_of`。
- **发布**（`releases`）：2025-11 到 2026-10 的 LMSYS 博客和 SGLang / vLLM / slime / Miles / AReaL 的 GitHub release，每条是 `releases:<项目>:<版本>`。页面里按项目和版本取链接，库里没有的条目会直接报错。

构建时会检查：

- 匿名点不能带 `src` / `url`；
- 匿名数据集的描述里不能出现出处类的词；
- id 不重复；
- 没有空数据集。

加公开数字：在 `data/public/` 对应的 JSON 里加一行（`id`、`date`、`src`、`url`、数字）。加匿名实测：在 `data/anon/` 加一个 JSON（标题、测量条件、维度、指标和点）。然后重新构建：

```
uv run tools/build_db.py                     # 重新生成 data/db.json 和 src/data/db.js
uv run tools/build_db.py --check             # 只检查生成结果是否最新
uv run tools/build_db.py --sqlite x.sqlite   # 另导出 SQLite（每个数据集一个视图），方便临时查询
```

页面里用 `DB.where("cp_full_model", {model: "qwen35", cp: 8})` 这样的查询取点。数据的文字写在数据文件里，译文放在 `locales/<语言>/data.js`（见 [docs/i18n.md](docs/i18n.md)）。

## 测试

```
cd tests && npm install
npm test               # engine + plane + sim + i18n + shell + regression
npm run engine         # 在 Node 里单独跑计算核心：每个模型 × GPU × 拓扑 × 引擎 / 框架 / 精度 / 模块；
                       # 不抛错、没有 NaN、文字都是消息；打印和实测对照的中位数
npm run plane          # 在 Node 里单独跑数据平面：两种口径下的每个调用，理想屋顶不比建模慢，命令行输出和接口一致
npm run shell          # 界面外壳：每页在三种口径下都能显示，页面 = 数据平面 = 打印出来的命令
npm run sim            # 训练通信和 sim/traces 里 PyTorch 实际发出的集合通信对照（不需要 Python）
npm run i18n           # 代码里没有界面文字（每一页做一遍伪本地化渲染来查）；用到的 key 都存在
npm run regression     # 所有模型 × 框架 × 预设跑一遍，再把每一页渲染一遍：报 NaN / undefined 和脚本错误
npm run timing         # 每一页的渲染时间
node snapshot.js ../index.html new.json base.json   # 导出每个视图的可见文字，和基线比对
```

改了 sim 的用例后重新生成 trace：`cd sim && uv run python -m scouter_sim.run`（见 [sim/README.md](sim/README.md)）。
