# 计算核心

计算核心在 `src/engine/`：数据表加纯函数。它不碰 DOM、不出任何语言的文字、不读数据点库，所以浏览器里和 Node 里跑的是同一份代码。界面（`src/ui/`）只调用它、把结果画出来；需要的物理量都从这里取，不在界面里重算。

## 模块

加载顺序就是下表的顺序：`index.html` 的 `<script>` 和 `src/engine/node.js` 的 `FILES` 必须一致（`tests/engine.js` 检查）。每个文件定义一个全局对象。

| 文件 | 全局 | 内容 |
|---|---|---|
| `msg.js` | `Msg` | 消息：`Msg.of(key, 参数)`；`.invalid()` 标出“这个配置跑不起来”；`Msg.id` 比较两条消息 |
| `common.js` | `Common` | 数据类型与峰值（`dtB`、`peakOf`）、带上限的因果 key 数 `capKeys`（滑窗、DSA top-k）、roofline 公式（`compT`、`memT`、`ridge`、`attainable`、`linkRidge`、`roof`）、默认效率 `G0`、理想屋顶 `IDEAL`（效率 1、固定开销 0）和口径表 `BASIS`、模型 / GPU / 拓扑表（`MODELS`、`HW`、`TOPO`）、参数量 `derive`、层排布 `layerPlan`、链路与集合通信时间；按 kernel 类别的效率表 `KEFF`、`etaK`、`gK` |
| `par.js` | `Par` | CP / EP / FSDP 一层的通信与时间（`cpLayer`、`epLayer`、`fsdpDet`），一次集合通信的记录 `xc` |
| `infer.js` | `Infer` | 推理引擎画像 `ENGINES`、默认配置 `BASE`、`step` / `decode` / `prefill` / `memory`，闭环压测 `serve` |
| `train.js` | `Train` | 训练一步 `run`、默认配置 `BASE`、精度配方 `PREC` |
| `quant.js` | `Quant` | 低精度 GEMM、一层 MoE、训练的三个 GEMM、理想加速比 |
| `arch.js` | `Arch` | 架构页：模块划分 `present`、每个模块一层的 `measure`、`facts`、示意图和读数用的机制量 `explain` |
| `calib.js` | `Calib` | 和实测对照：每个数据集筛哪些点、实测是哪个字段、预测怎么算（`SETS`；`set` 字段指向别的数据集，比如 cookbook 的两个视图都读 `sglang_cookbook`），公开数据的 replay（`mokRatio`），cookbook 行到推理配置的映射 `serveCfg` |
| `kv.js` | `KV` | KV cache：每种层每个 token / 每个请求的字节（`perRank`、`bytesAt`），各引擎的前缀缓存画像 `ENGINES`（页大小、tail、线性状态只在 block 边界、装回是否逐层重叠），默认的 agent 会话 `BASE`，`plan`：会话工作集、HBM / 主机 / 远端的驻留比例（容量 / 工作集）、命中率和每轮 TTFT |
| `trace.js` | `Trace` | 实测 trace 对照：kernel 名按正则归类（`CAT`、`catOf`），本页算子归到同样的类别（`OPCAT`），`scan` / `kernelsOf` 汇总 trace（按 rank 平均、每条 stream 取独占时间、kernel 并集与首尾跨度、通信下限），`compare` 逐类比每步时间 |

## 接口约定

- **输入**：模型（`MODELS` 里的一项，或同样结构的对象）、GPU（`HW`）、拓扑（`TOPO`）、配置（从 `Infer.BASE` / `Train.BASE` 复制再覆盖）、效率参数 `g`（从 `Common.G0` 复制再覆盖）。
- **输出**：普通对象，能 JSON 序列化。时间是秒，字节和 FLOPs 是原始数，带宽是 GB/s。
- **文字**：名字、警告、说明都是 `Msg`。渲染由调用方决定：界面启动时设 `Msg.render = m => t(m.key, m.p)`；Node 里不设，渲染成 key。key 登记在 `locales/<语言>/engine_*.js`。输出里的普通字符串只能是标识符（key、枚举值）或引擎表里的名字（模型、GPU、引擎、kernel、格式），不能是要翻译的文字。
- **警告**：`r.warn` 是消息列表，`w.bad` 为真表示配置不可行。
- **口径**：`g` 用 `Common.BASIS.ideal`（理想下限，`lb = 1`：效率 1、没有开销、只算必要的流量，一步取关键路径和各资源总量里最大的）或 `Common.BASIS.modeled`（= `G0`，尽量贴近现实）；凡是两种口径算法不同的地方，计算核心里都看 `g.lb`；实测不是计算，走数据点库（`Scouter.points` / `kernels` / `calib`）。
- **按 kernel 类别的效率**：`Common.KEFF` 每一格是（类别、`c` 算力 / `m` 带宽、精度、SM 版本、η、出处 `ref`）。`gK(g, 类别, 精度, hw)` 把 `g.etaC` / `g.etaM` 按 η ÷ `G0` 的比例缩放（不超过 1）；`g.kEff` 为 0（理想屋顶就是 0）时不查表。moe 查不到用 gemm，SM103 查不到用 SM100。只用在推理的算子上，训练不变。`ref` 指向的数据点由 `tests/plane.js` 核对。
- **默认值归计算核心**：`Common.G0`、`Common.IDEAL`、`Infer.BASE`、`Train.BASE`、`Calib.BASE` 都是冻结的，调用方复制后再改。
- **不读数据点库**：`Calib.SETS` 只给筛选条件和预测函数，数据点由调用方传进来（界面从 `DB` 取，测试直接读 `data/db.json`）。

## 在 Node 里用

```js
const E = require("./src/engine/node.js")();
const c = Object.assign({}, E.Train.BASE, { fw: "torchtitan", N: 8, ep: 8, dispatcher: "standard", cpType: "allgather" });
const r = E.Train.run(E.MODELS.qwen35, E.HW.H200, E.TOPO.hgx8, c, E.G0);
console.log(r.stepT, r.mfu, r.comm.map(x => x.key));   // 一步的秒数、MFU、各类通信
```

计算核心外面还有一层数据平面入口 `src/plane/scouter.js`（`Scouter.*`）：把计算核心和数据点库包成一组调用（`catalog`、`roofline`、`infer`、`train`、`module`、`serve`、`sweep`、`sets`、`points`、`kernels`、`eff`、`calib`、`trace`），输入是 key 和覆盖项，输出能 JSON 序列化。界面的“命令行 / JSON”按钮、`tools/scouter.js` 和 agent 用的都是它，用法见 [`docs/api.md`](api.md)：

```
node tools/scouter.js catalog
node tools/scouter.js infer --model kimik3 --hw B200 --engine vllm --phase prefill --n 32768 --basis ideal
node tools/scouter.js train --model qwen35 --fw torchtitan --ep 8 --g.etaC 0.6
node tools/scouter.js module --model glm53f --phase decode --B 64 --lang zh
```

## 和真实并行实现对照

`sim/` 是一个独立的 Python 项目：用 PyTorch 自己的 FSDP2、DTensor TP 和 all-to-all EP（按 torchtitan 的包法）在 CPU gloo 进程上跑一个小模型的一个 micro-batch，把发出的每一次集合通信记成 JSON trace（`sim/traces/`）。它不知道 Scouter 的成本模型，Scouter 也不加载它；两边只通过 `sim/cases/*.json`（模型用本页的 `MODELS` 结构）和 trace 见面。`tests/sim.js` 把 `Train.run` 的通信清单和 trace 逐项对照（按组、操作、前向 / 反向，字节差 1% 以内、次数相同）。改了 `Par` / `Train` 里的集合通信就跑一遍；细节见 [`sim/README.md`](../sim/README.md)。

## 规则

1. 物理只写在这里。界面要的机制量（示意图里的行数、ridge、所需 batch）从 `Arch.explain`、`Common` 的 roofline 函数取。
2. 不出文字、不碰 DOM（`window`、`document`、`localStorage`……）、不读数据点库（`DB`）、不调 `t()`。
3. 语言相关的规则（比如名字怎么缩短）不写在代码里，写成 locale 文件里的函数值（`common.fmt.shortName`）；代码也不按显示出来的文字分支（`tests/i18n.js` 检查）。

## 检查

```
cd tests
npm run engine      # 每个模型 × GPU × 拓扑 × 推理引擎 / 训练框架 / 精度 / EP 后端 / 架构页 / 低精度 / 和实测对照：
                    # 不抛错、没有 NaN、能 JSON 序列化、文字都是消息；静态检查不碰界面、加载顺序一致
npm run flops       # 每个模型的矩阵乘 FLOPs 按配置独立重写一遍，和各算子的 FLOPs 对照（0.5% 以内）
npm run sim         # 训练通信和 PyTorch 实际发出的集合通信一致
npm test            # engine + plane + flops + sim + i18n + shell + regression
```
