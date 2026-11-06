/* 公共层：模型结构、硬件、拓扑、参数推导、roofline 与集合通信的基础函数。推理和训练各自的算子图在 infer.js / train.js。 */
const Common = (function(){

/* 每个元素的字节数（含块 scale）：fp8 是 block FP8 权重，每 128×128 一个 fp32 scale；fp8a 是 block FP8 激活，每 1×128 一个 fp32；
   mxfp8 / mxfp4 每 32 个一个 e8m0；nvfp4 每 16 个一个 e4m3（第二级 fp32 scale 另算） */
const DT = { fp32:4, bf16:2, fp8:1+4/16384, fp8a:1+4/128, mxfp8:1+1/32, mxfp4:0.53125, nvfp4:0.5625 };
const dtB = d => DT[d] ?? 2;
const peakKey = d => d==="fp32"?"bf16":d==="fp4"||d==="mxfp4"||d==="nvfp4"?"fp4":d==="fp8"||d==="fp8a"||d==="mxfp8"?"fp8":"bf16";

/* 结构字段来自 HF config 与三个引擎 / 两个训练框架的模型文件（交叉核对见“口径与来源”）。 */
const MODELS = {
  qwen35: { name:"Qwen3.5-35B-A3B", short:"Qwen3.5-35B", family:"qwen", D:2048, L:40, V:248320, mtp:1,
    full:{ n:10, kind:"gqa", Hq:16, Hkv:2, dh:256, gate:true },
    lin:{ n:30, kind:"GDN", Hk:16, Hv:32, dk:128, dv:128, conv:4 },
    moe:{ E:256, k:8, F:512, Fs:512, sharedGate:true, Dlat:0 }, dense:{ n:0, F:0 },
    wDefault:"bf16", official:Msg.of("engine_common.model.qwen35.official"),
    note:Msg.of("engine_common.model.qwen35.note") },
  kimik3: { name:"Kimi K3", short:"Kimi K3", family:"kimi", D:7168, L:93, V:163840, mtp:0,
    full:{ n:24, kind:"mla", H:96, qLora:1536, kvLora:512, nope:128, rope:64, vh:128, gate:true, nope_only:true },
    lin:{ n:69, kind:"KDA", Hk:96, Hv:96, dk:128, dv:128, conv:4 },
    moe:{ E:896, k:16, F:3072, Fs:6144, sharedGate:false, Dlat:3584 }, dense:{ n:1, F:33792 },
    attnRes:8, wDefault:"mxfp4", official:Msg.of("engine_common.model.kimik3.official"),
    note:Msg.of("engine_common.model.kimik3.note") },
  glm53f: { name:"GLM-5.3-Flash", short:"GLM-5.3-Flash", family:"glm", D:4096, L:45, V:154880, mtp:1,
    full:{ n:11, kind:"dsa", H:64, qLora:1536, kvLora:512, nope:256, rope:0, vh:256, gate:false, topk:2048, idxH:32, idxD:128, kpool:4 },
    lin:{ n:34, kind:"KDA", Hk:64, Hv:64, dk:128, dv:128, conv:4 },
    moe:{ E:288, k:8, F:2048, Fs:2048, sharedGate:false, Dlat:0 }, dense:{ n:3, F:12288 },
    mhc:4, wDefault:"fp8", official:Msg.of("engine_common.model.glm53f.official"),
    note:Msg.of("engine_common.model.glm53f.note") },
  musespark: { name:Msg.of("engine_common.model.musespark.name"), short:"Muse Spark", family:"muse", D:8192, L:62, V:202048, mtp:0,
    full:{ n:62, kind:"gqa", Hq:128, Hkv:16, dh:64, gate:true, gateHead:true, swa:46, W:2048, gEvery:4, gOff:1 },
    lin:{ n:0, kind:"GDN", Hk:0, Hv:0, dk:128, dv:128, conv:4 },
    moe:{ E:256, k:8, F:4096, Fs:0, sharedGate:false, Dlat:4096 }, dense:{ n:0, F:0 },
    wDefault:"nvfp4", official:Msg.of("engine_common.model.musespark.official"),
    note:Msg.of("engine_common.model.musespark.note") },
  /* DeepSeek-V4.1-Flash（HF config.json 的 text_config；结构见 inference/model.py）。kind "csa"：每层一个 512 维 latent 同时当 K 和 V（MQA，
     64 个 query 头），每层自己的滑窗 KV（W = 128）；压缩的 latent 只在 kv source 层产生（cr = compress_ratios 的前 40 项，2 是两个 token
     压成一个），后面同 ratio 的层复用；index source 层自己打分选 top-k，第 20 层先挑 2048 个 8 位置的块，之后的 index 层只在块里重打分。
     输出是分组的低秩投影：o_groups 组各 (H·dh / oG) → oLora（bf16），再 oG·oLora → D。Engram：两层 n-gram 哈希表（fp8，按行切），
     每 token 每层查 (ngram − 1)·heads 行。DSpark：自带的 3 层草稿模型（滑窗 attention + 128 选 3 的 MoE），读目标第 37–39 层的特征 */
  dsv41f: { name:"DeepSeek-V4.1-Flash", short:"DSV4.1-Flash", family:"deepseek", D:5120, L:40, V:129280, mtp:0,
    full:{ n:40, kind:"csa", H:64, qLora:1280, dh:512, rope:64, oG:8, oLora:1024, oDt:"bf16", W:128, swaDt:"mxfp8",
      cr:[0,0, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], kvSrc:[2,8,14,20], idxSrc:[2,8,14,20,24,28,32,36],
      topk:512, idxH:32, idxD:128, idxDt:"mxfp4", cand:{ src:20, blocks:2048, bs:8 }, cmpDt:"fp32" },
    lin:{ n:0, kind:"GDN", Hk:0, Hv:0, dk:128, dv:128, conv:4 },
    moe:{ E:384, k:6, F:2304, Fs:2304, sharedGate:false, Dlat:0 }, dense:{ n:0, F:0 },
    mhc:4, engram:{ layers:[1,14], rows:[384006168,384016682], ngram:4, heads:8, hd:256 },
    drafter:{ kind:"dspark", layers:3, feats:3, block:5, rank:256, moe:{ E:128, k:3, F:2304, Fs:2304 } },
    wDefault:"mxfp4", dDefault:"fp8", kvDefault:"nvfp4", official:Msg.of("engine_common.model.dsv41f.official"),
    note:Msg.of("engine_common.model.dsv41f.note") },
  qwen38_27b: { name:Msg.of("engine_common.model.qwen38_27b.name"), short:"Qwen3.8-27B", family:"qwen", D:5120, L:64, V:248320, mtp:1,
    full:{ n:16, kind:"gqa", Hq:24, Hkv:4, dh:256, gate:true },
    lin:{ n:48, kind:"GDN", Hk:16, Hv:48, dk:128, dv:128, conv:4 },
    moe:{ E:0, k:0, F:0, Fs:0, sharedGate:false, Dlat:0 }, dense:{ n:64, F:17408 },
    wDefault:"bf16", official:Msg.of("engine_common.model.qwen38_27b.official"),
    note:Msg.of("engine_common.model.qwen38_27b.note") }
};

/* 规格表 dense 数值（TFLOP/s、GB/s、GB）；nvl / ib / host 为单 GPU 单向带宽。
   mem 取单卡 HBM 规格：B300 / GB300 按 Blackwell Ultra 的 288 GB；NVIDIA 的整机页面写 HGX B300 每 8 卡 2.1 TB（每卡约 262 GB）、
   GB300 NVL72 每 72 卡 20 TB（每卡约 278 GB），按整机数字算显存会更紧 */
const HW = {
  H100:{ name:"H100 SXM", sm:90, bf16:989, fp8:1979, fp4:1979, hbm:3350, mem:80, nvl:450, ib:50, host:25 },
  H200:{ name:"H200", sm:90, bf16:989, fp8:1979, fp4:1979, hbm:4800, mem:141, nvl:450, ib:50, host:25 },
  B200:{ name:"B200", sm:100, bf16:2250, fp8:4500, fp4:9000, hbm:8000, mem:180, nvl:900, ib:50, host:25 },
  B300:{ name:Msg.of("engine_common.hw.B300"), sm:103, bf16:2250, fp8:4500, fp4:13500, hbm:8000, mem:288, nvl:900, ib:100, host:25 },
  GB200:{ name:"GB200", grace:true, sm:100, bf16:2500, fp8:5000, fp4:10000, hbm:8000, mem:186, nvl:900, ib:50, host:200 },
  GB300:{ name:"GB300", grace:true, sm:103, bf16:2500, fp8:5000, fp4:15000, hbm:8000, mem:288, nvl:900, ib:100, host:200 },
  /* RTX PRO 6000 Blackwell Server Edition（SM120，GDDR7，没有 NVLink）。nvidia.com 规格页写 FP4 4 PF / FP8 2 PF / BF16 1 PF，没标稀疏；
     同一代 RTX Blackwell 架构白皮书的规格表把带稀疏的数写在斜杠后面（RTX 5090 FP4 1676 / 3352），按 188 SM、约 2.6 GHz 折算稠密 FP4 约 2 PF，
     所以这里按"规格页是带稀疏的数"减半。hbm 字段在这里是 GDDR7 带宽 1597 GB/s；nvl 是卡间 PCIe Gen5 x16 点对点（每方向 64 GB/s 规范速率）；
     ib 按每卡 400G 网卡。SM120 没有 tcgen05 / TMEM：只给 SM100 写的 kernel（trtllm-gen、FlashMLA SM100、DeepGEMM、V4.1 的 FP4 KV 布局）都不可用，见 sm10x */
  RTXP6000:{ name:"RTX PRO 6000 Server", pcie:true, sm:120, bf16:500, fp8:1000, fp4:2000, hbm:1597, mem:96, nvl:64, ib:50, host:25 }
};
const TOPO = {
  hgx8:{ name:Msg.of("engine_common.topo.hgx8"), gpus:8, domain:8, node:8 },
  nvl64:{ name:Msg.of("engine_common.topo.nvl64"), gpus:64, domain:72, node:4 },
  pcie8:{ name:Msg.of("engine_common.topo.pcie8"), gpus:8, domain:8, node:8, pcieOnly:true }
};   // 所有并行组都在一台机器 / 一个机柜内。跨机走 IB 的公式（link、Par.epLayer）还在，集群页把规模放大时会用到
/* 每种 GPU 默认放在哪种拓扑：GB 系列在 NVL72 机柜，PCIe 卡在只有 PCIe 的 8 卡服务器，其余在 HGX。界面和数据平面共用这条规则 */
const homeTopo = k => HW[k].grace ? "nvl64" : HW[k].pcie ? "pcie8" : "hgx8";
/* 数据中心 Blackwell（SM100 / SM103，tcgen05 + TMEM）。只看格式支持（FP4 / MXFP8 的块缩放 MMA）时用 sm >= 100，SM120 也有；
   只为 SM100 写的 kernel 用这个 */
const sm10x = hw => hw.sm>=100 && hw.sm<110;
const peakOf = (hw,d) => hw[peakKey(d)]*1e12;
/* roofline 的几个量（界面画图和标注也用这些，不另写公式）。peak 是 FLOP/s（peakOf 的结果），η 来自 g */
const compT = (flops, peak, g) => flops/(peak*g.etaC);                          // 按可达算力算的时间
const memT = (bytes, hw, g) => bytes/(hw.hbm*1e9*g.etaM);                       // 按可达 HBM 带宽算的时间
const ridge = (peak, hw, g) => peak*g.etaC/(hw.hbm*1e9*g.etaM);                 // 屋顶拐点（FLOP/B）
const attainable = (I, peak, hw, g) => Math.min(peak*g.etaC, hw.hbm*1e9*g.etaM*I);   // 强度 I 下能达到的 FLOP/s
const linkRidge = (peak, gbs, g) => peak*g.etaC/(gbs*1e9*g.etaAg);             // 每通信 1 字节要配多少 FLOPs 才藏得住（链路 GB/s）

/* 默认的效率系数与固定开销（每页共用，界面上可以改）：η 是可达峰值的比例，α 是集合通信延迟（µs），其余是 kernel / launch / 同步开销（µs） */
const freeze = o => { for(const v of Object.values(o)) if(v && typeof v==="object") freeze(v); return Object.freeze(o); };
/* gapUs：CUDA graph 里前一个 kernel 结束到下一个开始的间隔（vLLM PR 64846、flashinfer PR 8135 在 H200 上量到约 0.45 µs；推理配置 pdl = true 时去掉）。 */
const G0 = freeze({ etaC:0.7, etaM:0.8, etaN:0.7, etaAg:0.78, alphaNv:8, alphaIb:20, kernelUs:2, launchUs:4, graphLaunchUs:15, gapUs:0.45, schedUs:0, dpSyncUs:100, a2aLLus:25, deepepUs:10, barrierUs:5,
  etaTbo:0.5, kEff:1, lb:0 });   // etaTbo：TBO 时一个 micro-batch 的 a2a 能藏在另一个计算后面的比例（每层两次 a2a 各对上一段计算，默认按一半）
/* 理想下限：这个配置（模型、并行切法、批大小、引擎的 kernel 序列和重叠方式）不可能更快的时间。可达比例都是 1、固定开销和延迟都是 0，
   另外 lb = 1 换成下限的口径（计算核心里凡是 g.lb 的地方）：
   - 只算必要的量：不补齐 CUDA graph / FP4 hidden，routed expert 只读一定会碰到的最少 expert（TP 切时 top-k 个切片，按 expert 切时最忙的
     rank 上 ceil(k / EP) 个），负载完全均衡（EP 不均、CP 不均都按 1）；
   - 集合通信按信息量的下限：all-reduce 每个 rank 至少收一份完整结果（逻辑字节 1×，交换机内归约 NVLS / SHARP 可以做到），
     all-gather / reduce-scatter 是 (n − 1) / n，all-to-all 是发出去的 (n − 1) / n；
   - Ulysses 的 all-to-all 和 Ring / AllGather 一样按能和分块计算重叠算（建模口径按全部露出）；
   - 一步的时间除了按依赖和重叠排出来的关键路径，还要满足资源守恒：所有算子的计算时间之和、HBM 时间之和、通信时间之和都不能被重叠藏掉
     （两个重叠的分支不能各自吃满同一份算力或带宽）。
   这些口径参考 ByronHsu/k3-inference-calculator（Apache-2.0）对 Kimi K3 的做法，这里推广到所有模型；没有搬它的代码。
   BASIS 是界面和命令行的“口径”开关：ideal = 理想下限，modeled = 尽量贴近现实的建模（上面的默认效率和开销、期望的 expert 覆盖、
   引擎实际用的通信算法）。实测走数据点库，不在计算核心里 */
const IDEAL = freeze(Object.assign(Object.fromEntries(Object.keys(G0).map(k => [k, k.startsWith("eta") ? 1 : 0])), { lb:1 }));
const BASIS = freeze({ ideal: IDEAL, modeled: G0 });

/* 按 kernel 类别和 GPU 架构的效率表（g.kEff = 1 时生效；ideal 口径 kEff = 0）。每一格是一个公开或匿名实测点：达到的 TFLOP/s（或 GB/s）÷ 规格峰值，
   出处在数据点库 kernel_public（ref = 点的 id，tests/plane.js 核对数值）。η_c 是 Hopper BF16 大 GEMM 的参考值：表里的格子按 η / G0 的比例乘在 g 上，
   所以改 η_c / η_m 时整张表按比例跟着动；查不到的类别、精度或架构就用 g 本身。
   cls：gemm = 带权重的稠密线性层，attn = attention 核心；kind：c = 算力（乘 etaC），m = 带宽（乘 etaM）；dt 是 peakKey；sm 是 hw.sm，103 查不到时退到 100。 */
const KEFF = freeze([
  { cls:"gemm", kind:"c", dt:"bf16", sm:90,  eta:0.69, ref:"h200_kernels:bf16_gemm/8192/8192/8192/None/None" },   // 匿名实测 678 / 989
  { cls:"gemm", kind:"c", dt:"fp8",  sm:90,  eta:0.78, ref:"kernel_public:deepgemm-h800-fp8" },                   // DeepGEMM 1550 / 1979
  { cls:"attn", kind:"c", dt:"bf16", sm:90,  eta:0.75, ref:"kernel_public:fa3-h100-fp16" },                       // FA3 740 / 989
  { cls:"attn", kind:"m", dt:"bf16", sm:90,  eta:0.90, ref:"kernel_public:flashmla-h800-decode-mem" },            // FlashMLA decode 3000 / 3350 GB/s
  { cls:"gemm", kind:"c", dt:"bf16", sm:100, eta:0.67, ref:"kernel_public:cublas-b200-bf16-4096" },               // cuBLAS 1507 / 2250
  { cls:"attn", kind:"c", dt:"bf16", sm:100, eta:0.71, ref:"kernel_public:fa4-b200-bf16" },                       // FA4 1605 / 2250
  { cls:"gemm", kind:"c", dt:"bf16", sm:103, eta:0.72, ref:"kernel_public:cutlass-sm103-bf16-8192" },             // CUTLASS 1618 / 2250
  { cls:"gemm", kind:"c", dt:"fp4",  sm:103, eta:0.52, ref:"kernel_public:cutlass-sm103-nvfp4-8192" },            // CUTLASS NVFP4 7011 / 13500
  { cls:"attn", kind:"c", dt:"bf16", sm:103, eta:0.74, ref:"kernel_public:flashinfer-b300-fmha-hd128" }]);        // FMHA hd128 1658 / 2250
/* moe（grouped GEMM）没有自己的格子时用 gemm 的：每个 expert 的行数够多时，grouped GEMM 和稠密 GEMM 是同一类 kernel */
function etaK(cls, kind, dt, hw){
  const at = (c, sm) => KEFF.find(r => r.cls===c && r.kind===kind && r.dt===dt && r.sm===sm);
  for(const c of cls==="moe" ? ["moe", "gemm"] : [cls]){ const r = at(c, hw.sm) || (hw.sm===103 ? at(c, 100) : null); if(r) return r.eta; }
  return null;
}
/* 一个 kernel 实际用的效率：g 按表里这一格缩放（上限 1） */
function gK(g, cls, dt, hw){
  if(!g.kEff || !cls) return g;
  const c = etaK(cls, "c", dt, hw), m = etaK(cls, "m", dt, hw);
  if(c==null && m==null) return g;
  return Object.assign({}, g, c==null ? {} : { etaC: Math.min(1, g.etaC*c/G0.etaC) }, m==null ? {} : { etaM: Math.min(1, g.etaM*m/G0.etaM) });
}

/* GQA 的 q + 输出门宽度：门按元素（Qwen）是 Hq·dh，按头（Muse Spark）是 Hq */
const qoW = f => f.Hq*f.dh + (f.gate ? (f.gateHead ? f.Hq : f.Hq*f.dh) : 0);
/* 滑窗层每个 query 平均看到的 key 数（整段长 S 的 causal）；全注意力是 S/2 */
/* 带上限的 causal attention 里每个 query 平均看到的 key 数：前面已有 ctx 个 token，本块 C 个 query，每个 query 看 min(K, 位置) 个
   （K 是滑窗宽度或 DSA 的 top-k）。块跨过 K 时不能用 min(ctx + C/2, K)：那样把还没到 K 的 query 也算成 K（k 窗口 2048、
   C = 4096 时多算 33%；思路见 MatthewBonanni/attn-viz costs.js）。swaKeys 是整段从头算（ctx = 0）的情形，训练用 */
const capKeys = (ctx, C, K) => ctx>=K ? K : ctx+C<=K ? ctx+C/2 : ((K*K - ctx*ctx)/2 + (ctx+C-K)*K)/C;
const swaKeys = (S, W) => capKeys(0, S, W);

/* CSA（DeepSeek-V4.1）每层的角色，全部从 config 推出来：m = compress ratio（0 = 只有滑窗）；kv = kv source（自己压缩 KV，也存 index K）；
   idx = index source（自己打分选 top-k，其余层复用前面最近的那份）；cand = 候选块的来源层；useCand = 只在候选块里重打分。
   f.from 是这段模型从第几层开始（prefill 的 bounded replay 把模型切成前后两段，见 Infer.step） */
const csaPlan = f => Array.from({ length:f.n }, (_, j) => { const i = (f.from||0)+j, idx = f.idxSrc.includes(i);
  return { i, m:f.cr[i]||0, kv:f.kvSrc.includes(i), idx, cand:!!f.cand && i===f.cand.src, useCand:!!f.cand && idx && i>f.cand.src }; });
/* CSA 一份 512 维 latent 的字节：fp8 按 V4 的布局（448 fp8 nope + 64 bf16 rope + 8 B scale，SGLang dsv4/kv_layout.py 的 V4），
   Hopper 上 SGLang 只有这一种；nvfp4 是 V41_FP4（256 + 32 个 e4m3 scale），mxfp8 是 V41（512 + 16），其余按 DT */
const csaLatB = (f, dt) => dt==="fp8" ? (f.dh-f.rope) + f.rope*2 + 8 : f.dh*dtB(dt);
/* 每 token 的 KV（整个模型，单个 attention rank）：压缩 KV 和 index K 只在 kv source 层存，每 m 个 token 一份；滑窗 KV 每层都有，
   每个请求最多 W 个 token。kvDt = nvfp4 时滑窗用 f.swaDt（SGLang 的 main = V41 fp8、extra = V41_FP4 这一对），其余两者同精度 */
function csaKv(m, kvDt){
  const f = m.full, P = csaPlan(Object.assign({}, f, { from:0, n:f.cr.length }));
  const lat = csaLatB(f, kvDt), idx = f.idxD*dtB(f.idxDt), swa = csaLatB(f, kvDt==="nvfp4" ? f.swaDt : kvDt);
  const src = P.filter(x => x.kv);
  return { lat, idx, swa, main:src.reduce((s,x)=>s+lat/x.m, 0), index:src.reduce((s,x)=>s+idx/x.m, 0), swaTok:P.length*swa, W:f.W, src:src.length, layers:P.length };
}

/* 参数量：按层类型拆开，推理和训练都从这里取 */
function derive(m){
  const D=m.D, f=m.full, l=m.lin, e=m.moe;
  const r = { nMoe: m.L-m.dense.n, De: e.Dlat||D };
  if(f.n>0 && f.kind==="csa"){
    const P = csaPlan(f), Hd = f.H*f.dh;
    r.pAttnRep = D*(f.qLora + f.dh);                                 // wq_a + wkv（滑窗 KV），各 TP rank 复制
    r.pQKV = f.qLora*Hd;                                             // wq_b
    r.pOa = Hd*f.oLora;                                              // wo_a：oG 组，每组 (H·dh / oG) → oLora
    r.pO = r.pOa + f.oG*f.oLora*D;                                   // + wo_b
    r.nIdx = P.filter(x=>x.idx).length; r.nSrc = P.filter(x=>x.kv).length;
    r.pIdx1 = f.qLora*f.idxH*f.idxD + D*f.idxH;                      // indexer 的 q（wq_b）+ weights_proj
    r.pCmp = P.filter(x=>x.kv).reduce((s,x)=>s + D*f.dh*(x.m>1?2:1) + f.dh*f.idxD, 0);   // compressor 的 wkv（m > 1 再加 wgate）+ index K 的 wk
    r.pIdx = (r.nIdx*r.pIdx1 + r.pCmp)/f.n;                          // 摊到每层
    r.kvTokElems = f.dh; r.Hattn=f.H; r.dqk=f.dh; r.dv=f.dh; r.csa = P;
  } else if(f.n>0){
    if(f.kind==="gqa"){
      r.pQKV = D*(qoW(f) + 2*f.Hkv*f.dh); r.pO = f.Hq*f.dh*D; r.pAttnRep = 0;
      r.kvTokElems = 2*f.Hkv*f.dh; r.Hattn=f.Hq; r.dqk=f.dh; r.dv=f.dh;
    } else {
      const dq=f.nope+f.rope;
      r.pAttnRep = D*(f.qLora + f.kvLora + f.rope);                   // fused_qkv_a，各 TP rank 复制
      r.pQKV = f.qLora*f.H*dq + f.kvLora*f.H*(f.nope+f.vh) + (f.gate? D*f.H*f.vh : 0);
      r.pO = f.H*f.vh*D;
      if(f.kind==="dsa") r.pIdx = f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH; else r.pIdx=0;
      r.kvTokElems = f.kvLora + f.rope;                              // K3 NoPE 仍存 64 维槽位（576）；GLM rope=0（512）
      r.Hattn=f.H; r.dqk=dq; r.dv=f.vh;
    }
  } else { r.pQKV=0; r.pO=0; r.pAttnRep=0; r.kvTokElems=0; r.pIdx=0; }
  r.pIdx = r.pIdx||0;
  r.pAttn = r.pQKV + r.pO + r.pAttnRep + r.pIdx;
  if(l.n>0){
    // in_proj：q,k,v,输出门（满秩 H·dv）+ 每头的 β（GDN 另有 α）。KDA 没有 α，逐通道的 f 门是低秩的：
    // f_a D→dk（TP 各 rank 复制）再 f_b dk→H·dk（SGLang kimi_k3.py 的 f_a_proj / f_b_proj；k3-inference-calculator 同样这么算）
    const inW = 2*l.Hk*l.dk + 2*l.Hv*l.dv + (l.kind==="KDA" ? l.Hv : 2*l.Hv);
    r.linIn = inW; r.linFa = l.kind==="KDA" ? D*l.dk : 0; r.linFb = l.kind==="KDA" ? l.dk*l.Hk*l.dk : 0;
    r.pLin = D*inW + r.linFa + r.linFb + l.Hv*l.dv*D;
    r.linState = l.Hv*l.dk*l.dv; r.linConvDim = 2*l.Hk*l.dk + l.Hv*l.dv;
  } else { r.linIn=0; r.linFa=0; r.linFb=0; r.pLin=0; r.linState=0; r.linConvDim=0; }
  r.pExpert = 3*r.De*e.F;
  r.pRouted = e.E*r.pExpert;
  r.pShared = 3*D*e.Fs + (e.sharedGate? D : 0);
  r.pRouter = D*e.E;
  r.pLat = e.Dlat ? 2*D*e.Dlat : 0;
  r.pDense = 3*D*m.dense.F;
  r.pEmb = m.V*D; r.pHead = m.V*D;
  // Engram：哈希表（每层一张，rows 行 × hd）和每层的 wkv（(ngram − 1)·heads·hd → D·(hc + 1)）+ 两个 hc×D 的门向量
  const eg = m.engram, hc = m.mhc||1;
  r.nEng = eg ? eg.layers.length : 0; r.engCols = eg ? (eg.ngram-1)*eg.heads : 0;
  r.pEngTab = eg ? eg.rows.reduce((s,x)=>s+x, 0)*eg.hd : 0;
  r.pEngW1 = eg ? r.engCols*eg.hd*D*(hc+1) + 2*hc*D : 0;
  r.pEngW = r.nEng*r.pEngW1;
  r.total = f.n*r.pAttn + l.n*r.pLin + r.nMoe*(r.pRouted+r.pShared+r.pRouter+r.pLat) + m.dense.n*r.pDense + r.pEmb + r.pHead + r.pEngTab + r.pEngW;
  r.active = f.n*r.pAttn + l.n*r.pLin + r.nMoe*(e.k*r.pExpert+r.pShared+r.pRouter+r.pLat) + m.dense.n*r.pDense + r.pHead + r.pEngW;
  r.pExpertsAll = r.nMoe*r.pRouted;
  r.pDenseAll = r.total - r.pExpertsAll;
  return r;
}

/* 每层的类型：attn = full（全注意力）/ swa（滑窗）/ lin（线性注意力），mlp = moe / dense。
   全注意力层均匀插在线性层之间（第 every−1、2·every−1… 层，尾部不够时补齐）；滑窗层按 i % gEvery ≠ gOff 判断；前 dense.n 层是 dense FFN。 */
function layerPlan(m){
  if(m.full.kind==="csa") return csaPlan(m.full).map((x, j) => ({ attn: x.m>0 ? "full" : "swa", mlp: j>=m.dense.n ? "moe" : "dense" }));
  const f=m.full, every = m.lin.n>0 ? Math.round(m.L/Math.max(1,f.n)) : 1, sw=f.swa||0;
  let left=f.n; const out=[];
  for(let i=0;i<m.L;i++){
    const full = m.lin.n===0 || ((i+1)%every===0 && left>0) || (m.L-i<=left); if(full) left--;
    const swa = full && sw>0 && i%(f.gEvery||4)!==(f.gOff||0);
    out.push({ attn: !full ? "lin" : swa ? "swa" : "full", mlp: i>=m.dense.n ? "moe" : "dense" });
  }
  return out;
}

/* ECMP：F 条流按哈希落到 U 条上行链路，最忙那条的期望流数。固定种子的蒙特卡洛，结果可复现；流很多时用正态近似 */
const ECMP_MEMO = new Map();
function ecmpMax(F, U){
  F = Math.max(1, Math.round(F)); U = Math.max(1, Math.round(U));
  if(U===1) return F;
  const key = F+"/"+U; if(ECMP_MEMO.has(key)) return ECMP_MEMO.get(key);
  let v;
  if(F > 20000) v = F/U + Math.sqrt(2*(F/U)*Math.log(U));
  else {
    let s = 0x9e3779b9 ^ (F*31 + U), tot = 0; const trials = Math.max(64, Math.min(400, Math.floor(200000/F))), bins = new Uint32Array(U);
    const rnd = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s>>>15, 1|s); t = t + Math.imul(t ^ t>>>7, 61|t) ^ t; return ((t ^ t>>>14)>>>0)/4294967296; };
    for(let k=0;k<trials;k++){ bins.fill(0); let mx=0; for(let i=0;i<F;i++){ const b=(rnd()*U)|0; if(++bins[b]>mx) mx=bins[b]; } tot += mx; }
    v = tot/trials;
  }
  ECMP_MEMO.set(key, v); return v;
}
/* 叶交换机往 spine 的上行：每个 rank 网卡流量里 upF 要上行，一个叶交换机下 P 个口、收敛比 over（上行口 U = P/over），
   F 条流哈希到 U 条上行。返回上行比网卡本身慢多少倍（≤ 1 表示网卡先成为瓶颈）。自适应路由按均匀摊开算 */
function spineX(net, upF, flowsPerNic){
  if(!(upF>0)) return 0;
  const P = net.ports||128, over = net.over||1, U = Math.max(1, Math.round(P/over));
  const nicUp = upF*P;                                              // 这个叶交换机下等效有多少张网卡的流量要上行
  if(!net.ecmp) return nicUp/U;
  const F = Math.max(1, Math.round(nicUp*(flowsPerNic||1)*(net.qps||1)));
  return nicUp/F*ecmpMax(F, U);                                     // 每条流 nicUp/F 张网卡的量，最忙的上行口装 E[max] 条
}
/* 链路：组内 peer 都在同一个 NVLink 域就走 NVLink。跨域时（集群页的 topo.net 给出 scale-out 参数，没有时 segment 无限大、不收敛）：
   · 环（AG / RS / AR）：NCCL 每个 channel 从不同网卡出域（rail），域内有 d 个组员就有 d 张网卡并行，最多到域内 NVLink 的速度（不比同样的环放在域里快）。
     按 NCCL 的环序，每个 channel 在一个 segment 只跨出一次，上行流量只占 d / m
   · 点对点（PP、CP ring、halo）：边界那张卡自己的一张网卡；m 个组员里有一个要跨 segment
   · all-to-all：发往本域 / 本 segment 其他域 / 其他 segment 的三部分分别走 NVLink、网卡、网卡 + spine
   主页面的拓扑都在一个域里，这些分支只在集群页和手动加大推理卡数时用到 */
function link(hw, topo, n, stride){
  if(n<=1) return { bw:Infinity, p2p:Infinity, a2a:Infinity, intra:true, alpha:0 };
  const s = stride||1, nvl = hw.nvl*1e9, net = topo.net||{}, blk = Math.min(topo.domain, topo.gpus||topo.domain);
  if(n*s<=topo.domain) return { bw:nvl, p2p:nvl, a2a:nvl, intra:true, inNet:!!net.nvls, d:n, m:n };
  const nic = hw.ib*1e9, d = s>=blk ? 1 : Math.min(n, Math.max(1, Math.floor(blk/s)));
  const seg = net.segment||Infinity, m = s>=seg ? 1 : Math.min(n, Math.max(d, Math.floor(seg/s)));
  const remF = (n-d)/(n-1), farF = (n-m)/(n-1);
  const xRing = m<n ? spineX(net, d/m, 1) : 0, xP2p = m<n ? spineX(net, 1/m, 1) : 0;
  const xA2a = m<n ? spineX(net, farF/remF, Math.max(1, (n-m)/Math.max(1, Math.min(d, net.rails||8)))) : 0;
  const bw = Math.min(d*nic, nvl)/Math.max(1, xRing);
  const p2p = nic/Math.max(1, xP2p);
  const a2a = 1/Math.max((d-1)/(n-1)/nvl, remF*Math.max(1, xA2a)/nic);
  return { bw, p2p, a2a, intra:false, inNet:!!net.sharp, d, m, remF, farF, nic, nvl, xRing, xP2p, xA2a };
}
/* ring all-gather / reduce-scatter：msg 为整份张量字节 */
function agT(msg, n, lk, g){ if(n<=1||msg<=0) return 0; return (lk.intra?g.alphaNv:g.alphaIb)*1e-6*Math.log2(n)+ msg*(n-1)/n/(lk.bw*(g.etaAg||g.etaN)); }
/* all-reduce。lk.inNet：NVLS（NVSwitch 里做归约）或 SHARP（IB 交换机里做归约），每个 rank 只发一份、收一份 */
function arT(msg, n, lk, g, algo){
  if(n<=1||msg<=0) return 0; const a=(lk.intra?g.alphaNv:g.alphaIb)*1e-6;
  if(g.lb || lk.inNet) return a + msg/(lk.bw*g.etaN);              // 下限：每个 rank 至少要收到一份完整的归约结果
  if(algo==="oneshot") return a + msg*(n-1)/(lk.a2a*g.etaN);       // 每个 rank 读其余 n−1 份整消息，受单卡带宽限制
  if(algo==="twoshot") return 2*a + 2*msg*(n-1)/n/(lk.a2a*g.etaN);
  return 2*a*Math.log2(n) + 2*msg*(n-1)/n/(lk.bw*g.etaN);         // NCCL ring
}
/* all-to-all：每卡发 sent 字节，(n−1)/n 离开本卡；跨域时按 link 给的 a2a 带宽（三段里最慢的那段） */
function a2aT(sent, n, lk, g, alphaUs){ if(n<=1||sent<=0) return 0; return (alphaUs ?? (lk.intra?g.alphaNv:g.alphaIb))*1e-6 + sent*(n-1)/n/(lk.a2a*g.etaN); }

/* MoE 路由翻转：每层 router 在训练和推理里选得不一样的概率 p（R3 论文实测约 10%）。按各层独立算，一个 token 平均有 nMoe·p 层不同，
   至少一层不同的概率是 1 − (1 − p)^nMoe；实测往往比独立假设低（翻转在层间相关），所以这是上界 */
const routeFlips = (nMoe, p) => ({ exp:nMoe*p, any:1-Math.pow(1-p, nMoe) });

/* 单个算子的 roofline：t = max(FLOPs/峰值, 字节/带宽, kernel 数 × 最短 kernel 时间) */
function roof(flops, bytes, peak, hw, g, kernels){
  const tc = compT(flops, peak, g), tm = memT(bytes, hw, g), tf=(kernels||0)*g.kernelUs*1e-6;
  const t = Math.max(tc, tm, tf);
  return { t, tc, tm, tf, bound: t===tf&&tf>0 ? "launch" : tc>=tm ? "compute" : "memory", I: bytes>0?flops/bytes:Infinity };
}

return { DT, dtB, peakKey, peakOf, compT, memT, ridge, attainable, linkRidge, G0, IDEAL, BASIS, KEFF, etaK, gK, freeze, MODELS, HW, TOPO, homeTopo, sm10x, derive, layerPlan, csaPlan, csaLatB, csaKv, qoW, capKeys, swaKeys, ecmpMax, spineX, link, agT, arT, a2aT, routeFlips, roof };
})();
