# Scouter

An interactive roofline lab for MoE training and inference. Pick a model and a GPU, and Scouter estimates a decode step, a chunked prefill or a post-training step kernel by kernel. It then shows where each attention and MLP module sits on the roofline, and how changing the architecture moves it.

**Live:** https://hecate0821.github.io/Scouter/ · 中文说明：[README.zh.md](README.zh.md)

> Every number is an estimate derived from open-source implementations and public data. It can differ a lot from what you measure. Use it to learn, not to plan capacity.

The interface is in Chinese for now. All of its text lives in `locales/`, so you can add a language without touching code. See [docs/i18n.md](docs/i18n.md).

## Pages

Open `index.html` directly. It works from `file://` and needs no build step. You can also serve the folder with any static server.

| Tab | What it shows |
|---|---|
| Overview | One decode step and one post-training step of the same model, side by side. |
| Model architecture | **Model overview:** the layer stack (full, sliding-window and linear attention; MoE and dense MLP), one card per module, parameter counts, and KV/state cache versus context. **One module:** a diagram of the mechanism; sliders for its architecture knobs; the module's roofline as batch, context or sequence length changes, with the unedited preset as a ghost; throughput curves; the kernel list; and links to the code each assumption follows. **Compare:** every model's modules on one roofline, attention and MLP curves, and the per-layer time split. |
| Post-training | One step (fwd, dgrad, wgrad, recompute, communication, optimizer, memory). Parallelism: every collective in a one- or two-layer sample, with FSDP, CP and EP shown one dimension at a time. Low-precision recipes: Megatron, Transformer Engine and torchtitan defaults only. A comparison with anonymized measurements. |
| Inference | One decode or prefill step. Deployment and parallelism: TP, DP-attention, EP and PD disaggregation. Low-precision inference. Speculative decoding: MTP, EAGLE-3, DFlash and DSpark. |
| RL | Train/inference mismatch. Deterministic rollout: batch invariance, reproducibility and bitwise train/inference match. Low-precision rollout and weight sync. Low-precision training and QAT. |
| Appendix | Public data points and the past year's releases, quantization basics, and the modeling method with the open-source code it follows. |

Addresses look like `#tab/subtab/anchor`, for example `#arch/module/lin`, `#infer/spec` or `#train/par/cp`. Old addresses such as `#par`, `#quant`, `#public` and `#appendix/models` redirect to their new place.

**Models:**

| Model | Attention | MLP |
|---|---|---|
| Qwen3.5-35B-A3B | Gated DeltaNet + gated GQA, 3 : 1 | MoE with a shared expert |
| Kimi K3 | KDA + MLA | MoE with a shared expert; first layer dense |
| GLM-5.3-Flash | KDA + DSA (DeepSeek sparse attention) | MoE with a shared expert; first 3 layers dense |
| Muse Spark Open (816B-A42B) | GQA: 46 sliding-window (2048) layers + 16 global layers | MoE |
| Qwen3.8-27B | Gated DeltaNet + gated GQA, 3 : 1 | dense SwiGLU (the dense baseline) |

**GPUs:** H100, H200, B200, B300, GB200 and GB300.

**Topologies:** one 8-GPU HGX node, or 64 GPUs in an NVL72 rack. Every parallel group stays inside one NVLink domain.

## Where the numbers come from

The ground truth is the code, PRs and docs of SGLang, vLLM, TensorRT-LLM, Megatron-LM, torchtitan, TorchRL, slime, Miles and AReaL, plus public blogs and papers. The appendix lists the modeling method. Every module page links to the files its assumptions follow.

The architecture explorer has no formulas of its own. It takes the same kernel-by-kernel step model the inference and training pages use, groups the kernels into attention and MLP modules, and divides by the number of layers each kernel covers.

- **Inference** uses an engine-neutral reference profile. It includes only optimizations that the engines all turn on by default, and runs on one GPU without CUDA-graph padding.
- **Training** uses Megatron-Core defaults on one GPU, without recompute.

| Part | Assumed by default | Follows |
|---|---|---|
| Full attention (GQA, sliding window) | FlashAttention | flash-attention; SGLang `flashattention_backend.py`; vLLM v1 `flash_attn.py`; TensorRT-LLM `attention_backend` |
| MLA | Decode: up-projections absorbed into the query and output (FlashMLA). Prefill: the prefix KV is expanded back into per-head K/V. | FlashMLA; SGLang `deepseek_v2.py`; vLLM `mla_attention.py` |
| DSA | DeepSeek's open-source path: an FP8 lightning indexer (DeepGEMM) picks the top-k keys, then FlashMLA sparse attention runs over them | DeepSeek-V3.2-Exp, DeepGEMM, FlashMLA; SGLang `dsa_backend.py`; vLLM `flashmla_sparse.py`, `indexer.py` |
| Linear attention (Gated DeltaNet, KDA) | flash-linear-attention: the fused recurrent kernel for decode, the chunked kernel (chunk 64) for prefill and training | fla `ops/gated_delta_rule`, `ops/kda`; SGLang `hybrid_linear_attn_backend.py`; vLLM `gdn_attn.py` |
| MoE | Fused grouped GEMM over the routed experts | SGLang `fused_moe_triton`; vLLM `fused_moe`; Megatron `moe/experts.py` |
| Prefill | Chunked prefill: 8192-token chunks on GPUs with less than 160 GB of memory, 16384 otherwise | SGLang server args; vLLM `arg_utils.py` |

The roofline model:

- **Time per kernel:** max(FLOPs / (peak × 0.7), bytes / (HBM bandwidth × 0.8), kernel floor).
- **Ridge point:** peak × 0.7 / (HBM × 0.8).
- **Peaks used:** attention cores run at the BF16 peak. GEMMs run at the peak for their weight precision, with FP8 and MXFP4 / NVFP4 mapped to the FP8 and FP4 peaks.

The mechanism diagrams reimplement the visual ideas of [Attention Lab](https://divagr.com/attention-lab/), credited under each diagram.

## Layout

```
index.html              page shell; loads the scripts below in order (plain <script> tags sharing one global scope)
css/roofline.css
src/i18n/i18n.js        t(key, params), data-i18n attributes, language choice (?lang=, saved choice, browser language)
locales/zh/*.js         all UI text, one file per namespace (docs/i18n.md)
src/engine/             the models; no DOM
  common.js             model, hardware and topology tables; derive(); link times
  infer.js              one inference step: SGLang / vLLM / TokenSpeed kernel sequences, the engine-neutral reference,
                        speculative decoding (draft, verify, commit)
  train.js              one training step: Megatron-Core / torchtitan
  par.js                FSDP / CP / EP communication and overlap
  quant.js              low-precision GEMM, one MoE layer, the three training GEMMs
  arch.js               architecture explorer: step kernels grouped into attention / MLP modules per layer
src/data/
  db.js                 generated: the data-point store (const DB_RAW)
  query.js              DB.set / DB.where / DB.one / DB.meta
src/ui/                 one renderer per page; app.js loads last and handles routing and the first render
  base.js state.js config.js   formatting and plotting base, global state, presets
  arch.js archviz.js    architecture explorer pages; mechanism diagrams
  overview.js train.js par.js calib.js comm.js infer.js tradeoff.js quant.js spec.js rl.js
  releases.js public.js dbview.js charts.js
data/
  anon/*.json           anonymized data points, one data set per file
  public/*.json         public data points (hand-curated, each with src / url)
  db.json               generated: canonical form, one point per line
tools/build_db.py       builds data/db.json and src/data/db.js
tests/                  jsdom checks
docs/i18n.md            how UI text and languages work
```

## Data points

```
{ schema: 1,
  sets:   { <id>: { title, origin: "anon" | "public", stage, hw, gpus, topo, model, conditions, dims: [...], metrics: { <name>: { unit, label } } } },
  points: [ { id, set, ...dims, ...metrics } ] }
```

- **Anonymized points** (`origin: "anon"`) are measured numbers. They keep only the measurement conditions and the numbers, with no source. All points in a set share the same dims and metrics. The `id` is built from the dims.
- **Public points** (`origin: "public"`) come from public PRs, blogs, papers, release notes and docs, and carry `src` / `url`. The build computes the weight `w` of each quantization evidence point from recency × hardware match × source type. The reference date is `sets.quant_evidence.as_of`.
- **Releases** are LMSYS blog posts and SGLang / vLLM / slime / Miles / AReaL GitHub releases from 2025-11 to 2026-10. Each is keyed `releases:<project>:<version>`. Pages look up release links by project and version, and a missing entry is an error.

The build fails if:

- an anonymized point carries `src` / `url`;
- an anonymized set's description contains source-like words;
- an id is duplicated;
- a set is empty.

To add a public number, add a row (`id`, `date`, `src`, `url`, numbers) to the matching file in `data/public/`. To add a measurement, add a JSON file to `data/anon/` with a title, conditions, dims, metrics and points. Then rebuild:

```
uv run tools/build_db.py                     # regenerate data/db.json and src/data/db.js
uv run tools/build_db.py --check             # only check that the generated files are current
uv run tools/build_db.py --sqlite x.sqlite   # also export SQLite (one view per set) for ad-hoc queries
```

Pages read points with queries like `DB.where("cp_full_model", {model: "qwen35", cp: 8})`.

## Tests

```
cd tests && npm install
npm run regression     # every model × framework × preset, then every page: reports NaN / undefined and script errors
npm run i18n           # no UI text left in code; every key used exists; no page renders a missing key
npm run timing         # render time of each page
node snapshot.js ../index.html new.json base.json   # dump the visible text of every view and diff it with a baseline
```
