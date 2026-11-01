# Scouter

An interactive roofline lab for MoE training and inference. Pick a model and a GPU, and Scouter estimates a decode step, a chunked prefill or a post-training step kernel by kernel. It then shows where each attention and MLP module sits on the roofline, and how changing the architecture moves it.

**Live:** https://hecate0821.github.io/Scouter/ · 中文说明：[README.zh.md](README.zh.md)

> Every number is an estimate derived from open-source implementations and public data. It can differ a lot from what you measure. Use it to learn, not to plan capacity.

The interface is in Chinese for now. All of its text lives in `locales/`, so you can add a language without touching code. See [docs/i18n.md](docs/i18n.md).

## Pages

Open `index.html` directly. It works from `file://` and needs no build step. You can also serve the folder with any static server.

The layout follows a developer console:

- **Sidebar:** the pages, grouped by tab. Press `/` to search them.
- **Top bar:** the model, GPU and topology, plus the **source** switch:
  - **Ideal lower bound:** the time this configuration cannot beat. Every op runs at peak with no overheads and moves only the traffic it must: no CUDA graph padding, only the experts certain to be hit, perfect load balance, and all-reduce at 1× its message. The step is at least the critical path and each resource's total (compute, HBM, communication), so overlapping two ops never hides shared bandwidth. The conventions follow [k3-inference-calculator](https://github.com/ByronHsu/k3-inference-calculator) (Apache-2.0), generalized here to every model. It stays below every measured point in the calibration sets (at most 0.88 of the measurement);
  - **Modeled:** as close to reality as the model gets (the default): reachable efficiency per kernel class, fixed launch and sync costs, the expected expert coverage under uniform routing, load imbalance, and the collective algorithms the engines actually use;
  - **Measured:** only real data points from public sources and anonymized measurements.
- **Parameter panel** on the right: the controls for the current page.
- **CLI** and **JSON** buttons: show the `Scouter.*` call behind the current page, as a `tools/scouter.js` command and as its output.
- **Copy link:** copies the URL. It keeps the context, for example `?model=kimik3&hw=B200&src=ideal#infer/step`.
- **Light / Dark:** switches the theme. It shares its setting and colors with [hecate0821.github.io](https://hecate0821.github.io/). Text uses the system sans-serif font, with a sans-serif (hei) font for Chinese.

| Tab | What it shows |
|---|---|
| Overview | One decode step and one post-training step of the same model, side by side. |
| Model architecture | **Model overview:** the layer stack (full, sliding-window and linear attention; MoE and dense MLP), one card per module, parameter counts, and KV/state cache versus context. **One module:** a diagram of the mechanism; sliders for its architecture knobs; the module's roofline as batch, context or sequence length changes, with the unedited preset as a ghost; throughput curves; the kernel list; and links to the code each assumption follows. **Compare:** every model's modules on one roofline, attention and MLP curves, and the per-layer time split. |
| Post-training | One step (fwd, dgrad, wgrad, recompute, communication, optimizer, memory). Parallelism: every collective in a one- or two-layer sample, with FSDP, CP and EP shown one dimension at a time. Low-precision recipes: Megatron, Transformer Engine and torchtitan defaults only. A comparison with anonymized measurements. |
| Inference | One decode or prefill step. Deployment and parallelism: TP, DP-attention, EP and PD disaggregation. Low-precision inference. Speculative decoding: MTP, EAGLE-3, DFlash and DSpark. **Against measurements:** the SGLang cookbook's published benchmark rows replayed as closed-loop runs (TPOT and TTFT); and a profiler trace you load, compared with the step kernel category by kernel category. |
| RL | Train/inference mismatch. Deterministic rollout: batch invariance, reproducibility and bitwise train/inference match. Low-precision rollout and weight sync. Low-precision training and QAT. |
| Appendix | Measured H200 kernels against the ideal and modeled rooflines, with Nsight-Compute-style SOL efficiency; the per-kernel-class efficiency table and the public number behind each cell; public data points with their confidence, and the past year's releases; quantization basics; and the modeling method with the open-source code it follows. |

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
- **Per kernel class:** in inference, GEMMs and attention cores take their efficiency from a small table by class, precision and GPU generation (Hopper, Blackwell SM100, Blackwell Ultra SM103). Each cell names the public or measured number it comes from, for example cuBLAS BF16 on B200, FlashAttention-3 / 4, FlashMLA, DeepGEMM and CUTLASS SM103 NVFP4. Setting `kEff` to 0 goes back to the flat 0.7 / 0.8.
- **Kernel-level overheads:** 0.45 µs of idle time between kernels inside a CUDA graph, removed by programmatic dependent launch (`pdl`), following [vLLM #64846](https://github.com/vllm-project/vllm/pull/64846) and [FlashInfer #8135](https://github.com/flashinfer-ai/flashinfer/pull/8135); NVFP4 / MXFP4 MoE pads the expert hidden size to a multiple of 256 on Blackwell, following [vLLM #39510](https://github.com/vllm-project/vllm/pull/39510).
- **Serving benchmark:** `serve` runs `conc` requests with `isl` input and `osl` output tokens in a closed loop. TPOT is the decode step plus the other requests' prefills spread over the output. Against the 16 calibratable SGLang cookbook rows (GLM-5.3-Flash on GB300, Kimi K3 on B300; rows that are a single burst of requests are left out), the median predicted / measured TPOT is 1.00 and 10 rows are within ±25%. MTP at concurrency 1 is about 2.3× too fast, because per-step scheduling and draft overheads are not modeled. Measured TTFT is 95–185 ms above the model, which covers only GPU time.

The mechanism diagrams reimplement the visual ideas of [Attention Lab](https://divagr.com/attention-lab/), credited under each diagram.

## Data plane and agents

All numbers come from one data plane: the models in `src/engine/`, the data points in `src/data/`, and one entry point, `src/plane/scouter.js`. The page only draws what it returns. The same calls run in Node and on the command line, so an agent can use Scouter as a reference calculator without a browser:

```
node tools/scouter.js infer --model kimik3 --hw B200 --basis ideal --B 128 --pick stepT,tComp
node tools/scouter.js sweep --run train --model qwen35 --over.ep 4,8 --pick stepT,mfu
node tools/scouter.js points --set inference_public --where.engine SGLang
node tools/scouter.js serve --model kimik3 --hw B300 --N 8 --conc 16 --isl 8192 --osl 1024 --pick tpot,ttft0,tokPerGpu
node tools/scouter.js calib --set cookbook_tpot --pick median,n
node tools/scouter.js trace --model qwen35 --hw H200 --B 64 --steps 20 --file.trace trace.json
```

[AGENTS.md](AGENTS.md) lists every call, its options and the fields it returns.

## Layout

```
index.html              page shell; loads the scripts below in order (plain <script> tags sharing one global scope)
css/roofline.css
src/i18n/i18n.js        t(key, params), data-i18n attributes, language choice (?lang=, saved choice, browser language)
locales/zh/*.js         all UI text, one file per namespace; data.js holds translations of data text (docs/i18n.md)
src/engine/             the models: tables and pure functions; no DOM, no text, no data store (docs/engine.md)
  msg.js                Msg.of(key, params): the engine returns messages, the caller renders them
  common.js             model, hardware and topology tables; derive(); roofline helpers; link and collective times
  infer.js              one inference step: SGLang / vLLM / TokenSpeed kernel sequences, the engine-neutral reference,
                        speculative decoding (draft, verify, commit)
  train.js              one training step: Megatron-Core / torchtitan
  par.js                FSDP / CP / EP communication and overlap
  quant.js              low-precision GEMM, one MoE layer, the three training GEMMs
  arch.js               architecture explorer: step kernels grouped into attention / MLP modules per layer, and the
                        mechanism quantities each diagram draws
  calib.js              calibration: which measured points each data set reproduces, and the prediction for each
  trace.js              a profiler trace's kernels grouped by category and compared with one step
  node.js               loads the engine in Node (one vm context), for tests
src/data/
  db.js                 generated: the data-point store (const DB_RAW)
  query.js              DB.set / DB.where / DB.one / DB.meta
src/plane/
  scouter.js            the data plane's entry point (Scouter.*): engine + data store, JSON in and out (AGENTS.md)
  node.js               loads the data plane in Node, for tests and tools/scouter.js
src/ui/                 one renderer per page; app.js loads last and handles routing and the first render
  theme.js shell.js     theme before first paint; sidebar, top bar, source switch, CLI / JSON views, URL state
  base.js state.js config.js   formatting and plotting base, global state, presets
  arch.js archviz.js    architecture explorer pages; mechanism diagrams
  overview.js train.js par.js calib.js comm.js infer.js tradeoff.js quant.js spec.js rl.js
  releases.js public.js dbview.js charts.js
data/
  anon/*.json           anonymized data points, one data set per file
  public/*.json         public data points (hand-curated, each with src / url)
  db.json               generated: canonical form, one point per line
tools/build_db.py       builds data/db.json and src/data/db.js
tools/scouter.js        the data plane from the command line, printing JSON
tools/import_cookbook.py   regenerates data/public/sglang_cookbook.json from the SGLang cookbook at a pinned commit
tools/import_ncu.py     turns an Nsight Compute raw CSV into an anonymized kernel data set
tools/reduce_trace.py   shrinks a large profiler trace (.json / .json.gz) to its GPU kernel events
sim/                    separate Python project: the training layouts on CPU gloo processes with PyTorch FSDP2 / TP / EP,
                        recording every collective as a JSON trace (sim/README.md)
tests/                  Node checks: engine, data plane, FLOPs parity, sim, i18n, shell, regression and snapshot (jsdom), timing
AGENTS.md               the data plane for agents: calls, options, returned fields
docs/engine.md          the engine's interface and rules
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
- **Confidence:** every set is `measured` (our anonymized sets), `reported` (a number stated upstream) or `derived` (computed from published numbers, or measured under a simulated condition). A row can override its set's. Public sets carry `as_of`, the date they were curated or the upstream commit was read.
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
uv run --no-project tools/import_cookbook.py --ref <sha>        # re-import the SGLang cookbook benchmark rows
uv run --no-project tools/import_ncu.py report.csv --name b200_moe --hw B200 --dtype fp4 --conditions "..."
```

`tools/import_ncu.py` documents the `ncu --csv --page raw` command to run first. Each kernel becomes a point with its duration and DRAM bytes; with `--flops` it also gets analytic FLOPs. `Scouter.kernels --set <name>` then gives each kernel's SOL efficiency, and the profile (tool version, clocks, replay mode, metrics) is kept with the set.

Pages read points with queries like `DB.where("cp_full_model", {model: "qwen35", cp: 8})`. Data text is written in the data files; translations of it go in `locales/<lang>/data.js` (see [docs/i18n.md](docs/i18n.md)).

## Tests

```
cd tests && npm install
npm test               # engine + plane + flops + sim + i18n + shell + regression
npm run engine         # the engine alone in Node: every model × GPU × topology × engine / framework / precision / module;
                       # no throw, no NaN, text only as messages; prints the calibration medians
npm run plane          # the data plane alone in Node: every call under both bases, ideal never slower than modeled,
                       # tools/scouter.js prints what the API returns
npm run shell          # the console: every page under every source, and page = data plane = printed CLI command
npm run flops          # every model's matmul FLOPs, written out again from the config with textbook formulas, against the engine
npm run sim            # the engine's training collectives against the PyTorch traces in sim/traces (no Python needed)
npm run i18n           # no UI text in code (checked with a pseudo-localized render of every page); every key used exists
npm run regression     # every model × framework × preset, then every page: reports NaN / undefined and script errors
npm run timing         # render time of each page
node snapshot.js ../index.html new.json base.json   # dump the visible text of every view and diff it with a baseline
```

After changing a simulator case, regenerate its trace with `cd sim && uv run python -m scouter_sim.run` (see [sim/README.md](sim/README.md)).
