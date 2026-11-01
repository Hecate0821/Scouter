# Scouter data plane API

Scouter estimates MoE inference and post-training steps kernel by kernel and places every kernel, module and step on a
GPU roofline. Everything the page shows comes from one **data plane**, and you can call that plane directly, with no browser.

## Layers

| Layer | Where | Rule |
|---|---|---|
| Data plane: models | `src/engine/` | Tables and pure functions. No DOM, no display text (names and warnings are `Msg` keys), no data store. |
| Data plane: measurements | `src/data/` (generated from `data/`) | The data-point store: public numbers with `src` / `url`, and anonymized measurements. |
| Data plane: entry point | `src/plane/scouter.js` | `Scouter.*`: plain objects in, JSON-serializable objects out. The only API you need. |
| View | `index.html`, `src/ui/`, `css/`, `locales/` | Renders what the plane returns. It never computes physics or holds data. |
| Collective simulator | `sim/` | A separate Python project (PyTorch on CPU gloo). It shares only JSON files with the rest (`sim/README.md`). |

The same `src/plane/scouter.js` runs in the browser, in Node (`src/plane/node.js`) and behind the command line
(`tools/scouter.js`). `tests/shell.js` checks that the page, the API and the printed command give the same numbers.

## Sources

Every computed number has a **basis**. Measurements are separate:

| Source | Meaning |
|---|---|
| `ideal` | A strict lower bound for this configuration (model, parallel layout, batch, the engine's kernel sequence and overlap). Each op takes `max(FLOPs / peak, bytes / HBM bandwidth, link bytes / link bandwidth)` with 100% efficiency and no launch, sync or collective latency (`Common.IDEAL`, `lb: 1`). Only necessary traffic counts: no CUDA graph or FP4 padding, routed experts limited to those certain to be hit (top-k slices under TP, `ceil(k / EP)` experts on the busiest rank under EP), perfect EP / CP balance, all-reduce at 1× its message (the receive floor, reachable with in-switch reduction). The step is the largest of the critical path and the total compute, HBM and communication time (`bound`), so overlap never hides shared resources. The conventions follow [k3-inference-calculator](../README.md#acknowledgements) (Apache-2.0), generalized to all models; no code is taken from it. `tests/plane.js` checks that it stays below every measured point in every calibration set. |
| `modeled` (default) | As close to reality as the model gets: the roofline at the reachable fractions `η` (`Common.G0`: GEMM 0.7, HBM 0.8, network 0.7), plus fixed kernel, launch, sync and collective costs, the expected expert coverage under uniform routing, load imbalance, CUDA graph padding, and the collective algorithm each engine uses (one-shot / two-shot / ring). Inference GEMMs and attention cores also take a per-class, per-GPU efficiency from `Common.KEFF` (see below). |
| `measured` | Data points, not a computation. Public numbers from SGLang, vLLM, TensorRT-LLM, Megatron, torchtitan, slime, Miles, AReaL PRs, docs, blogs and papers, plus anonymized single-node measurements. Use `points`, `kernels`, `eff`, `calib` and `trace`. |

Every data set has a **confidence**, returned by `sets` and shown in the appendix:

| Confidence | Meaning |
|---|---|
| `measured` | Our own anonymized measurements (`data/anon/`). |
| `reported` | A number stated in an upstream PR, doc, blog or paper (`data/public/`, the default). |
| `derived` | Computed from published numbers, or measured under a simulated condition (for example SGLang cookbook rows run with a fixed simulated MTP accept length). |

Public sets also carry `asOf`, the date they were curated or the upstream commit was read. A row can carry its own
`confidence` when it differs from its set's.

On the page, the **Ideal / Modeled / Measured** switch in the top bar picks the source. The **CLI** and **JSON** buttons show the
`Scouter.*` call behind the current page and its output. The URL keeps the context:
`?model=kimik3&hw=B200&topo=hgx8&src=ideal#infer/step`.

## Command line

Run these from the repository root with Node 18 or later. They need no install, and each prints JSON.

```
node tools/scouter.js catalog                                   # models, GPUs, topologies, bases, engines, defaults, data sets
node tools/scouter.js roofline --hw B200 --flops 4e12 --bytes 2e9 --dtype fp8 --basis ideal
node tools/scouter.js infer  --model kimik3 --hw B200 --engine vllm --phase prefill --n 32768 --pick stepT,tComp,tComm
node tools/scouter.js train  --model qwen35 --fw torchtitan --ep 8 --basis ideal --pick stepT,mfu
node tools/scouter.js module --model glm53f --phase decode --B 64 --pick measure.mods.full
node tools/scouter.js sweep  --run infer --model qwen35 --over.B 1,8,64,256 --over.phase decode,prefill --pick stepT
node tools/scouter.js sets                                       # measured data sets
node tools/scouter.js points --set inference_public --where.engine SGLang
node tools/scouter.js kernels                                    # measured H200 GEMM / copy points against the rooflines, with SOL η
node tools/scouter.js eff                                        # the per-kernel-class efficiency table and the point behind each cell
node tools/scouter.js calib  --set h200_kernels --basis ideal    # the model replaying measured points
node tools/scouter.js calib  --set cookbook_tpot                 # SGLang cookbook benchmark rows against serve
node tools/scouter.js serve  --model kimik3 --hw B300 --N 8 --conc 16 --isl 8192 --osl 1024   # closed-loop benchmark estimate
node tools/scouter.js kv     --model qwen35 --hw H200 --N 8 --sessions 256 --hostGB 200       # KV cache of multi-turn agent sessions
node tools/scouter.js trace  --model qwen35 --hw H200 --B 64 --steps 20 --file.trace trace.json   # a profiler trace by kernel category
echo '{"model":"kimik3","hw":"GB300","topo":"nvl64","B":256}' | node tools/scouter.js infer --json -
```

These options apply to every command:

- `--model`, `--hw` and `--topo` pick table entries. `catalog` lists them.
- `--basis ideal|modeled` picks the source for computed numbers.
- `--g.<key>` overrides one efficiency parameter, such as `--g.etaC 0.6`.
- `--pick a,b.c` keeps only those fields; a dot means a nested field.
- Any other `--key value` overrides the run's default config (`catalog.defaults`). Values are parsed as JSON when possible.
- `--json '{...}'` passes the whole query as one object; `--json -` reads it from stdin. A custom model can be passed as an object in `model`.
- `--file.<key> path` reads a JSON file into one field of the query, for example `--file.trace` or `--file.points`.
- `--lang zh` renders the message keys as Chinese text.

## Node and browser

```js
const { Scouter } = require("./src/plane/node.js")();
const a = Scouter.infer({ model: "kimik3", hw: "B200", basis: "ideal", B: 128 });
const b = Scouter.infer({ model: "kimik3", hw: "B200", B: 128 });
console.log(b.stepT / a.stepT);       // how far the modeled engine sits above the hardware roofline
```

In the page's browser console, `window.Scouter` is the same object.

## What comes back

All units are SI: times in seconds, sizes in bytes, work in FLOPs. Fields named `TFLOPs` and `GBs` are rates. `Infinity` is
printed as the string `"Infinity"`.

| Call | Main fields |
|---|---|
| `roofline` | `peaks.{bf16,fp8,fp4}.{peakTFLOPs, attainableTFLOPs, ridge}`, `hbmGBs`, `attainableGBs`. With `flops` / `bytes`, also `kernel.{t, tc, tm, bound, I, TFLOPs}`. |
| `infer` | Runs `Infer.decode` or `Infer.prefill` by `phase`, the same path as the page, so `specK` draft and commit, `tokPerGpu`, `tpot` and `ttft` / `chunks` / `nChunks` are included. `stepT`, `tComp`, `tComm`, `bound` (ideal only: `path`, `compute`, `hbm`, `comm` and which one set the step), `ops[]` (`key`, `name`, `group`, `layers`, `flops`, `bytes`, `t`, `tc`, `tm`, `tf`, `bound`, `I`), `comm[]`, `warn[]`, `query.cfg` (the full config used). |
| `train` | Config = `Train.BASE`, then `Train.FW_DEF[fw]` (`catalog.defaults.trainFw`), then your keys. A value the framework does not define is an invalid-layout warning, not a silent default. `stepT`, `mfu`, `tokPerGpu`, `ops[]` (`fwd` / `dgrad` / `wgrad` each with time), `comm[]`, `mem`, `warn[]`, `query.cfg`. |
| `module` | `measure.mods.<full\|lin\|moe\|dense>` with `t`, `flops`, `bytes`, `byBound` and `ops[]` for one layer; `modules`; `facts`. |
| `serve` | A closed-loop benchmark of `conc` requests with `isl` input and `osl` output tokens: `tpot` (decode step plus the prefills of the other requests spread over the `osl − 1` decode steps, `tpotDecode` + `interf`; TPOT is (E2E − TTFT) / (osl − 1) as in bench_serving and genai-bench), `ttft0` (unloaded prefill), `tokPerGpu` (input + output tokens per second per GPU, the SGLang cookbook convention), `outTokPerS`, `N`, `mem`, `warn[]`. |
| `kv` | KV cache of multi-turn agent sessions (config = `Infer.BASE` + `KV.BASE`: `sys`, `turnIn`, `turnOut`, `turns`, `sessions`, `hostGB`, `remoteGB`, `inclusive`, `replicas`, `routing`). All bytes per GPU. `perTok` (`full` KV per token over full-attention layers, `swa` per token over sliding-window layers capped at `W`, `snap` = one linear-attention state snapshot), `page` (physical block; vLLM enlarges it for hybrid models until one attention page holds one state page) and `gran` (where the next turn can hit), `ws1` (bytes a session leaves between turns), `cap` / `fit` per tier, `frac` (`hbm`, `host`, `remote`, `miss`: share of sessions, capacity / working set as in the Che approximation), `hitRate` (by input tokens), `ttft` / `ttftNone` / `ttftHit` (mean per turn), `turns[]` (per sampled turn: `hbm`, `host`, `remote`, `miss`, `none` TTFT and the load times), `sweep[]` over sessions, `space[]` over context, `dtypes[]`. |
| `sweep` | `rows[]`: the varied keys plus the picked fields. `run` can be `roofline`, `infer`, `train`, `module`, `serve` or `kv`. |
| `sets` | Per set: `origin`, `stage`, `confidence`, `asOf`, hardware, size, dims, metrics, and the `calib` sets that use it. |
| `points` | `meta` (title, origin, stage, hardware, conditions) and `points[]`. Public points carry `src` and `url`. |
| `kernels` | `points[]` (`op`, `I`, `TFLOPs`, `GBs`, and the SOL efficiencies `etaC` = FLOPs / time / peak and `etaM` = bytes / time / HBM peak), `measured` (the best of each), `profile` (how the set was measured), and the `ideal` and `modeled` rooflines. `--set` picks another kernel set (for example one made by `tools/import_ncu.py`); `points` with `hw` and `dtype` passes points directly. |
| `eff` | `reference` (`etaC`, `etaM`) and `table[]`: one cell per kernel class (`gemm`, `attn`), kind (`c` compute, `m` memory), precision and SM version, with `eta`, its ratio to the reference, and the `evidence` point it comes from. |
| `calib` | `rows[]` (`p`, `meas`, `pred`, `ratio`) and `median` of predicted / measured; `data` is the data set the rows come from. |
| `trace` | Per kernel category (`gemm`, `attn`, `lin`, `moe`, `comm`, `samp`, `elem`, `other`): measured and predicted seconds per step, their ratio and the top kernels; `total`; `ranks`; `overlap` (time two kernels on one stream overlapped, as with PDL, counted once); `gpu` (`busy` = union of kernel time, `wall`, `idle`, and the model's `step`); `comm` (`floor` = each comm kernel's fastest 10% of calls × its calls, `wait` = measured − floor, the time spent waiting for slower ranks); and `unmatched`, the most expensive kernels no rule recognized. The trace is a torch profiler / Chrome trace (`{traceEvents}`, GPU kernel events only; merged multi-rank traces are split by `args.source_idx`, a `[TPxx]` pid, or pid, and averaged per rank) or a list `[{name, us, calls}]`; `steps` is how many steps it covers. The prediction per category includes ops that overlap on another stream, because their kernels are in the trace too. |

Calibration sets (`catalog.calib`): the anonymized `cp_attention`, `cp_full_model`, `efsdp_full_model`, `efsdp_comm` and
`h200_kernels`, and the public `cookbook_tpot` / `cookbook_ttft`.
The cookbook sets replay the SGLang cookbook benchmark rows marked `calib: true` (`tools/import_cookbook.py` documents which
rows and why) through `serve`. Rows with `num_prompts` under twice the concurrency are one burst, not a steady state, and are
not calibrated. Accept lengths measured on the random dataset are upper bounds: random prompts make outputs loop, and
bench_serving reports a cumulative server-side average.

## Efficiency knobs

These are fields of `g` (`--g.<key>`), with `Common.G0` as the default and `Common.IDEAL` as the hardware roofline:

| Key | Default | Meaning |
|---|---|---|
| `etaC`, `etaM`, `etaN` | 0.7, 0.8, 0.7 | Reachable fraction of the compute, HBM and network peaks. |
| `kEff` | 1 (ideal 0) | 1 looks up GEMM and attention efficiency per kernel class, precision and GPU in `Common.KEFF`, scaled by `etaC / G0.etaC` (and `etaM` for memory-bound cells). MoE falls back to GEMM; SM103 falls back to SM100. 0 uses `etaC` / `etaM` everywhere. Training is not affected. |
| `gapUs` | 0.45 (ideal 0) | Idle time between kernels inside a CUDA graph, in µs. The config field `pdl: true` (programmatic dependent launch) removes it. |

In the inference config, NVFP4 / MXFP4 MoE weights on SM100 and later pad the expert hidden size to a multiple of 256.
`eAct` (expert activation precision) defaults to what SGLang does: MXFP4 runs W4A8 at the FP8 peak on Blackwell (the FlashInfer
TRT-LLM path) and W4A16 at the BF16 peak on Hopper (Marlin); NVFP4 on Hopper runs W4A16; everything else matches the weights.
`eAct: "same"` forces W4A4 / W8A8. Causal attention against a capped key set (sliding window, DSA top-k) counts keys with
`Common.capKeys`. On SGLang, Kimi K3 at TP8 on Blackwell with at most 12 tokens per rank splits the latent up projection by
column and all-gathers it (`latag`), as `gemm_ag.py` does.

Context parallelism with load balancing (`cpLB`, zigzag) assumes a balanced split. ring-flash-attention measures zigzag at
about 90% of single-GPU flash-attention ÷ 8 (forward + backward, CP8, 8K tokens per GPU), so balanced CP here is 9–17% optimistic
on the attention core; the unbalanced ring (52–54%) matches. Miles' FSDP backend runs CP through ring-flash-attention's HF
adapter, which is the all-gather path on contiguous chunks with no zigzag: model it as `cpType: "allgather"`, `cpLB: false`.
Training memory: with `reshard: false` (torchtitan) the BF16 parameters gathered in the forward pass stay until the backward
pass; on Hopper, FP8 recipes keep a second, transposed copy of each quantized weight (the FP8 wgmma reads both operands K-major).

`warn[]` entries with `bad: true` mean the configuration cannot run, for example because it doesn't fit in memory. Names and
warnings are messages `{ key, p }`; their text is in `locales/<lang>/engine_*.js`.

## Changing the code

- Physics belongs only in `src/engine/`. A new view-facing quantity goes into the engine or into `Scouter`, never into `src/ui/`.
- Text goes into `locales/<lang>/`, never into code (`docs/i18n.md`). The pseudo-locale test fails on any hardcoded word.
- Data goes into `data/public/*.json` (with `src` and `url`) or `data/anon/*.json`. Rebuild with `uv run --no-project tools/build_db.py`.
  - `tools/import_cookbook.py --ref <sha>` regenerates `data/public/sglang_cookbook.json` from the SGLang cookbook at that commit.
  - `tools/import_ncu.py report.csv --name ... --hw ...` turns an Nsight Compute raw CSV into an anonymized kernel set.
  - `tools/reduce_trace.py trace.json.gz` shrinks a profiler trace too large to load (hundreds of MB) to its GPU kernel events.
- A new `Common.KEFF` cell must name a data point (`ref: "<set>:<id>"`) whose measured efficiency matches it; `tests/plane.js` checks.
- Run `cd tests && npm install && npm test` before you push. It runs the engine, data-plane, FLOPs parity, collective, i18n, shell and regression checks.
