# The HuggingArch backend

For real models, Scouter's realistic numbers come from [HuggingArch](https://github.com/shenh10/HuggingArch) (MIT, by Han Shen), a per-operator roofline simulator that walks a model's checked architecture spec. Scouter keeps two things of its own: the **ideal lower bound** (`Common.IDEAL`), and Scouter's **modeled** basis (`Common.G0`) for models HuggingArch has no spec for (edited models, Muse Spark Open). This page explains how the two meet. Back to [README](../README.md).

## Pieces

| Piece | What it is |
|---|---|
| `vendor/HuggingArch/` | Git submodule: [Hecate0821/HuggingArch](https://github.com/Hecate0821/HuggingArch), a fork of `shenh10/HuggingArch`. The fork's changes are listed under [The fork](#the-fork). |
| `tools/hga/` | A `uv` project (`pyproject.toml`) with `export.py`, which runs HuggingArch offline and writes the exports. |
| `src/data/hga/<model>.js` | The exports: one file per real model, generated, loaded by the page only when that model is picked (`index.js` is the registry, `HGA_DATA`). |
| `src/engine/hga.js` | `Hga`: the signatures, the grid, interpolation, and the comparison with the ideal lower bound. Pure functions, like the rest of `src/engine/`. |
| `src/plane/scouter.js` | `Scouter.hga`, `Scouter.hgaTrain`, `Scouter.hgaCells`: the API the page, the CLI and the tests share. |
| `src/ui/hga.js` | The first block of *Inference · One step* and *Training · One step*. It only draws what the plane returns. |

Real models are `Hga.REAL`: `qwen35` (Qwen/Qwen3.5-35B-A3B), `kimik3` (moonshotai/Kimi-K3), `glm53f` (zai-org/GLM-5.3-Flash), `dsv41f` (deepseek-ai/DeepSeek-V4.1-Flash) and `qwen38_27b` (Qwen/Qwen3.8-27B). Cards are `Hga.CARD`: H100, H200, B200, B300, GB200, GB300 (HuggingArch has no RTX PRO 6000).

On the page, the middle source in the top bar reads **HuggingArch** for a real, unedited model. Pages HuggingArch does not cover show the ideal lower bound, and the source note says so. Editing the model turns it into a custom model, and the switch goes back to **Modeled**.

## Signatures

A signature is one deployment, described twice: as a HuggingArch `ParallelConfig` (`hga`, used by the export) and as a Scouter config (`cfg`, used for the ideal lower bound of the same deployment). There are five per engine, shared by every model; `moeOnly` signatures are skipped for dense models. A signature's `moe` part applies only to MoE models (`Hga.sigOf(engine, id, model)` resolves it): the three TP training signatures put the routed experts on EP8 rather than splitting them by TP. Expert TP is where the two sides would describe different deployments. Megatron's alltoall dispatcher all-gathers the top-k-duplicated rows in the expert-TP group, while HuggingArch's walk all-reduces the hidden states. TorchTitan with EP off splits the experts by TP, which Scouter's torchtitan model does not express. With TP attention every rank already holds every token, so neither engine dispatches in `tp8ep8`: vLLM runs its all-to-all kernels only with DP > 1 (`use_all2all_kernels`), SGLang's default `--moe-a2a-backend` is none, and the expert outputs join the TP all-reduce. HuggingArch derives the same from the token layout; DeepEP is covered by the DP-attention signatures. `flags` are the engine's launch arguments, shown as a label: nothing is launched.

| Engine | Id | GPUs | Layout |
|---|---|---|---|
| SGLang | `tp8` | 8 | attention and MoE tensor-parallel |
| | `tp8ep8` | 8 | TP8 attention, EP8 experts; no dispatch, expert outputs join the TP all-reduce |
| | `dpa8ep8` | 8 | DP attention 8, EP8, DeepEP |
| | `dpa16ep16` | 16 | DP attention 16, EP16, DeepEP |
| | `tp8dcp8` | 8 | TP8 with decode context parallelism 8 |
| vLLM | `tp8` | 8 | attention and MoE tensor-parallel |
| | `tp8ep8` | 8 | TP8 attention, EP8 experts; no dispatch, expert outputs join the TP all-reduce |
| | `dp8ep8` | 8 | DP8 attention, EP8, DeepEP low latency |
| | `dp16ep16` | 16 | DP16 attention, EP16 |
| | `tp8dcp8` | 8 | TP8 with DCP 8 |
| Megatron-Core | `tp8sp` | 64 | TP8 + SP, selective recompute, DP 8; MoE experts EP8 |
| | `tp4pp2` | 64 | TP4 + SP × PP2, selective recompute; MoE experts EP8 |
| | `ep8` | 64 | EP8 (alltoall), selective recompute |
| | `tp2ep8pp4` | 64 | TP2 + SP × EP8 × PP4, full recompute |
| | `tp2cp4` | 64 | TP2 × CP4, experts EP4 × ETP2, full recompute |
| torchtitan | `fsdp` | 64 | FSDP over 64 shards, full recompute |
| | `hsdp` | 64 | 8 replicas × 8 shards, full recompute |
| | `fsdp_tp8` | 64 | FSDP × TP8 + SP, selective recompute; MoE experts EP8 |
| | `fsdp_ep8` | 64 | FSDP × EP8, full recompute |
| | `fsdp_cp4` | 64 | FSDP × CP4, experts EP4, full recompute |

The full definitions, flags included, are `Hga.SIG` (`node tools/scouter.js catalog --pick hga.sig`).

## The grid

`Hga.GRID`, read by `export.py` from the engine so the two sides cannot drift:

- **Inference:** prompt length `isl` ∈ {1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K}, decode batch per DP rank `b` ∈ {1, 2, 4, …, 512}, 1024 output tokens. The decode step is taken at context `isl + osl/2`, HuggingArch's convention. Prefill is one request, not chunked (`prefill_chunk_size = None`), and the ideal side does the same (`C = isl`). Precision is as the checkpoint ships, on both sides: weights, dense layers and KV from the model's defaults, and the expert activations from `eActShipped` where the checkpoint sets them (DeepSeek-V4.1-Flash ships FP8 activations, so its experts run W4A8 on Hopper too, while the inference page keeps the engine's own Hopper kernel, W4A16).
- **Training:** sequence length `seq` ∈ {4K, 8K, 16K, 32K}, micro-batch `mbs` ∈ {1, 2}, 64 GPUs, 2M tokens per step, BF16 mixed precision with Adam.

The page and the plane interpolate between grid points bilinearly in log(x) and log(value): step time, bytes and FLOPs follow power laws in batch and context. A point outside the grid is clamped to its edge. `exact` says whether a point is on the grid. A training step is not interpolated whole, because its micro-batch count is an integer that jumps between grid points. Only the per-micro-batch quantities are interpolated: each pass's terms, the FSDP traffic per micro-batch, the activations per micro-batch in flight, and the bubble's share. The step is then rebuilt with the fork's formulas at the point's own `m = ceil(global tokens / (DP × seq × mbs))`, the same `m` the ideal uses.

## Export

```bash
cd tools/hga
uv run python export.py                                   # every real model, inference and training, every card
uv run python export.py --model kimik3 --kind infer --engine sglang --sig tp8 --card B300
```

One HuggingArch walk per (model, signature, card) is substituted at every grid point (`capacity.iter_sweep_cell`), so a cell takes 1–40 s. Runs merge into the existing file, so a partial run only replaces the cells it priced. Different models can export in parallel; one model's file should have one writer at a time.

A refusal is an answer, not a failure: the cell keeps `{ok: false, reason}` with HuggingArch's own words, and the page prints it verbatim. Typical reasons:

- the weights do not fit on a rank (Kimi K3 on 8 GPUs other than B300 / GB300);
- DCP splitting across KV head shards on GQA models with few KV heads;
- DCP not implemented for DSA's sparse top-k or DeepSeek-V4.1's sink softmax;
- tensor cuts that do not tile the ranks.

`node tools/scouter.js hgaCells --model glm53f` lists every cell and its reason.

To add a real model: its HuggingArch spec has to exist (`vendor/HuggingArch/backend/arch/snapshots`), then add it to `Hga.REAL` and export. To add a signature: add it to `Hga.SIG`, add its label to `locales/*/hga.js` (`sig.<engine>.<id>`), and export with `--sig <id>`.

## Data format

`src/data/hga/<model>.js` calls `HGA_DATA.register(model, data)` with:

- `meta`: `{model, hf, hga, upstream, generated, grid}`. `hga` is the fork commit the cells were walked on, with `-dirty` appended if the fork had uncommitted changes.
- `infer["<engine>.<sig>"][card]`: `{ok, isl[], b[], osl, maxB[] (per isl, HuggingArch's max decode batch per DP rank), pre[isl], dec[isl][b], warn[]}`. A `null` point means no request fits at that context.
- `train["<engine>.<sig>"][card]`: `{ok, seq[], mbs[], parallel (the ParallelConfig the forward walked), pts[seq][mbs], warn[], assumptions[]}`.

Points are flat arrays to keep the files small, three significant digits:

- **Inference phase point** (`Hga.POINT`, then the bound split, then per class):
  - `t`: the phase time HuggingArch publishes.
  - `tc`, `tm`, `tn`: Σ compute, Σ HBM and Σ communication time over rows.
  - `s0`: Σ over rows of max(c, m, n), i.e. without the launch floor.
  - `fl`: FLOPs per rank.
  - `by`: HBM bytes per rank, of which `wb` are weights and `kv` are KV.
  - `cb`: communication bytes.
  - `nl`: kernel launches.
  - Then the time spent in rows bound by compute, memory, comm and launch.
  - Then, for each class in `Hga.ROLES` (`attn`, `moe`, `dense`, `router`, `other`, `comm`), its `[t, tc, tm, tn]`.
- **Training pass** (`Hga.PASS`), for forward / backward / recompute: `[t, compute, memory, comm, launch, flops, bytes, commBytes, launches]`, then one time per class.
- **Training step:** `st = [micro-batches, t_pipe, bubble, dp_serial, dp_exposed, t_step, t_opt, tokens/s/GPU, MFU]`, `mem = [weights, grads, optimizer, activations, total, hbm]`, `dp = [[name, collective, group, bytes, time], …]`.

HuggingArch row roles map to classes as follows: `attention` → `attn`, `ffn_moe` and `ffn_moe_latent` → `moe`, `ffn_moe_shared` and `ffn_dense` → `dense`, `router` → `router`, collectives → `comm`, anything else → `other`. Scouter's op keys, inference and training, map the same way (`ROLE_OF` in `src/engine/hga.js`; the training `act`, `permute` and `unperm` belong to routed MoE).

## The comparison

`Scouter.hga({model, engine, sig, hw, isl, b, axis})` returns, per phase (`decode`, `prefill`):

| Field | Meaning |
|---|---|
| `ideal`, `hga` | Each side's `t`, `compute`, `memory`, `comm`, `flops`, `bytes`, `commBytes`, `kernels`, `byRole`. HuggingArch adds `launch`, `weightBytes`, `kvBytes`, `byBound` and `byRoleBound` (each class's compute / HBM / comm time). |
| `waterfall` | `[{key: "ideal", t}, {key: "acc", d}, {key: "ser", d}, {key: "launch", d}, {key: "hga", t}]`. The pieces add up exactly. |
| `resources` | `{compute, memory, comm, flops, bytes, commBytes, kernels}`, each `[ideal, hga]`. |
| `roles` | `[{role, ideal, hga}]`, time per class. |
| `ratio` | HuggingArch ÷ ideal. |

Beside the phases: `throughput` (`ideal`, `hga` tok/s/GPU at the decode step), `maxB` and `fits` (capacity per DP rank on each side), `exact`, `warn` (the ideal side's messages), `hgaWarnings` (HuggingArch's, verbatim), `source` (fork commit, export date), and with `axis: "b"` or `"isl"` a `curve` of `{x, decode, prefill, tok, fits, maxB}` along the grid.

The waterfall splits the gap between the ideal step `I` and HuggingArch's step `H = Σ_op max(c, m, n, launch)`:

- **`acc` (accounting)** = max(ΣC, ΣM, ΣN) of HuggingArch − `I`. The two sides count different amounts. HuggingArch counts:
  - every op's activation reads and writes,
  - the intermediate tensors left after fusion,
  - the experts the batch actually touches,
  - α-β collectives with the engine's algorithm.

  The ideal counts only the traffic that must move.
- **`ser` (per-op serialization)** = Σ max(c, m, n) − max(ΣC, ΣM, ΣN). HuggingArch takes each op's own bound and sums them, so one op's compute cannot hide behind another's memory traffic. The ideal lets the whole step overlap, by resource conservation.
- **`launch`** = `H` − Σ max(c, m, n): HuggingArch's 1 µs floor per kernel.

`Scouter.hgaTrain({model, engine, sig, hw, seq, mbs, axis: "seq"})` returns:

| Field | Meaning |
|---|---|
| `ideal`, `hga` | Each side's step: `t`, `micro`, `m` (micro-batches per DP rank), `pipe`, `bubble`, `dp` (exposed), `dpSerial`, `opt`, `fwd`, `bwd`, `rc`, `mfu`, `tokPerGpu`, `mem`, `fits`. HuggingArch adds `memParts`, `pass` (per-pass detail, unpacked) and `dpParts` (each collective). |
| `waterfall` | `ideal` → `micro` (m × forward + backward + recompute) → `bubble` → `dp` → `opt` → `hga`. Rounding of the three-digit export is folded into `micro`; an `other` piece appears only if the rest exceeds 0.5% of the step. |
| `passes` | `[{key, ideal, hga}]` for forward, backward and recompute of one micro-batch. |
| `roles` | `[{role, ideal: {fwd, bwd, rc, t}, hga: {fwd, bwd, rc, t}}]`, one micro-batch per class. The ideal side counts only the communication compute cannot hide; the FSDP gather it cannot hide and PP's point-to-point go in its forward `comm`. On each side the classes add up to `micro`. |

`Scouter.hgaCells({model})` lists which cells were priced and why the rest were refused.

### When HuggingArch comes in under the ideal

The ideal is a lower bound on Scouter's own count of each operator's work, not on any model's answer. Where HuggingArch counts less for some class, its step can come in under the ideal. The page says so instead of calling it a gap, and names the class where HuggingArch counts less (`roles`). `node tests/plane.js` walks every priced cell on and off the grid and fails below 0.85 of the ideal; `HGA_UNDER=1 node tests/plane.js` lists every phase under the ideal with that class.

Every large shortfall so far meant the two sides described different deployments or one side had a bug, not a real speedup. Fixed:

- training collectives were priced with vLLM's custom or NVLS all-reduce (the fork now prices them as NCCL runs them);
- the fork lacked Megatron's sequence-parallel re-gather in backward, and the expert all-to-all under sequence parallelism when `ep == tp`;
- expert TP was described differently by the two sides (the TP signatures now put experts on EP8), and Scouter's ETP GEMM had the wrong shapes;
- vLLM `tp8ep8` was charged an expert dispatch vLLM does not run with DP = 1, and SGLang `tp8ep8` was DeepEP on Scouter's side only;
- DeepSeek-V4.1-Flash's shipped W4A8 experts were priced at BF16 on Hopper;
- the training ideal got the tokens of a whole CP group as its tokens per GPU, and off the grid HuggingArch's step was interpolated with a fractional micro-batch count.

What remains (42 phases of the walk, none under 0.92) are two per-operator differences. In both, HuggingArch counts less than the kernels Scouter models have to move:

- **Linear-attention core** (GDN in qwen35 and qwen38_27b, KDA in kimik3 and glm53f; training): HuggingArch prices the chunked delta rule as one kernel that reads q, k and v and writes o, about 0.8 KB per value head per token. FLA's chunked kernels also write and re-read their intermediates (the chunk matrices A, w, u and the states h), about 5 KB, and Scouter counts them.
- **NVLS all-reduce** (qwen38_27b TP8 prefill): HuggingArch moves (N − 1)/N of the tensor each way per rank. The ideal's bound is one full copy, because every rank has to receive every reduced element.

## The fork

The fork adds only what Scouter needs, and each change is marked in its code:

- **`backend/training/`**: a training step on top of the forward walk. Its derivation and assumptions are in the fork's `docs/training.md` and in `ASSUMPTIONS`, which each export carries. In short:
  - the forward is the serving prefill graph at batch = micro-batch;
  - backward per launch class: GEMM is dgrad + wgrad, attention 2.5×, element-wise 2× the bytes;
  - recompute is full or selective;
  - sequence parallelism splits the tokens of replicated element-wise launches, and Megatron's backward all-gathers each block's input once more for the weight gradient;
  - with sequence parallelism and an expert group equal to the TP group (`ep == tp`), each MoE layer dispatches and combines the rank's 1/TP of the tokens by all-to-all over EP (the walk, with every token on every TP rank, has no exchange there to re-price);
  - collectives in the graph are re-priced as NCCL runs them (the walk prices them with the serving binding's custom or symmetric-memory all-reduce, which training frameworks do not run);
  - DP / FSDP collectives are priced with HuggingArch's own collective law;
  - 1F1B bubble;
  - Adam at 30 B/param.

  These training numbers are the fork's, not upstream HuggingArch's.
- **`capacity.iter_sweep_cell(keep_infeasible=True)`** and **`CapacityKnobs.walk_unloadable`**: a forward the serving memory budget refuses still yields its rows, because a training rank shards what a serving rank must hold whole.
- **A sweep over a deployment whose weights cannot load** yields that refusal at every point, instead of failing.

## License and credit

HuggingArch is Copyright (c) 2026 shenh10 and used under the MIT License. The full text ships in `vendor/HuggingArch/LICENSE`, and the copyright notice stays in the fork. The [README](../README.md#huggingarch) has the acknowledgement and the citation. Being used here does not mean HuggingArch's author has reviewed Scouter's numbers.
