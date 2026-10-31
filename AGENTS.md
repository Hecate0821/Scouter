# Scouter for agents

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
| `ideal` | Hardware roofline. Each kernel takes `max(FLOPs / peak, bytes / HBM bandwidth)`, with 100% efficiency and no launch, sync or collective latency. This is a lower bound for the GPU. The kernel sequence and overlap rules are still those of the chosen engine or framework. |
| `modeled` (default) | The roofline at the reachable fractions `η` (`Common.G0`: GEMM 0.7, HBM 0.8, network 0.7), plus fixed kernel, launch, sync and collective costs. These follow the open-source engines' defaults. |
| `measured` | Data points, not a computation. Public numbers from SGLang, vLLM, TensorRT-LLM, Megatron, torchtitan, slime, Miles, AReaL PRs, docs, blogs and papers, plus anonymized single-node measurements. Use `points`, `kernels` and `calib`. |

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
node tools/scouter.js kernels                                    # measured H200 GEMM / copy points against the rooflines
node tools/scouter.js calib  --set h200_kernels --basis ideal    # the model replaying measured points
echo '{"model":"kimik3","hw":"GB300","topo":"nvl64","B":256}' | node tools/scouter.js infer --json -
```

These options apply to every command:

- `--model`, `--hw` and `--topo` pick table entries. `catalog` lists them.
- `--basis ideal|modeled` picks the source for computed numbers.
- `--g.<key>` overrides one efficiency parameter, such as `--g.etaC 0.6`.
- `--pick a,b.c` keeps only those fields; a dot means a nested field.
- Any other `--key value` overrides the run's default config (`catalog.defaults`). Values are parsed as JSON when possible.
- `--json '{...}'` passes the whole query as one object; `--json -` reads it from stdin. A custom model can be passed as an object in `model`.
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
| `infer` | `stepT`, `tComp`, `tComm`, `ops[]` (`key`, `name`, `group`, `layers`, `flops`, `bytes`, `t`, `tc`, `tm`, `tf`, `bound`, `I`), `comm[]`, `warn[]`, `query.cfg` (the full config used). |
| `train` | `stepT`, `mfu`, `tokPerGpu`, `ops[]` (`fwd` / `dgrad` / `wgrad` each with time), `comm[]`, `mem`, `warn[]`, `query.cfg`. |
| `module` | `measure.mods.<full\|lin\|moe\|dense>` with `t`, `flops`, `bytes`, `byBound` and `ops[]` for one layer; `modules`; `facts`. |
| `sweep` | `rows[]`: the varied keys plus the picked fields. |
| `points` | `meta` (title, origin, stage, hardware, conditions) and `points[]`. Public points carry `src` and `url`. |
| `kernels` | `points[]` (`op`, `I`, `TFLOPs`, `GBs`), `measured` (best GEMM and copy), and the `ideal` and `modeled` rooflines. |
| `calib` | `rows[]` (`p`, `meas`, `pred`, `ratio`) and `median` of predicted / measured. |

`warn[]` entries with `bad: true` mean the configuration cannot run, for example because it doesn't fit in memory. Names and
warnings are messages `{ key, p }`; their text is in `locales/<lang>/engine_*.js`.

## Changing the code

- Physics belongs only in `src/engine/`. A new view-facing quantity goes into the engine or into `Scouter`, never into `src/ui/`.
- Text goes into `locales/<lang>/`, never into code (`docs/i18n.md`). The pseudo-locale test fails on any hardcoded word.
- Data goes into `data/public/*.json` (with `src` and `url`) or `data/anon/*.json`. Rebuild with `uv run --no-project tools/build_db.py`.
- Run `cd tests && npm install && npm test` before you push. It runs the engine, data-plane, collective, i18n, shell and regression checks.
