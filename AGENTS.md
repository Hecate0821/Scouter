# Scouter for agents

Static site plus pure-JS models. No build step, no framework. Node 18+ for the CLI and tests, `uv` for Python tools.
Full API: [docs/api.md](docs/api.md). Modeling details: [docs/modeling.md](docs/modeling.md).

## Run the web UI

```bash
python3 -m http.server 8000   # then open http://localhost:8000/ (opening index.html from file:// also works)
```

State lives in the URL: `?model=kimik3&hw=B200&topo=hgx8&src=ideal#infer/step`. In the browser console, `window.Scouter` is the same API as the CLI.

## Run a simulation

Every call prints JSON. `--basis ideal|modeled`, `--pick a,b.c` filters fields, any other `--key value` overrides the config.

```bash
node tools/scouter.js catalog                                    # models, GPUs, topologies, defaults, data sets
node tools/scouter.js infer  --model kimik3 --hw B300 --N 8 --B 64 --pick stepT,tComp,tComm,bound
node tools/scouter.js train  --model qwen35 --fw megatron --ep 8 --pick stepT,mfu,mem.used
node tools/scouter.js serve  --model kimik3 --hw B300 --N 8 --conc 16 --isl 8192 --osl 1024 --pick tpot,ttft0
node tools/scouter.js kv     --model qwen35 --hw H200 --N 8 --sessions 256 --hostGB 200 --pick hitRate,ttft,frac
node tools/scouter.js sweep  --run infer --model qwen35 --over.B 1,8,64,256 --pick stepT
node tools/scouter.js calib  --set cookbook_tpot --pick median,n   # model vs. measured points
node tools/scouter.js trace  --model qwen35 --hw H200 --B 64 --steps 20 --file.trace trace.json
node tools/scouter.js hga    --model qwen35 --engine sglang --sig tp8 --hw H200 --isl 8192 --b 64 --axis b   # HuggingArch vs ideal
```

In Node: `const { Scouter } = require("./src/plane/node.js")();`. Collective traces from real PyTorch (CPU gloo): `cd sim && uv run python -m scouter_sim.run`.

## Change the calculation

| What | Where |
|---|---|
| Model / GPU / topology tables, roofline, collectives, `G0` (modeled) and `IDEAL` (lower bound) | `src/engine/common.js` |
| Inference step, `serve` | `src/engine/infer.js` |
| KV cache: bytes per layer type, prefix hits across tiers, TTFT per turn | `src/engine/kv.js` |
| Training step | `src/engine/train.js`, `src/engine/par.js` |
| Calibration against measured sets | `src/engine/calib.js` |
| Public / anonymized data points | `data/public/*.json`, `data/anon/*.json`, then `uv run --no-project tools/build_db.py` |
| Text (UI, warnings, op names) | `locales/<lang>/*.js` |
| Real models' simulation: signatures, grid, comparison | `src/engine/hga.js`; data from `cd tools/hga && uv run python export.py` (needs `git submodule update --init`), see [docs/hga.md](docs/hga.md) |
| HuggingArch itself (training step, sweep options) | the fork in `vendor/HuggingArch` (its own repo, MIT); commit and push there, then bump the submodule here |

Rules:

- Physics only in `src/engine/`. `src/ui/` renders what `Scouter.*` returns and never recomputes.
- Real models (`Hga.REAL`) take their realistic numbers from HuggingArch; never tune `Common.G0` to match it. `src/data/hga/*.js` is generated: re-export, do not hand-edit.
- A signature in `Hga.SIG` is one deployment on both sides (`hga`/`train` for the export, `cfg` for the ideal). If one side cannot express a layout, change the signature, not one side. After a change, walk every cell (`node tests/plane.js`; `HGA_UNDER=1` lists every phase under the ideal with the class HuggingArch counts less for). It fails below 0.85 of the ideal: so far that has meant a mismatch or a bug, not a speedup. A small shortfall can be a per-operator accounting difference (docs/hga.md lists the known ones); name it there before accepting it.
- Every formula that differs between the two bases checks `g.lb`. The ideal basis must stay a true lower bound: `tests/plane.js` fails if it exceeds any measured calibration point.
- No display text in code: engine returns `Msg.of(key, params)`; the i18n test fails on hardcoded words.
- Public data rows carry `src` and `url`; anonymized rows carry neither. Cite only public sources.
- `Common.KEFF` cells must name the data point they come from (`ref: "<set>:<id>"`).

Check, then deploy (GitHub Pages serves `gh-pages`):

```bash
cd tests && npm install && npm test
```

```bash
git push origin main main:gh-pages
```
