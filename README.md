# Scouter

An interactive roofline lab for LLM training and inference, MoE and dense. Pick a model and a GPU, and Scouter estimates a decode step, a chunked prefill or a post-training step kernel by kernel, then places every attention and MLP module on the roofline.

**Live:** https://hecate0821.github.io/Scouter/ · 中文：[README.zh.md](README.zh.md)

> Every number is an estimate derived from open-source implementations and public data. Use it to learn, not to plan capacity.

## Quick start

```bash
python3 -m http.server 8000      # or just open index.html; no build step
```

```bash
node tools/scouter.js infer --model kimik3 --hw B200 --basis ideal --B 128 --pick stepT,bound
```

```bash
cd tests && npm install && npm test
```

## Three sources

The top-bar switch (and `--basis` on the CLI) picks where a number comes from:

| Source | Meaning |
|---|---|
| **Ideal lower bound** | The time this configuration cannot beat: peak throughput, no overheads, only necessary traffic, and a step at least as long as the critical path and each resource's total. It stays below every measured calibration point. Conventions follow [k3-inference-calculator](https://github.com/ByronHsu/k3-inference-calculator) (Apache-2.0), generalized to all models. |
| **Modeled** (default) | As close to reality as the model gets: per-kernel-class efficiency, launch and sync costs, expected expert coverage, load imbalance, and the collectives each engine actually uses. |
| **Measured** | Data points only: public numbers (with source links) and anonymized measurements. |

## Pages

| Tab | What it shows |
|---|---|
| Overview | One decode step and one post-training step of the same model, side by side. |
| Model architecture | Layer stack, per-module roofline and knobs, mechanism diagrams, and all models compared. |
| Post-training | One step (fwd / dgrad / wgrad, recompute, comm, optimizer, memory), parallelism, low-precision recipes. |
| Inference | Decode / prefill, deployment and parallelism, low precision, speculative decoding, the KV cache (bytes per layer type, prefix hit rate of multi-turn agent sessions across HBM / host / remote tiers, how SGLang, vLLM, TRT-LLM and TokenSpeed manage it), and checks against the SGLang cookbook and your own profiler trace. |
| RL | Train/inference mismatch, deterministic rollout, low-precision rollout and QAT. |
| Appendix | Measured kernels vs. the rooflines, the efficiency table, public data points, and the modeling method. |

**Models:** Qwen3.5-35B-A3B, Kimi K3, GLM-5.3-Flash, Muse Spark Open, Qwen3.8-27B (dense baseline). **GPUs:** H100, H200, B200, B300, GB200, GB300. **Topologies:** HGX 8-GPU node, NVL72 rack (64 GPUs).

The URL keeps the context, e.g. `?model=kimik3&hw=B200&src=ideal#infer/step`. UI text lives in `locales/` ([docs/i18n.md](docs/i18n.md)).

## Layout

```
index.html, css/, src/ui/   the page: renders what the data plane returns, no physics
src/engine/                 the models: pure functions, no DOM, no text (docs/engine.md)
src/plane/scouter.js        Scouter.*: the single API used by the page, the CLI and tests (docs/api.md)
data/{public,anon}/*.json   data points → tools/build_db.py → data/db.json, src/data/db.js
locales/<lang>/             all UI and engine text
tools/                      scouter.js (CLI), build_db.py, importers, reduce_trace.py
sim/                        PyTorch-on-CPU collective simulator (sim/README.md)
tests/                      Node checks: engine, plane, FLOPs, sim, i18n, shell, regression
```

## Docs

- [AGENTS.md](AGENTS.md): how to run, simulate and change the model (for people and coding agents).
- [docs/api.md](docs/api.md): every `Scouter.*` call, option and returned field.
- [docs/modeling.md](docs/modeling.md): what each kernel follows, the roofline formulas, calibration and the data-point format.
- [docs/engine.md](docs/engine.md), [docs/i18n.md](docs/i18n.md): engine and text conventions (Chinese).

## Sources

Ground truth is the public code, PRs and docs of SGLang, vLLM, TensorRT-LLM, Megatron-LM, torchtitan, TorchRL, slime, Miles and AReaL, plus public blogs and papers. Every module page links to the code its assumptions follow. Mechanism diagrams reimplement the visual ideas of [Attention Lab](https://divagr.com/attention-lab/).
