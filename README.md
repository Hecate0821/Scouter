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
| **Ideal lower bound** | The time this configuration cannot beat: peak throughput, no overheads, only necessary traffic, and a step at least as long as the critical path and each resource's total. It stays below every measured calibration point. Conventions follow [k3-inference-calculator](#acknowledgements), generalized to all models. |
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

## Acknowledgements

Scouter is built on what these open-source projects publish: their code, PRs, docs and benchmarks. Public blogs and papers are cited next to the numbers they support. Every module page and data point links to the exact source. Being listed here does not mean a project has reviewed or endorsed Scouter's numbers.

| Area | Projects |
|---|---|
| Inference engines | [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [TokenSpeed](https://github.com/lightseekorg/tokenspeed), [DeepInfer](https://github.com/deepseek-ai/DeepInfer) |
| Training and RL | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [torchtitan](https://github.com/pytorch/torchtitan), [TorchRL](https://github.com/pytorch/rl), [slime](https://github.com/THUDM/slime), [Miles](https://github.com/radixark/miles), [AReaL](https://github.com/areal-project/AReaL), [PyTorch](https://github.com/pytorch/pytorch) |
| Kernels | [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [CUTLASS](https://github.com/NVIDIA/cutlass), [TransformerEngine](https://github.com/NVIDIA/TransformerEngine), [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) |
| MoE parallelism | [DeepEP](https://github.com/deepseek-ai/DeepEP), [dist_moe](https://github.com/meta-pytorch/dist_moe), [mixture-of-kittens](https://github.com/cursor/mixture-of-kittens) |
| KV cache | [Dynamo](https://github.com/ai-dynamo/dynamo), [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [llm-d](https://github.com/llm-d/llm-d) |
| Low precision, speculative decoding | [torchao](https://github.com/pytorch/ao), [SpecForge](https://github.com/sgl-project/SpecForge) |
| Benchmarks and profiling | [sgl-cookbook](https://github.com/sgl-project/sgl-cookbook), [sglang-amd-cookbook](https://github.com/jhinpan/sglang-amd-cookbook), [lora_perf_lora_profile](https://github.com/yushengsu-thu/lora_perf_lora_profile), [llm_reproducibility](https://github.com/nanomaoli/llm_reproducibility), [torch_utils](https://github.com/fzyzcjy/torch_utils), [combine_traces](https://github.com/tlrmchlsmth/combine_traces) |
| Model layouts | [open-weights-carousel](https://github.com/NickLucche/open-weights-carousel), [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| Ideal lower bound | [k3-inference-calculator](#acknowledgements) |
| Mechanism diagrams | [Attention Lab](https://divagr.com/attention-lab/) |
