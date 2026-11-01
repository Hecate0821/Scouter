# Modeling and data

How Scouter turns a configuration into kernel times, and how its data points are stored. Back to [README](../README.md).

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

Pages read points with queries like `DB.where("cp_full_model", {model: "qwen35", cp: 8})`. Data text is written in the data files; translations of it go in `locales/<lang>/data.js` (see [docs/i18n.md](i18n.md)).
