# scouter-sim

Runs Scouter's training layouts with real PyTorch parallelism on CPU processes over gloo, and records every collective
those processes issue. The point is to check the engine's communication model against what the frameworks actually do,
not to time anything: CPU gloo says nothing about GPU speed. Which collectives run (group, op, calls, bytes, forward or
backward) is decided by FSDP2, DTensor and the token dispatcher above the backend, so the same code issues them on GPUs.

## Decoupled by design

- The simulator never imports the web app and the app never loads the simulator. They meet only in JSON:
  `cases/*.json` (inputs) and `traces/*.json` (outputs). `tests/sim.js` checks both directions statically.
- The recorder knows nothing about Scouter's cost model: a trace holds raw facts (op, process group, dtype, element
  counts, all-to-all splits). Converting them to the engine's byte convention happens in `tests/sim.js`.
- Traces are checked in, so the Node test runs without Python or torch.

## Run

```
cd sim
uv sync                                   # torch (CPU build) + numpy
uv run python -m scouter_sim.run          # every case in cases/  -> traces/
uv run python -m scouter_sim.run cases/fsdp4_ep2_moe.json
cd ../tests && npm run sim                # compare with the engine
```

All cases take about 2.5 s: most of it importing torch once, then about 0.3 s per case. Workers are forked from the process that
imported torch (Linux or macOS; no Windows), and `uv sync` precompiles the bytecode.

## Cases

```json
{ "name": "fsdp4_ep2_moe", "about": "…",
  "model":  { … one entry in the engine's MODELS schema (D, L, V, full, lin, moe, dense) … },
  "layout": { "N": 4, "tp": 1, "ep": 2 },
  "tok": 512 }
```

`tok` is the tokens per rank per micro-batch (the engine's `tok`). Supported subset: full GQA attention on every layer
(optionally gated), dense SwiGLU FFN for the first `dense.n` layers, then MoE (router, stacked experts, optional shared
expert and gate). TP applies to dense models; MoE models use EP. No CP or PP yet.

## What is mirrored

| Piece | Follows |
|---|---|
| FSDP2 wrapping | torchtitan [`apply_fsdp_to_decoder`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/fsdp.py): `tok_embeddings` alone with `set_unshard_in_backward(False)`; `[norm, lm_head]` not resharded after forward; every block resharded; in MoE blocks routed experts on the expert-FSDP mesh, the rest on the dense mesh (`ShardPlacementResult` per parameter); bf16 params, fp32 reduce |
| EP | torchtitan [`AllToAllTokenDispatcher`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/common/moe/token_dispatcher.py): int64 token-count exchange, then dispatch and combine as autograd all-to-alls |
| TP | sequence parallelism with PyTorch's TP API: vocab-parallel embedding, column-parallel `wq wk wv w1 w3`, row-parallel `wo w2`, column-parallel `lm_head` with a vocab-parallel loss (the layout torchtitan's [decoder sharding](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/common/decoder_sharding.py) describes) |

Choices that keep the comparison exact: routing is balanced by construction (token `t` goes to experts
`(t + j·E/k) mod E`, so each rank sends exactly `1/ep` of its rows to every rank); RMSNorm has no weight (the engine
does not count norm weights). FSDP2 pads each parameter's first dimension to a multiple of the shard count, which the
engine does not model; the test allows 1%.

## How collectives are recorded

`scouter_sim/record.py` is a `TorchDispatchMode`. FSDP2 issues `c10d` ops and DTensor / functional collectives issue
`_c10d_functional` ops; the mode sees both. For ops on DTensors it returns `NotImplemented` first, so DTensor desugars
into plain ops and collectives that then come back through the mode (the same approach as PyTorch's `CommDebugMode`);
otherwise collectives issued inside an op, such as the vocab-parallel loss reductions, would be missed.

## Trace

```json
{ "schema": 1, "case": "…", "torch": "2.14.1+cpu", "backend": "gloo", "layout": {…}, "tok": 512,
  "mesh": { "dp": 4, "edp": 2, "ep": 2 }, "params": { "attn": …, "expert": …, … },
  "ranks": [ [ { "op": "AG", "dim": "dp", "n": 4, "dtype": "bfloat16", "elem": 2, "numel": 262144, "pass": "fwd" },
               { "op": "A2A", "dim": "ep", "n": 2, "dtype": "int64", "elem": 8, "numel": 8, "row": 1,
                 "send": [4, 4], "recv": [4, 4], "me": 0, "pass": "fwd" },
               { "op": "A2A", "dim": "ep", "n": 2, "dtype": "bfloat16", "elem": 2, "numel": 262144, "row": 256,
                 "send": [512, 512], "recv": [512, 512], "me": 0, "pass": "fwd" }, … ], … ] }
```

One list per rank. `op` is AG / RS / AR / A2A; `dim` is the mesh dimension the group belongs to; `numel` is the full
tensor (gathered output, scattered input, reduced tensor, or the all-to-all input); `send` / `recv` are rows per peer and
`me` is this rank's index in the group. The int64 all-to-all is the dispatcher's token-count exchange; the bf16 ones are
dispatch and combine. The first record above is the embedding's all-gather (V × D = 1024 × 256).
