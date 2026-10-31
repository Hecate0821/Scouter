"""Run one micro-batch (forward + backward) of a case on CPU gloo processes and write the collectives as a JSON trace.

    uv run python -m scouter_sim.run                  every case in cases/
    uv run python -m scouter_sim.run cases/x.json     one case

A case is {name, model, layout, tok}: `model` uses the Scouter engine's model schema, `layout` names the parallel sizes
(N, tp, ep) and `tok` is the tokens per rank per micro-batch. The parallelism follows torchtitan:
  - FSDP2 like torchtitan.distributed.fsdp.apply_fsdp_to_decoder: tok_embeddings in its own unit with no backward
    all-gather, [norm, lm_head] in one unit that does not reshard after forward, every block resharded after forward,
    MoE blocks with routed experts on the expert-FSDP mesh and everything else on the dense FSDP mesh, bf16 params,
    fp32 gradient reduction;
  - EP with the all-to-all dispatcher of torchtitan.models.common.moe.token_dispatcher (see model.MoE);
  - TP with sequence parallelism: vocab-parallel embedding, column-parallel wq / wk / wv / w1 / w3, row-parallel wo / w2,
    column-parallel lm_head with a vocab-parallel loss.
The trace holds what PyTorch actually issued on every rank; it knows nothing about Scouter's cost model.
"""
import json
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import ShardPlacementResult
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel, SequenceParallel,
                                               loss_parallel, parallelize_module)

from .model import Decoder, MoE, param_counts
from .record import CommRecorder

ROOT = Path(__file__).resolve().parent.parent


def meshes(lay):
    """Rank layout: TP innermost, then EP, then the rest of data parallel (torchtitan keeps TP and EP inside a node)."""
    N, tp, ep = lay["N"], lay.get("tp", 1), lay.get("ep", 1)
    dp = N // tp
    if tp > 1:
        mesh = init_device_mesh("cpu", (dp, tp), mesh_dim_names=("dp", "tp"))
        return dict(dp=mesh["dp"], tp=mesh["tp"])
    if ep > 1:
        mesh = init_device_mesh("cpu", (dp // ep, ep), mesh_dim_names=("edp", "ep"))
        return dict(dp=mesh._flatten("dp"), edp=mesh["edp"], ep=mesh["ep"])
    return dict(dp=init_device_mesh("cpu", (dp,), mesh_dim_names=("dp",)))


def apply_tp(model, tp_mesh):
    parallelize_module(model, tp_mesh, {
        "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "norm": SequenceParallel(),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    })
    for blk in model.layers.values():
        if isinstance(blk.feed_forward, MoE):
            raise ValueError("sim: TP on MoE blocks is not modeled; use EP")
        parallelize_module(blk, tp_mesh, {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "attention.wq": ColwiseParallel(), "attention.wk": ColwiseParallel(), "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            "feed_forward.w1": ColwiseParallel(), "feed_forward.w3": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        })


def apply_fsdp(model, M):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    cfg = dict(mesh=M["dp"], mp_policy=mp_policy)
    fully_shard(model.tok_embeddings, **cfg, reshard_after_forward=True)
    model.tok_embeddings.set_unshard_in_backward(False)        # embedding backward never reads the weight
    fully_shard([model.norm, model.lm_head], **cfg, reshard_after_forward=False)
    for blk in model.layers.values():
        if isinstance(blk.feed_forward, MoE) and "edp" in M:
            experts = set(blk.feed_forward.experts.parameters())
            edp_info, dp_info = _get_mesh_info(M["edp"], None), _get_mesh_info(M["dp"], None)
            fully_shard(blk, **cfg, reshard_after_forward=True,
                        shard_placement_fn=lambda p, e=experts: ShardPlacementResult(placement=Shard(0), mesh_info=edp_info if p in e else dp_info))
        else:
            fully_shard(blk, **cfg, reshard_after_forward=True)
    fully_shard(model, **cfg)


def worker(rank, case, port, out):
    dist.init_process_group("gloo", rank=rank, world_size=case["layout"]["N"], init_method=f"tcp://127.0.0.1:{port}")
    torch.manual_seed(0)
    M = meshes(case["layout"])
    dims = {frozenset(dist.get_process_group_ranks(m.get_group())): name for name, m in M.items()}
    model = Decoder(case["model"], ep_group=M["ep"].get_group() if "ep" in M else None)
    if "tp" in M:
        apply_tp(model, M["tp"])
    apply_fsdp(model, M)
    torch.manual_seed(1 + (dist.get_rank(M["dp"].get_group())))   # different data per data-parallel rank, same within TP
    tokens = torch.randint(0, case["model"]["V"], (1, case["tok"]))
    labels = torch.randint(0, case["model"]["V"], (1, case["tok"]))
    rec = CommRecorder(dims)
    with rec:
        logits = model(tokens)
        if "tp" in M:
            with loss_parallel():
                loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())
                rec.phase = "bwd"
                loss.backward()
        else:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())
            rec.phase = "bwd"
            loss.backward()
    records = [dict(r, **{"pass": r.pop("pass_")}) for r in rec.records]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, records)
    if rank == 0:
        trace = dict(schema=1, case=case["name"], torch=torch.__version__, backend="gloo", layout=case["layout"], tok=case["tok"],
                     mesh={k: m.size() for k, m in M.items()}, params=param_counts(case["model"]),
                     note="collectives issued on each rank during one micro-batch (forward + backward); raw sizes, no cost model",
                     ranks=gathered)
        Path(out).write_text(json.dumps(trace, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(paths):
    paths = [Path(p) for p in paths] or sorted((ROOT / "cases").glob("*.json"))
    for p in paths:
        case = json.loads(p.read_text())
        out = ROOT / "traces" / p.name
        mp.spawn(worker, args=(case, free_port(), str(out)), nprocs=case["layout"]["N"], join=True)
        n = sum(len(r) for r in json.loads(out.read_text())["ranks"])
        print(f"{case['name']}: {n} collectives on {case['layout']['N']} ranks -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1:])
