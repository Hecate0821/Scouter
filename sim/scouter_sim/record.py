"""Record every collective a process issues, at the dispatcher level.

FSDP2 issues c10d ops (torch.ops.c10d.*); DTensor and the functional collectives issue torch.ops._c10d_functional.*.
A TorchDispatchMode sees both, so nothing here knows which framework issued a collective. A record keeps raw facts
only: op, process group, dtype, element counts and all-to-all splits. Turning them into a cost model's byte convention
is the consumer's job (tests/sim.js in Scouter).
"""
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode

# op name -> (kind, positions); a tensor list at the input position means a coalesced call
C10D = {
    "_allgather_base_": ("AG", dict(out=0, inp=1, pg=2)),
    "allgather_into_tensor_coalesced_": ("AG", dict(out=0, inp=1, pg=2)),
    "_reduce_scatter_base_": ("RS", dict(out=0, inp=1, pg=2)),
    "reduce_scatter_tensor_coalesced_": ("RS", dict(out=0, inp=1, pg=2)),
    "allreduce_": ("AR", dict(inp=0, pg=1)),
    "alltoall_base_": ("A2A", dict(out=0, inp=1, pg=2, osplit=3, isplit=4)),
}
FUNCTIONAL = {
    "all_gather_into_tensor": ("AG", dict(inp=0, n=1, pg=2)),
    "all_gather_into_tensor_out": ("AG", dict(inp=0, n=1, pg=2)),
    "all_gather_into_tensor_coalesced": ("AG", dict(inp=0, n=1, pg=2)),
    "reduce_scatter_tensor": ("RS", dict(inp=0, n=2, pg=3)),
    "reduce_scatter_tensor_coalesced": ("RS", dict(inp=0, n=2, pg=3)),
    "all_reduce": ("AR", dict(inp=0, pg=2)),
    "all_reduce_coalesced": ("AR", dict(inp=0, pg=2)),
    "all_to_all_single": ("A2A", dict(inp=0, osplit=1, isplit=2, pg=3)),
}


def _group(obj):
    if isinstance(obj, str):
        return _resolve_process_group(obj)
    return dist.ProcessGroup.unbox(obj)  # c10d ops carry the group as a TorchBind object


def _tensors(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


class CommRecorder(TorchDispatchMode):
    """with CommRecorder(dims) as rec: ... ; rec.phase = "fwd" / "bwd" labels what follows.
    dims maps a frozenset of global ranks to the name of that process group (a mesh dimension)."""

    def __init__(self, dims):
        super().__init__()
        self.dims, self.phase, self.records = dims, "fwd", []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Let DTensor run first: it desugars into plain ops and collectives, which then come back through this mode
        # (the same trick as torch.distributed.tensor.debug.CommDebugMode). Without it, collectives DTensor issues
        # inside an op (vocab-parallel loss, redistributes during sharding propagation) would be missed.
        if any(issubclass(t, DTensor) for t in types):
            return NotImplemented
        kwargs = kwargs or {}
        table = C10D if func.namespace == "c10d" else FUNCTIONAL if func.namespace == "_c10d_functional" else None
        spec = table and table.get(func._schema.name.split("::")[-1])
        if spec:
            self._record(spec, args)
        return func(*args, **kwargs)

    def _record(self, spec, args):
        kind, at = spec
        pg = _group(args[at["pg"]])
        ranks = dist.get_process_group_ranks(pg)
        n = len(ranks)
        for i, x in enumerate(_tensors(args[at["inp"]])):
            r = dict(pass_=self.phase, op=kind, dim=self.dims.get(frozenset(ranks), "?"), n=n,
                     dtype=str(x.dtype).replace("torch.", ""), elem=x.element_size())
            if kind == "AG":   # full = gathered output
                r["numel"] = x.numel() * n
            elif kind in ("RS", "AR"):   # full = the input every rank contributes
                r["numel"] = x.numel()
            else:   # A2A: rows sent to / received from each peer, and this rank's position in the group
                isplit, osplit = args[at["isplit"]], args[at["osplit"]]
                rows = x.shape[0] if x.dim() else 1
                isplit = list(isplit) if isplit else [rows // n] * n
                osplit = list(osplit) if osplit else [rows // n] * n
                r.update(numel=x.numel(), row=x.numel() // max(1, rows), send=isplit, recv=osplit, me=ranks.index(dist.get_rank()))
            self.records.append(r)
