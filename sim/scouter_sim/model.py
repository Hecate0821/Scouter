"""A small decoder built from a Scouter model description (the engine's MODELS schema).

Supported subset: full GQA attention on every layer (optionally with the output gate), dense SwiGLU FFN for the first
dense.n layers, then MoE (router, stacked SwiGLU experts, optional shared expert and shared-expert gate). Parameter
shapes match Common.derive() in Scouter's engine term by term, so per-layer parameter counts agree exactly. RMSNorm
has no weight because the engine does not count norm weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol


def check(m):
    f, l, e = m["full"], m["lin"], m["moe"]
    if f["kind"] != "gqa" or f["n"] != m["L"] or l["n"] or f.get("swa"):
        raise ValueError("sim supports full GQA attention on every layer only")
    if e.get("Dlat"):
        raise ValueError("sim does not support MoE latent projections (Dlat)")


class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, x.shape[-1:])


def q_width(f):
    return f["Hq"] * f["dh"] + ((f["Hq"] if f.get("gateHead") else f["Hq"] * f["dh"]) if f.get("gate") else 0)


class Attention(nn.Module):
    def __init__(self, D, f):
        super().__init__()
        self.f, dh = f, f["dh"]
        self.wq = nn.Linear(D, q_width(f), bias=False)   # with the gate, each head's query and gate sit side by side
        self.wk = nn.Linear(D, f["Hkv"] * dh, bias=False)
        self.wv = nn.Linear(D, f["Hkv"] * dh, bias=False)
        self.wo = nn.Linear(f["Hq"] * dh, D, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        f, dh = self.f, self.f["dh"]
        q = self.wq(x)
        hq = q.shape[-1] // (q_width(f) // f["Hq"])   # heads on this rank (TP shards whole heads)
        q = q.view(B, S, hq, -1)
        gate = None
        if f.get("gate"):
            q, gate = q[..., :dh], q[..., dh:]
        k = self.wk(x).view(B, S, -1, dh)
        v = self.wv(x).view(B, S, -1, dh)
        rep = hq // k.shape[2]
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return self.wo(o.reshape(B, S, -1))


class FeedForward(nn.Module):
    def __init__(self, D, Fd):
        super().__init__()
        self.w1, self.w3 = nn.Linear(D, Fd, bias=False), nn.Linear(D, Fd, bias=False)
        self.w2 = nn.Linear(Fd, D, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Experts(nn.Module):
    """The experts this EP rank owns, stacked like torchtitan's GroupedExperts: w1 / w3 [e, F, D], w2 [e, D, F]."""

    def __init__(self, n, D, Fd):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, Fd, D) * D ** -0.5)
        self.w3 = nn.Parameter(torch.randn(n, Fd, D) * D ** -0.5)
        self.w2 = nn.Parameter(torch.randn(n, D, Fd) * Fd ** -0.5)

    def forward(self, x, j):
        return F.silu(x @ self.w1[j].t()) * (x @ self.w3[j].t()) @ self.w2[j].t()


class MoE(nn.Module):
    """Top-k MoE with the all-to-all token dispatch of torchtitan's AllToAllTokenDispatcher: an int64 count exchange,
    then dispatch and combine as autograd all-to-alls (so backward issues the mirrored pair).
    Routing is balanced by construction: token t (numbered across the EP group) goes to experts (t + j·E/k) mod E,
    which sends exactly 1/ep of every rank's rows to each rank, the uniform-routing case."""

    def __init__(self, D, e, ep_group=None):
        super().__init__()
        self.E, self.k, self.group = e["E"], e["k"], ep_group
        self.ep = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.router = nn.Linear(D, self.E, bias=False)
        self.experts = Experts(self.E // self.ep, D, e["F"])
        self.shared = FeedForward(D, e["Fs"]) if e.get("Fs") else None
        self.shared_gate = nn.Linear(D, 1, bias=False) if e.get("Fs") and e.get("sharedGate") else None

    def forward(self, x):
        B, S, D = x.shape
        xt = x.reshape(-1, D)
        T, E, k, ep = xt.shape[0], self.E, self.k, self.ep
        rank = dist.get_rank(self.group) if self.group is not None else 0
        tok = torch.arange(T) + rank * T
        ids = torch.stack([(tok + j * (E // k)) % E for j in range(k)], dim=1)                 # [T, k]
        scores = torch.softmax(self.router(xt).float(), dim=-1).gather(1, ids).to(x.dtype)     # keeps the router in the graph
        order = torch.argsort(ids.reshape(-1), stable=True)
        rows = xt.repeat_interleave(k, dim=0)[order]                                           # expert-sorted [T·k, D]
        per_expert = torch.bincount(ids.reshape(-1), minlength=E)
        if ep > 1:
            counts = torch.empty_like(per_expert)
            dist.all_to_all_single(counts, per_expert, group=self.group)                       # count exchange
            isplit = per_expert.view(ep, -1).sum(1).tolist()
            recv = counts.view(ep, -1)                                                         # [source rank, local expert]
            osplit = recv.sum(1).tolist()
            got = funcol.all_to_all_single_autograd(rows, osplit, isplit, self.group)          # dispatch
            got = funcol.wait_tensor(got) if hasattr(funcol, "wait_tensor") else got
        else:
            recv, got, isplit, osplit = per_expert.view(1, -1), rows, None, None
        out, at = [], 0
        for src in range(recv.shape[0]):
            for j in range(recv.shape[1]):
                c = int(recv[src, j])
                out.append(self.experts(got[at:at + c], j))
                at += c
        y = torch.cat(out)
        if ep > 1:
            y = funcol.all_to_all_single_autograd(y, isplit, osplit, self.group)               # combine
            y = funcol.wait_tensor(y) if hasattr(funcol, "wait_tensor") else y
        y = torch.empty_like(y).index_copy(0, order, y).view(T, k, D)
        y = (y * scores.unsqueeze(-1)).sum(1)
        if self.shared is not None:
            s = self.shared(xt)
            y = y + (s * torch.sigmoid(self.shared_gate(xt)) if self.shared_gate is not None else s)
        return y.view(B, S, D)


class Block(nn.Module):
    def __init__(self, m, i, ep_group=None):
        super().__init__()
        D = m["D"]
        self.attention_norm, self.ffn_norm = RMSNorm(), RMSNorm()
        self.attention = Attention(D, m["full"])
        self.feed_forward = FeedForward(D, m["dense"]["F"]) if i < m["dense"]["n"] else MoE(D, m["moe"], ep_group)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class Decoder(nn.Module):
    def __init__(self, m, ep_group=None):
        super().__init__()
        check(m)
        self.tok_embeddings = nn.Embedding(m["V"], m["D"])
        self.layers = nn.ModuleDict({str(i): Block(m, i, ep_group) for i in range(m["L"])})
        self.norm = RMSNorm()
        self.lm_head = nn.Linear(m["D"], m["V"], bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        for blk in self.layers.values():
            h = blk(h)
        return self.lm_head(self.norm(h))


def param_counts(m):
    """The same per-layer counts as Common.derive() for the supported subset."""
    D, f, e = m["D"], m["full"], m["moe"]
    attn = D * (q_width(f) + 2 * f["Hkv"] * f["dh"]) + f["Hq"] * f["dh"] * D
    shared = 3 * D * e.get("Fs", 0) + (D if e.get("Fs") and e.get("sharedGate") else 0)
    return dict(attn=attn, dense=3 * D * m["dense"]["F"], expert=3 * D * e.get("F", 0), router=D * e.get("E", 0), shared=shared,
                emb=m["V"] * D, head=m["V"] * D)
