"""Export HuggingArch simulations for Scouter's engine signatures.

    uv run --project tools/hga python tools/hga/export.py                # every real model
    uv run --project tools/hga python tools/hga/export.py --model kimik3 --kind infer --card B300

The signatures, the grid, the model map and the card map come from the JS
engine (`src/engine/hga.js`, read through `node`), so the two sides cannot
drift. HuggingArch is the git submodule at `vendor/HuggingArch` (a fork of
shenh10/HuggingArch, MIT): inference cells read one walk per (model,
signature, card) and every grid point off it (`capacity.iter_sweep_cell`);
training cells use the fork's `backend.training`. Output:
`src/data/hga/<model>.js`, merged with what is already there, so a partial
run only replaces the cells it priced.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HGA = ROOT / "vendor" / "HuggingArch"
OUT = ROOT / "src" / "data" / "hga"
sys.path.insert(0, str(HGA))

ROLE = {"attention": "attn", "ffn_moe": "moe", "ffn_moe_latent": "moe", "ffn_moe_shared": "dense",
        "ffn_dense": "dense", "router": "router"}
#: Fixed orders the JS side unpacks (`Hga.POINT`, `Hga.ROLES`): a point is a flat array.
ROLES = ["attn", "moe", "dense", "router", "other", "comm"]


def engine_tables() -> dict:
    js = ("const E=require('./src/engine/node.js')(); const H=E.Hga;"
          "const moe=Object.fromEntries(Object.keys(H.REAL).map(k=>[k, !!(E.MODELS[k].moe && E.MODELS[k].moe.E>0)]));"
          "const sigs=Object.fromEntries(Object.keys(H.REAL).map(k=>[k, Object.fromEntries(H.INFER.concat(H.TRAIN).map(e=>[e, H.sigsFor(e, k)]))]));"
          "process.stdout.write(JSON.stringify({SIGS:sigs, GRID:H.GRID, REAL:H.REAL, CARD:H.CARD, MOE:moe}))")
    return json.loads(subprocess.check_output(["node", "-e", js], cwd=ROOT))


def sig3(x: float) -> float:
    """Three significant digits: the export is a picture of a law, not the law."""
    if not x or not math.isfinite(x):
        return 0
    v = float(f"{x:.3g}")
    return int(v) if v.is_integer() and abs(v) < 1e15 else v


def aggregate(rows: list, t_phase: float) -> dict:
    """One phase's rows summed the way HuggingArch sums them (each row the max of
    its bounds), beside the per-bound sums, the per-role split and the bytes."""
    tc = tm = tn = s0 = fl = by = wb = kv = cb = 0.0
    nl = 0
    bb = [0.0, 0.0, 0.0, 0.0]
    role: dict = {}
    for r in rows:
        c, m, n, ln = (float(r.get(k) or 0.0) for k in ("t_compute_s", "t_memory_s", "t_comm_s", "t_launch_s"))
        t = float(r.get("t_sol_s") or 0.0)
        tc += c; tm += m; tn += n
        s0 += max(c, m, n)
        fl += float(r.get("flops_per_rank") or 0.0)
        by += float(r.get("access_bytes_per_rank") or 0.0)
        wb += float(r.get("weight_read_bytes_per_rank") or 0.0)
        kv += float(r.get("kv_read_bytes_per_rank") or 0.0)
        cb += float(r.get("comm_bytes_per_rank") or 0.0)
        if not r.get("is_comm") and not r.get("zero_cost") and t > 0:
            nl += int(r.get("count_per_step") or 1)
        bound = max((("compute", c), ("memory", m), ("comm", n), ("launch", ln)), key=lambda kv_: kv_[1])[0]
        bb[("compute", "memory", "comm", "launch").index(bound)] += t
        k = "comm" if r.get("is_comm") else ROLE.get(r.get("role") or "", "other")
        a = role.setdefault(k, [0.0, 0.0, 0.0, 0.0])
        a[0] += t; a[1] += c; a[2] += m; a[3] += n
    # t tc tm tn s0 fl by wb kv cb nl | bound: compute memory comm launch | per role (ROLES): t tc tm tn
    out = [sig3(x) for x in (t_phase, tc, tm, tn, s0, fl, by, wb, kv, cb)] + [nl] + [sig3(x) for x in bb]
    for k in ROLES:
        out += [sig3(x) for x in role.get(k, (0, 0, 0, 0))]
    return out


def infer_cell(model_id: str, engine: str, sig: dict, sku: str, grid: dict) -> dict:
    from backend.api._models import InferenceEstimateSweepRequest
    from backend.api.sweep_runtime import _estimate_plan
    from backend.inference.capacity import iter_sweep_cell
    req = InferenceEstimateSweepRequest(
        model_id=model_id, gpu_sku=sku, parallel=sig["hga"], framework=engine,
        prefer_measured=False, sol_only=True, avg_output_tokens=grid["osl"],
        prefill_chunk_size=None, seq_lens=grid["isl"],
        batch_grid=[{"key": f"b{b}", "prefill_batch_size": 1, "decode_batch_size": b} for b in grid["b"]])
    (cell,) = _estimate_plan(req).cells()
    pre, dec, max_b, warn = [], [], [], []
    for _index, points in iter_sweep_cell(cell):
        row_d, p0, mb = [], None, 0
        for pt in points:
            sol = pt["sol"]
            if sol.get("infeasibility_reason"):
                if (sol.get("memory") or {}).get("fits") is False:     # the weights do not fit: no context runs
                    return {"ok": False, "reason": sol["infeasibility_reason"]}
                row_d.append(None)                                      # this context holds no request
                continue
            warn = sol.get("parallel_warnings") or warn
            ph = sol["phases"]
            if p0 is None:
                p0 = aggregate(sol.get("per_op_timing") or [], float(ph["prefill"].get("t_sol_s") or 0.0))
            row_d.append(aggregate(sol.get("per_op_timing_decode") or [], float(ph["decode"].get("t_sol_s") or 0.0)))
            mb = sol.get("max_decode_batch_per_gpu")
        pre.append(p0)
        dec.append(row_d)
        max_b.append(int(mb or 0) if p0 is not None else 0)
    if all(p is None for p in pre):
        return {"ok": False, "reason": "no context fits one request"}
    return {"ok": True, "isl": grid["isl"], "b": grid["b"], "osl": grid["osl"], "maxB": max_b,
            "pre": pre, "dec": dec, "warn": warn}


def train_cell(model_id: str, fw: str, sig: dict, sku: str, grid: dict, moe: bool) -> dict:
    from backend.training import TrainingConfig, forward_points, parallel_for, training_step
    cfg = TrainingConfig(framework=fw, world=grid["world"], global_tokens=grid["gtok"], **sig["train"])
    par = parallel_for(cfg, moe)
    fwd = forward_points(model_id, sku, par, grid["seq"], grid["mbs"])
    pts, warn = [], []
    for seq in grid["seq"]:
        row = []
        for mbs in grid["mbs"]:
            r = training_step(fwd[(seq, mbs)], cfg, sku, seq=seq, micro_batch=mbs, moe=moe)
            warn = r["warnings"] or warn

            def pas(p: dict) -> list:
                return [sig3(p["t"]), sig3(p["compute"]), sig3(p["memory"]), sig3(p["comm"]), sig3(p["launch"]),
                        sig3(p["flops"]), sig3(p["bytes"]), sig3(p["comm_bytes"]), p["launches"]] + _roles(p["by_role"])
            st, mem = r["step"], r["memory"]
            row.append({"f": pas(r["micro"]["fwd"]), "b": pas(r["micro"]["bwd"]), "r": pas(r["micro"]["recompute"]),
                        "st": [st["micro_batches"], sig3(st["t_pipe"]), sig3(st["bubble"]), sig3(st["dp_comm"]["serial"]),
                               sig3(st["dp_comm"]["exposed"]), sig3(st["t_step"]), sig3(st["optimizer"]["t"]),
                               sig3(st["tokens_per_s_per_gpu"]), sig3(st["mfu"])],
                        "dp": [[p["name"], p["op"], p["world"], sig3(p["bytes"]), sig3(p["t"])] for p in st["dp_comm"]["parts"]],
                        "mem": [sig3(mem[k]) for k in ("weights", "grads", "optimizer", "activations", "total", "hbm")]})
        pts.append(row)
    return {"ok": True, "seq": grid["seq"], "mbs": grid["mbs"], "parallel": par, "pts": pts, "warn": warn,
            "assumptions": r["assumptions"]}


def _roles(by_role: dict) -> list:
    out = dict.fromkeys(ROLES, 0.0)
    for k, v in by_role.items():
        out["comm" if k == "comm" else ROLE.get(k, "other")] += v
    return [sig3(out[k]) for k in ROLES]


def hga_commit() -> str:
    """The fork commit the cells were walked on; "-dirty" when the walk ran on uncommitted code."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=HGA, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HGA, text=True).strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    start, end = text.index("{"), text.rindex("}")
    return json.loads(text[start:end + 1])


def save(path: Path, key: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    path.write_text(
        "/* 生成文件，不要手改：uv run --project tools/hga python tools/hga/export.py --model " + key + "\n"
        "   数据来自 HuggingArch（github.com/shenh10/HuggingArch，MIT）的 fork vendor/HuggingArch，提交 " + data["meta"]["hga"] + " */\n"
        "HGA_DATA.register(\"" + key + "\", " + body + ");\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--kind", choices=["infer", "train", "all"], default="all")
    ap.add_argument("--engine", action="append")
    ap.add_argument("--sig", action="append")
    ap.add_argument("--card", action="append")
    a = ap.parse_args()
    T = engine_tables()
    grid = T["GRID"]
    models = a.model or list(T["REAL"])
    cards = a.card or list(T["CARD"])
    kinds = ["infer", "train"] if a.kind == "all" else [a.kind]
    for mk in models:
        model_id = T["REAL"][mk]
        path = OUT / f"{mk}.js"
        data = load(path) or {"infer": {}, "train": {}}
        data["meta"] = {"model": mk, "hf": model_id, "hga": hga_commit(), "upstream": "shenh10/HuggingArch",
                        "generated": dt.date.today().isoformat(), "grid": grid}
        for kind in kinds:
            engines = [e for e in (["sglang", "vllm"] if kind == "infer" else ["megatron", "torchtitan"])
                       if not a.engine or e in a.engine]
            for eng in engines:
                for sig in T["SIGS"][mk][eng]:   # already resolved for this model (`Hga.sigsFor`: MoE-only and MoE expert placement)
                    if a.sig and sig["id"] not in a.sig:
                        continue
                    key = f"{eng}.{sig['id']}"
                    for ck in cards:
                        sku = T["CARD"][ck]
                        t0 = time.time()
                        try:
                            cell = (infer_cell(model_id, eng, sig, sku, grid) if kind == "infer"
                                    else train_cell(model_id, eng, sig, sku, grid, T["MOE"][mk]))
                        except Exception as e:  # a refusal is an answer: keep its reason
                            cell = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:600]}
                            if not isinstance(e, ValueError):
                                traceback.print_exc()
                        data.setdefault(kind, {}).setdefault(key, {})[ck] = cell
                        print(f"{mk} {kind} {key} {ck} ok={cell['ok']} {time.time() - t0:.1f}s"
                              + ("" if cell["ok"] else " — " + cell["reason"][:160]), flush=True)
                        save(path, mk, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
