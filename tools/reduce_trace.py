"""把很大的 torch profiler / Chrome trace（.json 或 .json.gz）缩成只剩 GPU kernel 的 trace，给推理页和 scouter.js trace 用。

profiler 导出的 trace 大头是 CPU 侧的 python_function / cuda_runtime 事件（一份 5 步的 decode trace 解压后可以到几百 MB，
浏览器和 Node 都读不进一个字符串）。这里只留 cat 是 kernel / gpu_op 的 X 事件，每个只保留 name、ts、dur、pid、tid 和
args.source_idx（合并过的多 rank trace 靠它分 rank），PDL 重叠、按 rank 平均这些口径照样能算。
加 --summary 则直接按 kernel 名汇总成 [{name, us, calls}]（没有时间轴，算不了 PDL 重叠和空闲）。

整个文件要读进内存：几百 MB 的 trace 大约要几 GB 内存。

  uv run --no-project tools/reduce_trace.py trace.json.gz > kernels.json
  node tools/scouter.js trace --model kimik3 --hw B300 --B 64 --steps 5 --file.trace kernels.json
"""
import argparse, gzip, json, re, sys

KERNEL = re.compile(r"kernel|gpu_op", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace"); ap.add_argument("--summary", action="store_true", help="按 kernel 名汇总成 [{name, us, calls}]")
    a = ap.parse_args()
    with (gzip.open if a.trace.endswith(".gz") else open)(a.trace, "rt", encoding="utf-8") as f: tr = json.load(f)
    evs = tr.get("traceEvents", []) if isinstance(tr, dict) else tr
    ks = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X" and isinstance(e.get("dur"), (int, float)) and KERNEL.search(str(e.get("cat", "")))]
    if not ks: sys.exit("trace 里没有 GPU kernel 事件（cat 是 kernel / gpu_op 的 X 事件）")
    if a.summary:
        by = {}
        for e in ks:
            k = by.setdefault(str(e["name"]), {"name": str(e["name"]), "us": 0.0, "calls": 0}); k["us"] += e["dur"]; k["calls"] += 1
        out = sorted(by.values(), key=lambda k: -k["us"])
    else:
        def slim(e):
            r = {"ph": "X", "cat": "kernel", "name": e["name"], "ts": e.get("ts"), "dur": e["dur"], "pid": e.get("pid"), "tid": e.get("tid")}
            src = (e.get("args") or {}).get("source_idx")
            if src is not None: r["args"] = {"source_idx": src}
            return r
        out = {"traceEvents": [slim(e) for e in ks]}
    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    print("%d 个 kernel 事件（原来 %d 个事件）" % (len(ks), len(evs)), file=sys.stderr)


if __name__ == "__main__":
    main()
