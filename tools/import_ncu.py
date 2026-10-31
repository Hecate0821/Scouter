"""把 Nsight Compute 的逐 kernel 报告导成数据点库的一个匿名数据集（data/anon/<name>.json，stage = kernel）。

先用 ncu 导出原始指标（基本单位，方便换算）：
  ncu --csv --page raw --print-units base \\
      --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\\
sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \\
      -o - python bench.py > report.csv
再导入：
  uv run --no-project tools/import_ncu.py report.csv --name b200_moe --hw B200 --dtype fp4 \\
      --conditions "单卡，DeepSeek 形状的 MoE 层，batch 64" --clocks base --ncu 2025.3 [--flops flops.json]

每个 kernel（按 ID）一个点：kernel（名字）、us（时长）、dram_bytes（DRAM 读 + 写）、sm_pct / dram_pct（ncu 报的占峰值百分比）。
--flops 给一个 JSON：{ "kernel 名或 ID": 解析 FLOPs }，有了就能算 ηc；ηm 和 ηc 由 Scouter.kernels 按 SOL 口径算（字节或 FLOPs ÷ 时长 ÷ 峰值）。
测量条件写进数据集的 profile（工具、版本、时钟、replay 方式、指标），和 conditions 一起显示。匿名数据集只留条件和数字：
kernel 名照 ncu 原样保留，描述里不要写机器名、路径和内部项目名（tools/build_db.py 会查）。
"""
import argparse, csv, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COLS = {"us": "gpu__time_duration.sum", "rd": "dram__bytes_read.sum", "wr": "dram__bytes_write.sum",
        "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed", "dram_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"}
# ncu 的单位（--print-units base 时是 nsecond / byte）；其余按十进制前缀换算
SCALE = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6,
         "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "%": 1}


def num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return None


def read(path):
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    head = next(i for i, r in enumerate(rows) if "Kernel Name" in r and "ID" in r)
    names, body = rows[head], rows[head + 1:]
    units = {}
    if body and num(body[0][names.index("ID")]) is None:      # 第二行是单位
        units = dict(zip(names, body[0])); body = body[1:]
    idx = {k: names.index(c) for k, c in COLS.items() if c in names}
    for need in ("us", "rd", "wr"):
        if need not in idx: sys.exit("缺指标 %s（见本文件开头的 ncu 命令）" % COLS[need])
    def val(r, k):
        v = num(r[idx[k]]) if k in idx else None
        u = units.get(COLS[k], "")
        if v is not None and u and u not in SCALE: sys.exit("不认识的单位 %r（用 --print-units base 导出）" % u)
        return None if v is None else v * SCALE.get(u, 1)
    out = []
    for r in body:
        if len(r) < len(names) or num(r[names.index("ID")]) is None: continue
        us = val(r, "us")
        if not units.get(COLS["us"]): us = us * 1e-3 if us is not None else None   # 没有单位行时按 ncu 默认的 nsecond
        p = {"kernel": r[names.index("Kernel Name")], "id": int(num(r[names.index("ID")])), "us": round(us, 4),
             "dram_bytes": int(val(r, "rd") + val(r, "wr")), "sm_pct": val(r, "sm_pct"), "dram_pct": val(r, "dram_pct"), "flops": None}
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv"); ap.add_argument("--name", required=True, help="数据集 id（文件名），小写字母、数字和下划线")
    ap.add_argument("--hw", required=True); ap.add_argument("--dtype", default="bf16", help="算 ηc 用哪个精度的峰值")
    ap.add_argument("--conditions", required=True); ap.add_argument("--title")
    ap.add_argument("--clocks", default="unknown", help="base（锁在基础频率）/ locked / unlocked")
    ap.add_argument("--ncu", default="unknown", help="Nsight Compute 版本"); ap.add_argument("--replay", default="kernel")
    ap.add_argument("--flops", help="JSON：{kernel 名或 ID: 解析 FLOPs}")
    ap.add_argument("--out", help="输出路径，默认 data/anon/<name>.json")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", a.name): sys.exit("--name 只能用小写字母、数字和下划线")
    pts = read(a.csv)
    if not pts: sys.exit("报告里没有 kernel 行")
    fl = json.load(open(a.flops)) if a.flops else {}
    for p in pts: p["flops"] = fl.get(str(p["id"]), fl.get(p["kernel"]))
    s = {"title": a.title or "Nsight Compute：" + a.name, "stage": "kernel", "hw": a.hw, "gpus": 1, "topo": None, "model": None,
         "conditions": a.conditions,
         "profile": {"tool": "Nsight Compute", "version": a.ncu, "clocks": a.clocks, "replay": a.replay, "dtype": a.dtype, "metrics": [COLS[k] for k in COLS]},
         "dims": ["kernel", "id"],
         "metrics": {"us": {"unit": "µs", "label": "时长"}, "dram_bytes": {"unit": "B", "label": "DRAM 读 + 写"},
                     "sm_pct": {"unit": "%", "label": "SM 吞吐 / 峰值"}, "dram_pct": {"unit": "%", "label": "访存吞吐 / 峰值"}, "flops": {"unit": "FLOP", "label": "解析 FLOPs"}},
         "points": pts}
    out = pathlib.Path(a.out) if a.out else ROOT / "data" / "anon" / (a.name + ".json")
    out.write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("%d 个 kernel → %s；再跑 uv run --no-project tools/build_db.py" % (len(pts), out))


if __name__ == "__main__":
    main()
