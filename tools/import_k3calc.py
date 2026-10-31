"""把 k3-inference-calculator（ByronHsu/k3-inference-calculator，Apache-2.0）的解析下限导成公开数据集 data/public/k3_calculator.json。

它是 Kimi K3 的逐层 roofline 计算器：效率 100%、集合通信无启动延迟，expert 权重只算确定的最少 expert 数，结果是下限。
数字的来法：在它的网页（https://byronhsu.github.io/k3-inference-calculator/）的浏览器控制台里跑 tools/k3calc_snippet.js，
页面用自己的 Pyodide 跑它的 Python 估算器；把打印出来的几行存成文本，再用本工具转换：
  uv run --no-project tools/import_k3calc.py lines.txt --ref <k3-inference-calculator 的 commit>

每个场景（硬件、TP8、decode / prefill、batch、上下文或序列长度）一行 group = "total"（下限的步长），再按模块各一行
（FLOPs、HBM 字节、各算子时间之和）。flops_same / bytes_same 标出两边口径一致、可以直接对的行：
- FLOPs：KDA 核心（它每个状态元素算 1 FLOP，下限；本页和 attn-viz 算 8）和逐元素算子（它每个输出元素算 1 次）不对；
- 字节：routed expert（它只读确定的最少 expert，本页按均匀路由的期望覆盖）、逐元素算子不对；prefill 的激活读写口径也不同，只对 decode。
"""
import argparse, datetime, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "https://github.com/ByronHsu/k3-inference-calculator"
HW = {"b300": "B300", "gb300": "GB300", "h200": "H200"}
TOPO = {"b300": "hgx8", "gb300": "nvl64", "h200": "hgx8"}
FLOPS_DIFF = {"kda_core", "other", "comm"}
BYTES_SAME = {"mla_proj", "mla_core", "kda_proj", "kda_core", "moe_rest", "dense", "lm"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines"); ap.add_argument("--ref", required=True, help="k3-inference-calculator 的 commit SHA")
    ap.add_argument("--retrieved", default=datetime.date.today().isoformat())
    ap.add_argument("--out", default=str(ROOT / "data" / "public" / "k3_calculator.json"))
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.ref): sys.exit("--ref 要完整的 40 位 commit SHA")
    url = f"{REPO}/blob/{a.ref}/runtime/python/sglang/benchmark/kimi_k3_theoretical/estimator.py"
    rows = []
    for line in pathlib.Path(a.lines).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        if line.startswith("error"): sys.exit("计算器报错：" + line)
        head, body = line.split(" | ")
        phase, hw, B, L, total, _sum, link, mb = head.split()
        B, L = int(B), int(L)
        sid = f"{hw}-tp8-{phase}-b{B}-{'ctx' if phase == 'decode' else 'seq'}{L}"
        base = {"src": "k3-inference-calculator", "url": url, "model": "kimik3", "hw": HW[hw], "topo": TOPO[hw], "tp": 8, "moePar": "tp",
                "phase": phase, "batch": B, "ctx": L if phase == "decode" else 0, "seq": L if phase == "prefill" else 0,
                "ref": a.ref, "retrieved": a.retrieved}
        rows.append(dict(id=sid + "-total", group="total", total_s=float(total), link_bytes=float(link), flops_same=False, bytes_same=False, **base))
        for item in body.split():
            g, v = item.split(":"); fl, by, t = (float(x) for x in v.split(","))
            rows.append(dict(id=f"{sid}-{g}", group=g, flops=fl, bytes=by, t_s=t,
                             flops_same=g not in FLOPS_DIFF, bytes_same=g in BYTES_SAME and phase == "decode", **base))
    pathlib.Path(a.out).write_text("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows) + "\n]\n", encoding="utf-8")
    print("%d 行 → %s；再跑 uv run --no-project tools/build_db.py" % (len(rows), a.out))


if __name__ == "__main__":
    main()
