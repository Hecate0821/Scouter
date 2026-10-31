"""把 SGLang cookbook 的公开测速数据导成数据点库的一个公开数据集（data/public/sglang_cookbook.json）。

来源：sgl-project/sglang（Apache-2.0）的 docs/src/snippets/configs/<org>/<model>-benchmarks.jsx。每个文件是一个 JS 对象字面量，
这里按 token 解析成 JSON（不执行下载的代码），钉在一个 commit 上，每一行带指向这个 commit 的 url。

只导 Scouter 有对应模型的文件：GLM-5.3-Flash（glm53f）和 Kimi K3（kimik3）。Qwen3.8-27B 的行是在未公开的 checkpoint 上测的，不导。
benchmarks 文件只有 hw / strategy / quant 这些选项名，部署形状（几张卡、TP / EP、KV 精度、投机深度、接受长度）在同目录的配方文件和
每条的 notes 里，由下面的 DEPLOY 手工对上，每条注明出处。对不上的行照样导入，calib 记为 false，只作数据展示。

tok/s/GPU 的口径是 (输入 + 输出) token / 时间 / 卡数（GLM-5.3-Flash GB300 c=1：2000 / (0.205 + 999 × 0.002) / 4 ≈ 227）。
并发 1 的行用这个关系反推卡数，和 DEPLOY 的卡数对不上就报错。TPOT 按 bench_serving / genai-bench 的口径是 (E2E − TTFT) / (osl − 1)。
num_prompts 不到并发两倍的行是一波突发（GLM-5.3-Flash 的 c80 是 80 条请求、80 并发：全体一起排队 prefill，不是稳态），calib 记为 false。
接受长度在 random 数据集上偏高（随机 token 的 prompt 容易让输出进入循环，草稿更容易猜中；jhinpan/sglang-amd-cookbook 的 Kimi K3 记录里
random 上 3.8，真实 prompt 上 2.33–2.94），而且 bench_serving 报的是服务端从启动以来的累计平均，不重置时会混进前面几格：当作上限看。

用法：
  uv run --no-project tools/import_cookbook.py                  # 取 sglang main 的当前 commit
  uv run --no-project tools/import_cookbook.py --ref <sha>      # 钉在指定 commit
然后 uv run --no-project tools/build_db.py 重建数据点库。
"""
import argparse, json, pathlib, re, sys, urllib.request
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "public" / "sglang_cookbook.json"
REPO = "sgl-project/sglang"
DIR = "docs/src/snippets/configs/"
FILES = {"glm53f": "zai-org/glm-5.3-flash", "kimik3": "moonshotai/kimi-k3"}
HW = {"h100": "H100", "h200": "H200", "b200": "B200", "b300": "B300", "gb200": "GB200", "gb300": "GB300"}
TOPO = {"gb200": "nvl64", "gb300": "nvl64"}

# 部署形状：model、match 的子集 → 本行的配置。spec：none / mtp（固定深度）/ mtp-adaptive / dspark；acc 是每步平均接受长度（含 bonus token），
# 可以按 isl 和并发给成表；sim 表示接受长度由 SGLANG_SIMULATE_ACC_LEN 定死。calib=False 的不进校准。
DEPLOY = [
 # 配方 Low Latency 卡片 "MTP 5/1/6"；notes：TP4 / EP1，flashinfer_trtllm MoE，固定 MTP 5/1/6，自然接受，FP8 KV + TRT-LLM DSA；
 # 接受长度 chat（1000 / 1000）5.98 / 5.98 / 4.7，summarization（8000 / 1000）6.0 / 6.0 / 4.5，对应并发 1 / 16 / 80
 ("glm53f", {"hw": "gb300", "strategy": "low-latency", "kvDsaPair": "fp8-trtllm", "quant": "fp8"},
  dict(N=4, moePar="tp", kvDt="fp8", spec="mtp", specK=5, acc={1000: {1: 5.98, 16: 5.98, 80: 4.7}, 8000: {1: 6.0, 16: 6.0, 80: 4.5}}, sim=False, calib=True)),
 # notes：TP4 / EP4（deep_gemm），BF16 KV + TileLang，投机关闭
 ("glm53f", {"hw": "gb300", "strategy": "high-throughput"}, dict(N=4, moePar="replicated", kvDt="bf16", spec="none", calib=True)),
 # notes：FP8 KV + TRT-LLM DSA，和 BF16 那几行同一协议（文件里的 TODO：仍是 TP4 / EP4 deep_gemm 命令）
 ("glm53f", {"hw": "gb300", "strategy": "high-throughput", "kvDsaPair": "fp8-trtllm", "quant": "fp8"},
  dict(N=4, moePar="replicated", kvDt="fp8", spec="none", calib=True)),
 # TODO 注释：早先 TP4 / EP4 命令的自适应 MTP，SGLANG_SIMULATE_ACC_LEN=3；自适应的草稿深度不知道，不进校准
 ("glm53f", {"hw": "gb300", "strategy": "low-latency"}, dict(N=4, moePar="replicated", kvDt="bf16", spec="mtp-adaptive", acc=3.0, sim=True, calib=False)),
 # notes：8 卡 TP8 / EP8，BF16 KV + TileLang DSA，SGLANG_SIMULATE_ACC_LEN=3；B200 / B300 的 TODO 注释说是自适应 MTP，H100 同一轮测量
 ("glm53f", {"hw": "h100", "strategy": "low-latency"}, dict(N=8, moePar="replicated", kvDt="bf16", spec="mtp-adaptive", acc=3.0, sim=True, calib=False)),
 ("glm53f", {"hw": "b200", "strategy": "low-latency"}, dict(N=8, moePar="replicated", kvDt="bf16", spec="mtp-adaptive", acc=3.0, sim=True, calib=False)),
 ("glm53f", {"hw": "b300", "strategy": "low-latency"}, dict(N=8, moePar="replicated", kvDt="bf16", spec="mtp-adaptive", acc=3.0, sim=True, calib=False)),
 # 配方 B300 Low Latency 单元：--tp-size 8，没有 EP / DCP；KV 默认 Auto（BF16）
 ("kimik3", {"hw": "b300", "strategy": "low-latency", "spec": "none"}, dict(N=8, moePar="tp", kvDt="bf16", spec="none", calib=True)),
 # 配方 B300 Balanced 单元：--tp-size 8 --dcp-size 8
 ("kimik3", {"hw": "b300", "strategy": "balanced", "spec": "none"}, dict(N=8, moePar="tp", kvDt="bf16", dcp=8, spec="none", calib=True)),
 # DSPARK 在 B300 上的草稿数和接受长度没有公布，不进校准
 ("kimik3", {"hw": "b300", "strategy": "low-latency", "spec": "dspark"}, dict(N=8, moePar="tp", kvDt="bf16", spec="dspark", calib=False)),
 ("kimik3", {"hw": "b300", "strategy": "balanced", "spec": "dspark"}, dict(N=8, moePar="tp", kvDt="bf16", dcp=8, spec="dspark", calib=False)),
]
# 量化选项 → 推理配置的权重精度（和低精度推理页的方案表一致）。没有 quant 的条目是模型默认的 checkpoint
QUANT = {"fp8": dict(wDt="fp8", dDt="fp8"), "mxfp4": dict(wDt="mxfp4", dDt="bf16"), "nvfp4": dict(wDt="nvfp4", dDt="bf16")}
DEFAULT_QUANT = {"glm53f": "fp8", "kimik3": "mxfp4"}


def parse_js(src):
    """JS 对象 / 数组字面量 → Python：字符串、数字、true / false / null、不带引号的键、注释、尾逗号。"""
    tok = re.compile(r'\s+|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`$]*`|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|[A-Za-z_$][\w$]*|[{}\[\]:,]', re.S)
    toks, i = [], 0
    while i < len(src):
        m = tok.match(src, i)
        if not m: raise ValueError("无法解析的字符 %r（位置 %d）" % (src[i:i + 20], i))
        t = m.group(0); i = m.end()
        if t.strip() and not t.startswith("//") and not t.startswith("/*"): toks.append(t)
    pos = 0
    def val():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t == "{":
            o = {}
            while toks[pos] != "}":
                k = toks[pos]; pos += 1
                k = json.loads(k) if k[0] == '"' else k[1:-1] if k[0] == "'" else k
                if toks[pos] != ":": raise ValueError("缺冒号：" + k)
                pos += 1; o[k] = val()
                if toks[pos] == ",": pos += 1
            pos += 1; return o
        if t == "[":
            a = []
            while toks[pos] != "]":
                a.append(val())
                if toks[pos] == ",": pos += 1
            pos += 1; return a
        if t[0] == '"': return json.loads(t)
        if t[0] in "'`": return t[1:-1]
        if t in ("true", "false", "null"): return {"true": True, "false": False, "null": None}[t]
        if re.match(r"-?\d", t): return float(t) if re.search(r"[.eE]", t) else int(t)
        raise ValueError("不支持的值：" + t)
    return val()


def fetch(path, ref):
    url = "https://raw.githubusercontent.com/%s/%s/%s" % (REPO, ref, path)
    with urllib.request.urlopen(url, timeout=30) as r: return r.read().decode("utf-8")


def head_sha():
    req = urllib.request.Request("https://api.github.com/repos/%s/commits/main" % REPO, headers={"Accept": "application/vnd.github.sha"})
    with urllib.request.urlopen(req, timeout=30) as r: return r.read().decode().strip()


def deploy_of(model, match):
    best = None
    for mk, sub, d in DEPLOY:
        if mk == model and all(match.get(k) == v for k, v in sub.items()) and (best is None or len(sub) > len(best[0])): best = (sub, d)
    if not best: return None
    # 条目多出来的维度（例如 quant / kvDsaPair）必须被 DEPLOY 的条件覆盖，否则不算对上
    extra = set(match) - set(best[0]) - {"quant", "pdMode"}
    return None if extra else best[1]


def rows_of(model, path, ref, src):
    out = []
    for e in parse_js(src.split("=", 1)[1].strip().rstrip(";")):
        if not e.get("speed"): continue
        m, d = e["match"], deploy_of(model, e["match"])
        quant = m.get("quant", DEFAULT_QUANT[model])
        for s in e["speed"]:
            w = s["workload"]; isl, osl, c = w["isl"], w["osl"], w["max_concurrency"]
            r = {"id": "-".join([model, m["hw"], m["strategy"]] + [str(m[k]) for k in ("quant", "kvDsaPair", "spec") if k in m] + ["%d-%d-c%d" % (isl, osl, c)]),
                 "src": "SGLang cookbook", "url": "https://github.com/%s/blob/%s/%s%s-benchmarks.jsx" % (REPO, ref, DIR, path),
                 "model": model, "hw": HW.get(m["hw"], m["hw"].upper()), "strategy": m["strategy"], "quant": quant,
                 "isl": isl, "osl": osl, "conc": c, "num_prompts": w.get("num_prompts"), "ttft_ms": s["ttft_ms"], "tpot_ms": s["tpot_ms"], "tok_s_gpu": s["tokens_per_sec_per_gpu"],
                 "stat": e.get("latencyPercentile", "Mean").lower(), "sglang": e.get("sglang_version"), "calib": False, "confidence": "reported"}
            if d:
                acc = d.get("acc")
                if isinstance(acc, dict): acc = acc.get(isl, {}).get(c)
                N = d["N"]
                r.update(topo=TOPO.get(m["hw"], "hgx8"), N=N, moePar=d["moePar"], kvDt=d["kvDt"], dcp=d.get("dcp", 1), spec=d["spec"],
                         specK=d.get("specK", 0), acc_len=acc, acc_sim=d.get("sim", False), **QUANT[quant])
                burst = w.get("num_prompts") is not None and w["num_prompts"] < 2 * c   # 一波突发，不是闭环稳态
                r["calib"] = bool(d["calib"] and m["hw"] in HW and (d["spec"] == "none" or acc) and not burst)
                if r["acc_sim"]: r["confidence"] = "derived"      # 接受长度是模拟的：只说明吞吐机制
                if c == 1:   # 用 tok/s/GPU 的口径反推卡数
                    n = (isl + osl) / (s["ttft_ms"] / 1e3 + (osl - 1) * s["tpot_ms"] / 1e3) / s["tokens_per_sec_per_gpu"]
                    if abs(n - N) / N > 0.08: raise SystemExit("%s：按 tok/s/GPU 反推是 %.2f 卡，DEPLOY 写的是 %d" % (r["id"], n, N))
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", help="sgl-project/sglang 的 commit；默认 main 的当前 commit")
    a = ap.parse_args()
    ref = a.ref or head_sha()
    rows = []
    for model, path in FILES.items(): rows += rows_of(model, path, ref, fetch(DIR + path + "-benchmarks.jsx", ref))
    ids = [r["id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup: sys.exit("重复 id：" + ", ".join(dup))
    meta = {"ref": ref, "retrieved": date.today().isoformat()}
    OUT.write_text("[\n" + ",\n".join(json.dumps(dict(r, **meta), ensure_ascii=False, separators=(",", ":")) for r in rows) + "\n]\n", encoding="utf-8")
    print("%d 行（%d 行进校准）→ %s；sglang@%s" % (len(rows), sum(r["calib"] for r in rows), OUT.relative_to(ROOT), ref[:10]))


if __name__ == "__main__":
    main()
