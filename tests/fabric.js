// The cluster pages (Fabric): defaults reproduce the main pages exactly, the ideal stays a lower bound,
// more bandwidth never hurts, oversubscription hurts all-to-all more than rings, NVLS only touches the modeled all-reduce,
// and every net key either changes a result somewhere or is marked inert where it changes nothing.
const E = require("../src/engine/node.js")();
const { MODELS, HW, TOPO, Train, Infer, Common, Fabric } = E;
const G0 = Common.G0, IDEAL = Common.IDEAL;
const fails = [], fail = m => { if (fails.length < 40) fails.push(m); };
const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
const PAIRS = [["H200", "hgx8"], ["B300", "hgx8"], ["GB200", "nvl64"], ["GB300", "nvl64"]];
const infOf = (m, N) => Object.assign({}, Infer.BASE, { N, dpA: m.moe.E ? N : 1, moePar: m.moe.E ? "deepep" : "tp", B: 32 });
const trnOf = topo => Object.assign({}, Train.BASE, { N: TOPO[topo].gpus });

// ---- identity: nothing changed → the same numbers as Train.run / Infer.decode / Infer.prefill ----
let nId = 0;
for (const mk of Object.keys(MODELS)) for (const [hk, tk] of PAIRS) for (const g of [G0, IDEAL]) {
  const m = MODELS[mk], hw = HW[hk], topo = TOPO[tk], ct = trnOf(tk);
  for (const N of [1, 8]) {
    const ci = infOf(m, N);
    if (!near(Train.run(m, hw, topo, ct, g).stepT, Fabric.train(m, hw, topo, ct, {}, g).stepT)) fail(`identity train ${mk} ${hk} ${g.lb ? "ideal" : "modeled"}`);
    if (!near(Infer.decode(m, hw, topo, ci, g).stepT, Fabric.decode(m, hw, topo, ci, {}, g).stepT)) fail(`identity decode ${mk} ${hk} N${N}`);
    if (!near(Infer.prefill(m, hw, topo, ci, g).ttft, Fabric.prefill(m, hw, topo, ci, {}, g).ttft)) fail(`identity prefill ${mk} ${hk} N${N}`);
    const s = Fabric.split("train", m, hw, topo, ct, {}, g);
    if (!near(s.t, s.nic + s.nvl + s.rest)) fail(`split does not add up ${mk} ${hk}`);
    nId++;
  }
}

// ---- ideal ≤ modeled under every fabric variation ----
const NETS = [{ scale: 8 }, { scale: 64, over: 8, route: "ecmp" }, { scale: 8, intra: "pcie" }, { scale: 16, nvls: true, sharp: true }, { scale: 4, gdr: false },
  { scale: 32, domain: 64 }, { scale: 16, nic: 200, pcie: 4 }, { scale: 8, nvlFrac: 0.5, ports: 16, over: 15 }];
let nLb = 0;
for (const mk of ["qwen35", "kimik3", "glm53f", "qwen38_27b"]) for (const [hk, tk] of [["H200", "hgx8"], ["GB300", "nvl64"]]) for (const net of NETS) {
  const m = MODELS[mk], hw = HW[hk], topo = TOPO[tk], ct = trnOf(tk), ci = infOf(m, 32);
  const pairs = [["train", ct], ["decode", ci], ["prefill", ci]];
  for (const [k, c] of pairs) {
    const a = Fabric.split(k, m, hw, topo, c, net, G0).t, b = Fabric.split(k, m, hw, topo, c, net, IDEAL).t;
    if (!(b <= a * (1 + 1e-9))) fail(`ideal slower than modeled: ${k} ${mk} ${hk} ${JSON.stringify(net)} ideal ${b} > modeled ${a}`);
    if (!Number.isFinite(a) || !Number.isFinite(b)) fail(`not finite: ${k} ${mk} ${hk} ${JSON.stringify(net)}`);
    nLb++;
  }
}

// ---- monotonic: more NIC, bigger domain, less oversubscription never slow things down ----
{
  const m = MODELS.kimik3, hw = HW.H200, topo = TOPO.hgx8, ct = trnOf("hgx8"), ci = infOf(m, 32);
  const tT = net => Fabric.train(m, hw, topo, ct, net, G0).stepT, tD = net => Fabric.decode(m, hw, topo, ci, net, G0).stepT;
  const mono = (name, f, vals, key, base, up) => { let prev = null;
    for (const v of vals) { const t = f(Object.assign({}, base, { [key]: v })); if (prev != null && (up ? t < prev * (1 - 1e-9) : t > prev * (1 + 1e-9))) fail(`${name}: ${key}=${v} gives ${t}, previous ${prev}`); prev = t; } };
  for (const [name, f] of [["train", tT], ["decode", tD]]) {
    mono(name, f, [200, 400, 800, 1600], "nic", { scale: 16, pcie: 6 }, false);
    mono(name, f, [8, 16, 32, 64, 128], "domain", { scale: 16 }, false);
    mono(name, f, [1, 2, 4, 8, 15], "over", { scale: 64, ports: 16 }, true);
    mono(name, f, [1, 2, 4, 8, 16], "qps", { scale: 64, ports: 16, route: "ecmp" }, false);
  }
}

// ---- oversubscription: all-to-all and point-to-point lose bandwidth, rings do not ----
{
  const hw = HW.H200, topo = Object.assign({}, TOPO.hgx8, { gpus: 1024 });
  const at = over => { const a = Fabric.apply(hw, topo, { over, ports: 16 }, G0); return { ring: Common.link(a.hw, a.topo, 1024, 1), p2p: Common.link(a.hw, a.topo, 2, 512) }; };
  const one = at(1), four = at(4);
  if (!near(one.ring.bw, four.ring.bw)) fail(`ring bandwidth changed with oversubscription: ${one.ring.bw} → ${four.ring.bw}`);
  if (!(four.ring.a2a < one.ring.a2a * 0.6)) fail(`all-to-all barely changed with 4:1: ${one.ring.a2a} → ${four.ring.a2a}`);
  if (!near(four.p2p.p2p, one.p2p.p2p / 4)) fail(`point-to-point across segments at 4:1 should be a quarter: ${one.p2p.p2p} → ${four.p2p.p2p}`);
  // rail striping: 8 members per domain use 8 NICs
  const l = Common.link(hw, TOPO.hgx8.gpus === 8 ? Object.assign({}, TOPO.hgx8, { gpus: 64 }) : TOPO.hgx8, 64, 1);
  if (!near(l.bw, 8 * hw.ib * 1e9)) fail(`ring across 8 HGX nodes should get 8 NICs: ${l.bw}`);
  if (Common.ecmpMax(128, 32) !== Common.ecmpMax(128, 32)) fail("ECMP estimate is not deterministic");
}

// ---- NVLS: changes the modeled TP all-reduce, not the ideal (which already sends one copy) ----
{
  const m = MODELS.qwen38_27b, hw = HW.H200, topo = TOPO.hgx8, ci = Object.assign({}, Infer.BASE, { N: 8, dpA: 1, moePar: "tp", B: 64 });
  const d = (net, g) => Fabric.decode(m, hw, topo, ci, net, g).stepT;
  if (!(d({ nvls: true }, G0) < d({}, G0))) fail("NVLS does not speed up the modeled TP all-reduce");
  if (!near(d({ nvls: true }, IDEAL), d({}, IDEAL))) fail("NVLS changes the ideal decode");
}

// ---- PCIe-only machines: above TP 2 the engines fall back from custom all-reduce to the NCCL ring ----
{
  const m = MODELS.qwen38_27b, hw = HW.H200, topo = TOPO.hgx8;
  const at = (tp, flag) => { const a = Fabric.apply(hw, topo, { intra: "pcie" }, G0), ci = Object.assign({}, Infer.BASE, { N: tp, dpA: 1, moePar: "tp", B: 16 });
    return Infer.decode(m, a.hw, Object.assign({}, a.topo, { pcieOnly: flag }), Object.assign({}, ci, { phase: "decode" }), G0).stepT; };
  if (!near(at(2, true), at(2, false))) fail("PCIe-only TP 2 should keep the custom all-reduce");
  if (near(at(4, true), at(4, false))) fail("PCIe-only TP 4 should use the NCCL ring, not the custom all-reduce");
  if (!Fabric.apply(hw, topo, { intra: "pcie" }, G0).topo.pcieOnly || Fabric.apply(hw, topo, {}, G0).topo.pcieOnly) fail("pcieOnly is set only for intra = pcie");
}

// ---- every net key changes something somewhere; keys marked inert change nothing ----
const ALT = { scale: [8], domain: [16, 64], intra: ["pcie"], nvlFrac: [0.5], nvls: [true], gdr: [false], pcie: [4], nic: [200, 1600], ports: [16, 128], over: [8], route: ["ecmp"], qps: [16], sharp: [true] };
const BASES = [{}, { scale: 16 }, { scale: 64, ports: 16 }, { scale: 64, ports: 16, route: "ecmp" }, { scale: 8, intra: "pcie" }];
const seen = {};
for (const mk of ["kimik3", "qwen38_27b", "qwen35"]) for (const [hk, tk] of [["H200", "hgx8"], ["GB300", "nvl64"]]) for (const g of [G0, IDEAL]) for (const base of BASES) for (const ci of [infOf(MODELS[mk], 8), infOf(MODELS[mk], 64), Object.assign({}, Infer.BASE, { N: 8, dpA: 1, moePar: "tp", B: 32 })]) {
  const m = MODELS[mk], hw = HW[hk], topo = TOPO[tk], ct = Object.assign({}, trnOf(tk), m.moe.E ? {} : { ep: 1 });
  if (ct.fw === "torchtitan" && m.moe.E === 0) ct.ep = 1;
  const sig = net => JSON.stringify([Fabric.train(m, hw, topo, ct, net, g).stepT, Fabric.decode(m, hw, topo, ci, net, g).stepT, Fabric.prefill(m, hw, topo, ci, net, g).ttft,
    Fabric.transfers(m, hw, topo, ci, net, g)]);
  const inert = Fabric.inert(hw, topo, base, g, ci), s0 = sig(base);
  for (const [k, vals] of Object.entries(ALT)) for (const v of vals) {
    if (base[k] === v) continue;
    const same = sig(Object.assign({}, base, { [k]: v })) === s0;
    if (inert[k] && !same) fail(`${k}=${v} is marked inert (${inert[k]}) but changes the result: ${mk} ${hk} ${g.lb ? "ideal" : "modeled"} ${JSON.stringify(base)} N${ci.N}`);
    if (!inert[k] && !same) seen[k] = (seen[k] || 0) + 1;
  }
}
for (const k of Object.keys(ALT)) if (!seen[k]) fail(`net key ${k} never changes any result`);
for (const k of Object.keys(Fabric.DEF)) if (!(k in ALT)) fail(`net key ${k} has no test value`);

console.log(`fabric: ${nId} identity checks, ${nLb} ideal ≤ modeled checks, ${Object.keys(seen).length} net keys reach the result`);
if (fails.length) { console.log("FAIL\n" + fails.join("\n")); process.exit(1); }
console.log("fabric ok");
