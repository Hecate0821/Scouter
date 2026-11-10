// The timeline follows the configuration. Each parallelism control is moved through the page itself and the chart is read back:
// the move has to show up, and what is drawn has to be what the engine returned for that configuration (lanes, cards, bytes, bar lengths).
const path = require("path");
const load = require("./load");
const page = path.resolve(__dirname, "..", "index.html");
const problems = [], say = m => { if (!problems.includes(m)) problems.push(m); };

const bind = (w, d) => (k, v) => { const el = d.querySelector(`[data-bind="${k}"]`); if (!el) return say(`no control for ${k}`);
  if (el.type === "checkbox") el.checked = !!v; else el.value = v;
  el.dispatchEvent(new w.Event(el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input", { bubbles: true })); };
const pick = (w, d) => (id, v) => { const el = d.getElementById(id); el.value = v; el.dispatchEvent(new w.Event("change", { bubbles: true })); };
const read = (d, id) => Array.from(d.getElementById(id).querySelectorAll(".xsec")).map(s => ({
  sub: !!s.querySelector(".xs.sub"),
  lanes: Array.from(s.querySelectorAll(".xtr:not(.sm) .xtl")).map(l => [l.firstChild.textContent, l.querySelector("small").textContent]),
  bars: Array.from(s.querySelectorAll("button.xtb")).map(b => { const m = /left:([\d.]+)%;width:([\d.]+)%/.exec(b.getAttribute("style"));
    return { sig: b.dataset.xk, text: b.textContent, left: +m[1], width: +m[2], over: b.classList.contains("over") }; }) }));
const print = chart => JSON.stringify(chart);
const opts2 = (d, id) => Array.from(d.getElementById(id).options).map(o => o.value);

/* What the engine says for the current training config, without going through trnPlan */
const TRN_TRUTH = `(()=>{ const r=trnRun(), {m,g}=ctx(), fs=r.par.fsdp, {kind}=xKinds(m), L=xSample(m,XV.trn.n);
  const xs=(key,blk)=>{ const e=r.comm.find(x=>x.key===key&&(!blk||x.blk===blk)); return e&&e.xs?e.xs.map(x=>({op:x.op,pass:x.pass,b:fB(x.bytes),t:x.t})):[]; };
  const per=(pick,f)=>sum(r.ops.filter(pick),o=>f(o)/(o.layers||1));
  const cf=r.par.coreFullOp, cs=r.ops.find(o=>o.key==="acore"&&o!==cf), core=a=>a==="swa"?cs:cf;
  const compOf=(k,f)=>(k.a==="lin"?per(o=>o.group==="lin",f):per(o=>o.group==="attn"&&(o.key!=="acore"||o===core(k.a)),f))+(k.f==="moe"?per(o=>o.group==="moe"&&o.key!=="dense",f):per(o=>o.key==="dense",f));
  const layers=L.map(i=>{ const k=kind(i), a=k.a==="lin"?"lin":"attn";
    return { i, comp:{ fwd:compOf(k,o=>o.fwd.t), bwd:compOf(k,o=>o.dgrad.t+o.wgrad.t+(o.rc?o.fwd.t:0)) },
      calls:{ tp:[...xs("tp",a),...(k.f==="moe"?(r.ops.some(o=>o.key==="shared"&&o.fwd.t>0)?xs("tp","moe"):[]):xs("tp","dense")),...(k.a==="full"?xs("tpidx","attn"):[])],
        cp:k.a==="lin"?xs("cplin","lin"):k.a==="swa"?xs("cpswa","swa"):xs("cpkv","full"), ep:k.f==="moe"?[...xs("a2a","moe"),...xs("cnt","moe")]:[], etp:k.f==="moe"?xs("etp","moe"):[] } }; });
  const fsdp={}; if(fs) for(const j of new Set(L.flatMap(i=>[i-1,i,i+1]))) if(j>=0&&j<m.L){ const u=fs.blocks[j], det=(P1,F,lk)=>Par.fsdpDet(P1,F,lk,g,fs.aG,fs.gB).map(x=>({op:x.op,pass:x.pass,b:fB(x.bytes)})); fsdp[j]={fsdp:det(u.d,fs.fs,fs.lkD),efsdp:det(u.e,fs.fe,fs.lkE)}; }
  const want={}; if(r.tp>1)want.tp=r.tp; if(r.cp>1)want.cp=r.cp; if(m.moe.E>0&&r.ep>1)want.ep=r.ep; if(m.moe.E>0&&r.etp>1)want.etp=r.etp; if(r.pp>1)want.pp=r.pp;
  if(fs){ if(fs.fs>1)want.fsdp=fs.fs; if(fs.fe>1&&m.moe.E>0)want.efsdp=fs.fe; }
  for(const x of r.stepX) want[x.dim]=x.n;
  const map={}; for(const [k,v] of xSigned(trnPlan(r,L,XV.trn.show)).map) map[k]=v.t;
  return JSON.stringify({ layers, fsdp, want, map, ranks:[r.tp,r.cp,r.ep,r.etp,r.pp], bad:r.warn.filter(x=>x.bad).map(x=>x.key), step:r.stepX.map(x=>({dim:x.dim,op:x.op,b:fB(x.bytes)})), pp:xs("pp","stage"),
    names:Object.fromEntries(Object.keys(XD).map(k=>[XD[k][0],k])), tOpt:r.tOpt, cpMeth:r.par.cpMeth, show:XV.trn.show }); })()`;

function checkTrain(w, d, where) {
  const T = JSON.parse(w.eval(TRN_TRUTH)), C = read(d, "x-t-grid"), bars = C.flatMap(s => s.bars), fail = m => say(`${where}: ${m}`);
  // Lanes and how many cards each one spans
  const seen = {}; for (const s of C) for (const [name, small] of s.lanes) seen[T.names[name]] = parseInt(small, 10);
  for (const k of new Set([...Object.keys(T.want), ...Object.keys(seen)])) if (T.want[k] !== seen[k]) fail(`lane ${k}: engine says ${T.want[k]} cards, chart shows ${seen[k]}`);
  // Per-layer calls, in bytes and wire time
  const bucket = (dim, op, pass, lay, b, t) => `${dim}|${op}|${pass}|${lay}=${b}@${t === undefined ? "no time" : t.toPrecision(6)}`;
  const drawn = bars.filter(b => b.sig[0] === "x").map(b => { const [, dim, op, pass, lay] = b.sig.split("#")[0].split("|"); return { b, dim, op, pass, lay, bytes: b.text.slice(op.length + 1), t: T.map[b.sig] }; });
  const wantCalls = [], gotCalls = drawn.filter(x => ["tp", "cp", "ep", "etp"].includes(x.dim) && /^\d+$/.test(x.lay)).map(x => bucket(x.dim, x.op, x.pass, x.lay, x.bytes, x.t)).sort();
  for (const l of T.layers) for (const dim in l.calls) for (const x of l.calls[dim]) if (T.show === "both" || x.pass === T.show || (x.pass === "rc" && T.show === "bwd")) wantCalls.push(bucket(dim, x.op, x.pass, l.i, x.b, x.t));
  wantCalls.sort();
  const miss = wantCalls.filter(x => !gotCalls.includes(x)), extra = gotCalls.filter(x => !wantCalls.includes(x));
  if (miss.length) fail(`engine calls missing from the chart: ${miss.slice(0, 4).join(", ")}`);
  if (extra.length) fail(`chart calls the engine does not have: ${extra.slice(0, 4).join(", ")}`);
  // FSDP prefetch and reshard calls
  for (const x of drawn.filter(y => y.dim === "fsdp" || y.dim === "efsdp")) { const E = T.fsdp[x.lay] && T.fsdp[x.lay][x.dim];
    if (!E || !E.some(e => e.op === x.op && e.pass === x.pass && e.b === x.bytes)) fail(`${x.dim} bar ${x.b.sig} shows ${x.bytes}, not in the engine's layer ${x.lay} calls`); }
  // Once per micro-batch or step
  for (const x of T.step) if (!drawn.some(y => y.dim === x.dim && y.op === x.op && y.bytes === x.b)) fail(`step call ${x.dim} ${x.op} ${x.b} is not drawn`);
  for (const x of T.pp) if (!drawn.some(y => y.dim === "pp" && y.op === x.op && y.bytes === x.b)) fail(`PP call ${x.op} ${x.b} is not drawn`);
  // Length is wire time: one scale per section
  for (const s of C) { const r = s.bars.filter(b => b.sig[0] === "x" && b.width > 0.2 && T.map[b.sig] > 0).map(b => b.width / T.map[b.sig]);
    if (r.length > 1 && Math.max(...r) / Math.min(...r) > 1.03) fail("comm bars in one section are not on one time scale"); }
  // A call sits in the section of its own layer and pass
  for (const s of C.filter(x => !x.sub)) { const own = s.bars.filter(b => b.sig[0] === "x" && !/\|(fsdp|efsdp)\|/.test(b.sig)).map(b => b.sig.split("#")[0].split("|")), fwd = own.some(x => x[3] === "fwd"), back = own.some(x => x[3] !== "fwd");
    if (fwd && back) fail(`one section mixes forward and backward calls: ${own.map(x => x.slice(1).join("|")).slice(0, 4).join(", ")}`);
    if (new Set(own.map(x => x[4])).size > 1) fail(`one section mixes layers: ${[...new Set(own.map(x => x[4]))]}`); }
  // Compute: the drawn lengths add up to the engine's per-layer time
  const scale = s => { const b = s.bars.filter(x => x.sig[0] === "x" && T.map[x.sig] > 0).sort((p, q) => q.width - p.width)[0]; return b && T.map[b.sig] / (b.width / 100); };
  const pass = s => { const x = s.bars.find(b => b.sig[0] === "x" && /^\d+$/.test(b.sig.split("#")[0].split("|")[4]) && !/\|(fsdp|efsdp)\|/.test(b.sig)); return x && x.sig.split("|")[3]; };
  const layerOf = s => { const x = s.bars.find(b => b.sig[0] === "x" && !/\|(fsdp|efsdp)\|/.test(b.sig)); return x && +x.sig.split("#")[0].split("|")[4]; };
  for (const s of C.filter(x => !x.sub)) { const S = scale(s), lay = layerOf(s), p = pass(s), L = T.layers.find(l => l.i === lay); if (!S || !L || !p) continue;
    const got = s.bars.filter(b => b.sig[0] === "c").reduce((a, b) => a + b.width / 100 * S, 0), want = p === "fwd" ? L.comp.fwd : L.comp.bwd;
    if (Math.abs(got - want) / want > 0.05) fail(`layer ${lay} ${p}: compute drawn ${got.toExponential(3)}s, engine ${want.toExponential(3)}s`); }
  return T;
}

/* Inference, the same way: the engine's records for whichever config the chart says it is showing */
const INF_TRUTH = `(()=>{ const pick=infCommCfg(), r=infRun(pick.over), {m}=ctx(), {kind}=xKinds(m), L=xSample(m,XV.inf.n), c=Object.assign({},S.inf,pick.over);
  const a2a=(c.dcpComm||r.P.dcpComm)==="a2a", OP={ar:"AR",latar:"AR",dpg:"AG",dps:"RS",disp:"A2A",comb:"A2A",cnt:"sync",lmg:"AG",epag:"AG",dcpq:"AG",dcplse:"AG",dcpo:a2a?"A2A":"RS",dcpkv:"AG"}, ag=r.moePar==="ag",
    dim={ar:"tp",latar:"tp",lmg:"tp",epag:"tp",disp:"ep",comb:"ep",cnt:"ep",dpg:ag?"ep":"dpa",dps:ag?"ep":"dpa",dcpq:"cp",dcplse:"cp",dcpo:"cp",dcpkv:"cp"};
  const cards={tp:Math.round(r.tpA),dpa:r.dpA,ep:r.N,cp:Math.max(1,Math.round(c.dcp||1))}, calls=r.comm.filter(e=>dim[e.key]).map(e=>({key:e.key,dim:dim[e.key],op:OP[e.key],b:fB(e.bytes/e.layers),t:e.t/e.layers,ov:e.overlap||""}));
  const per=f=>sum(r.ops.filter(f),o=>o.t/o.layers), cores=r.ops.filter(o=>o.key==="acore");
  const moeK=["router","latd","moeq","moe","latu","shared"], core=k=>k.a==="swa"?cores.find(o=>o.mod==="swa"):cores.find(o=>o.mod==="full")||cores.find(o=>!o.mod), compOf=k=>(k.a==="lin"?per(o=>o.group==="lin"):per(o=>o.group==="attn"&&o.key!=="acore")+(core(k)?core(k).t/core(k).layers:0))+(k.f==="moe"?per(o=>moeK.includes(o.key)):per(o=>o.key==="dense"));
  const map={}; for(const [k,v] of xSigned(infPlan(r,L)).map) map[k]=v.t;
  return JSON.stringify({ calls, cards, map, comp:Object.fromEntries(L.map(i=>[i,compOf(kind(i))])), auto:pick.auto, preset:pick.cf?pick.cf.id:"", engine:c.engine, N:c.N, B:c.B, names:Object.fromEntries(Object.keys(XD).map(k=>[XD[k][0],k])),
    work:r.ops.filter(o=>o.group!=="misc").map(o=>[o.key,(o.t/o.layers).toPrecision(3)]) }); })()`;

function checkInfer(w, d, where) {
  const T = JSON.parse(w.eval(INF_TRUTH)), C = read(d, "x-i-grid"), bars = C.flatMap(s => s.bars), fail = m => say(`${where}: ${m}`);
  const seen = {}; for (const s of C) for (const [name, small] of s.lanes) seen[T.names[name]] = parseInt(small, 10);
  const want = {}; for (const x of T.calls) want[x.dim] = T.cards[x.dim];
  for (const k of new Set([...Object.keys(want), ...Object.keys(seen)])) if (want[k] !== seen[k]) fail(`lane ${k}: engine says ${want[k]} cards, chart shows ${seen[k]}`);
  const drawn = bars.filter(b => b.sig[0] === "x").map(b => { const [, dim, op] = b.sig.split("|"); return { dim, op, b: b.text.slice(op.length + 1), t: T.map[b.sig], over: b.over, width: b.width, sig: b.sig }; });
  const hit = x => drawn.some(y => y.dim === x.dim && y.op === x.op && y.b === x.b && Math.abs(y.t - x.t) <= 1e-9 * x.t);
  for (const x of T.calls) if (!hit(x)) fail(`engine call ${x.key} (${x.op} ${x.b}) is not drawn`);
  for (const y of drawn) if (!T.calls.some(x => x.dim === y.dim && x.op === y.op && x.b === y.b)) fail(`chart call ${y.sig} (${y.b}) is not in the engine`);
  for (const x of T.calls) if (x.ov) for (const y of drawn.filter(z => z.dim === x.dim && z.op === x.op && z.b === x.b)) if (!y.over) fail(`${x.key} overlaps in the engine but ${y.sig} draws as blocking`);
  for (const s of C) { const r = s.bars.filter(b => b.sig[0] === "x" && b.width > 0.2 && T.map[b.sig] > 0).map(b => b.width / T.map[b.sig]);
    if (r.length > 1 && Math.max(...r) / Math.min(...r) > 1.03) fail("comm bars in one section are not on one time scale"); }
  for (const s of C.filter(x => !x.sub)) { const b = s.bars.filter(x => x.sig[0] === "x" && T.map[x.sig] > 0).sort((p, q) => q.width - p.width)[0]; if (!b) continue;
    const S = T.map[b.sig] / (b.width / 100), lay = b.sig.split("#")[0].split("|")[4], got = s.bars.filter(x => x.sig[0] === "c").reduce((a, x) => a + x.width / 100 * S, 0), wantC = T.comp[lay];
    if (wantC && Math.abs(got - wantC) / wantC > 0.05) fail(`layer ${lay}: compute drawn ${got.toExponential(3)}s, engine ${wantC.toExponential(3)}s`); }
  return T;
}

/* Every control on the rack: if the engine's answer for what the chart draws changes, the chart has to change */
function everyControl(w, d, prefix, id, truth, where) {
  const visible = el => { const f = el.closest("[data-fw]"); return !f || f.dataset.fw === w.eval("S.trn.fw"); };
  let tried = 0;
  for (const el of Array.from(d.querySelectorAll(`[data-bind^="${prefix}"]`)).filter(visible)) { const k = el.dataset.bind, set = bind(w, d), f = k.split(".")[1], base = w.eval(`S.${prefix}${f}`);
    if (SKIP_CONTROLS.includes(k)) continue;
    const alts = el.tagName === "SELECT" ? Array.from(el.options).map(o => o.value).filter(v => v !== String(base)) : el.type === "checkbox" ? [!base] : [+base * 2 || 1];
    for (const v of alts) { const cA = print(read(d, id)), eA = truth(); set(k, v); const cB = print(read(d, id)), eB = truth(); tried++;
      if (eA !== eB && cA === cB) say(`${where}: ${k}=${v} changes the engine's numbers but not the timeline`);
      set(k, base); if (print(read(d, id)) !== cA) say(`${where}: ${k} back to ${base} should restore the timeline`); } }
  if (tried < 8) say(`${where}: only ${tried} controls swept`);
}
const SKIP_CONTROLS = ["trn.fw", "inf.engine", "trn.prec"];   // these reset the whole rack, swept on their own

(async () => {
  const { w, d, errs } = await load(page, "#train/par", null, "?lang=zh");
  const set = bind(w, d), sel = pick(w, d), chart = () => print(read(d, "x-t-grid")), others = () => ["x-t-vol", "x-t-lede", "x-t-tbl"].map(id => d.getElementById(id).innerHTML).join();
  // These change when a call runs, not how much it sends. At a fixed token count S changes no bytes either, only the size of each
  // message; that moves the wire-time column through per-message latency, which the ideal basis (real models) does not have.
  const timingOnly = () => ["trn.epOverlap", "trn.tpOverlap", ...(w.eval("S.glob.lb") ? ["trn.S"] : [])];
  const moves = (k, v, base, where) => { const a = chart(), o = others(); set(k, v); const b = chart(), T = checkTrain(w, d, `${where} ${k}=${v}`);
    if (a === b) say(`${where}: ${k}=${v} does not move the timeline`);
    if (o === others() && !timingOnly().includes(k)) say(`${where}: ${k}=${v} does not move the byte bars, lede or table`);
    set(k, base); return T; };

  // One knob at a time, from the default and from Megatron's default
  const SWEEP = { torchtitan: { tp: [1, 2, 8], cp: [1, 2, 8], ep: [4, 8, 32, 64], pp: [2, 4], dispatcher: ["ag", "deepep", "distmoe", "mok"], cpType: ["allgather", "ring"], linCP: ["state"], recompute: ["full", "none"], tok: [16384], S: [16384] },
    megatron: { tp: [1, 4], cp: [1, 4], ep: [1, 2, 4, 16], etp: [2], pp: [2, 4], cpType: ["allgather"], linCP: ["headwise"], recompute: ["full", "none"], epOverlap: [true], tpOverlap: [true], overlapGrad: [true], overlapParam: [true], tok: [16384], S: [16384] } };
  // on the real model (ideal basis, HuggingArch beside it), then on the same architecture edited into a custom model (modeled basis)
  for (const edited of [false, true]) { const tag = edited ? "qwen35 edited" : "qwen35";
    if (edited) w.eval('S.model.short += " *"; syncBasis(); render(); applySource();');
    if (!!w.eval("S.glob.lb") === edited) say(`${tag}: the training page should be on the ${edited ? "modeled" : "ideal"} basis`);
    for (const fw of ["torchtitan", "megatron"]) { set("trn.fw", fw); checkTrain(w, d, `${tag}/${fw} default`);
      for (const k in SWEEP[fw]) { const base = w.eval(`S.trn.${k}`); for (const v of SWEEP[fw][k]) if (v !== base) moves(`trn.${k}`, v, base, `${tag}/${fw}`); } } }
  sel("g-model", "qwen35"); set("trn.fw", "torchtitan");
  if (!w.eval("S.glob.lb")) say("picking qwen35 again should bring back the real model and its ideal basis");

  // Overlap follows the method
  const cpBars = () => read(d, "x-t-grid").flatMap(s => s.bars).filter(b => /^x\|cp\|/.test(b.sig) && !b.sig.includes("|rc|"));
  set("trn.linCP", "state");
  for (const [m, op] of [["ulysses", "A2A"], ["allgather", "AG"], ["ring", "P2P"]]) { set("trn.cpType", m);
    const full = cpBars().filter(b => b.sig.split("|")[4].startsWith("3"));
    if (!full.some(b => b.sig.split("|")[2] === op)) say(`cpType=${m}: full attention should send ${op}, chart has ${[...new Set(full.map(b => b.sig.split("|")[2]))]}`);
    if (!full.some(b => b.over)) say(`cpType=${m}: full-attention CP should draw as overlap`); }
  set("trn.cpType", "ulysses"); set("trn.linCP", "ulysses");

  // CP lane comes and goes, compute shrinks as TP grows, wire time scales with the fabric
  const lane = name => read(d, "x-t-grid").some(s => s.lanes.some(l => l[0] === name));
  set("trn.cp", 1); if (lane("CP")) say("CP=1 should remove the CP lane"); set("trn.cp", 2); if (!lane("CP")) say("CP=2 should bring the CP lane back"); set("trn.cp", 4);
  set("trn.tp", 1); if (lane("TP")) say("TP=1 should remove the TP lane"); set("trn.tp", 4);
  set("trn.ep", 64); if (lane("expert FSDP")) say("EP=N leaves no expert FSDP to draw"); set("trn.ep", 16);

  // Open panels follow the config, and a bar that goes away takes its panel with it
  const panels = () => d.querySelectorAll("#x-t-grid .xp");
  const cpBar = () => Array.from(d.querySelectorAll('#x-t-grid button.xtb[data-xk^="x|cp|A2A|fwd"]'))[0];
  cpBar().click();
  const ladder = () => { const fig = Array.from(panels()[0].querySelectorAll(".xm")).find(f => f.querySelector("figcaption b").textContent.includes("CP")); const x = fig && fig.querySelector("text.xx.cur"), v = fig && fig.querySelector("text.xv.cur"); return x && v && [x.textContent, v.textContent]; };
  for (const cp of [2, 8, 4]) { set("trn.cp", cp); const l = ladder(), b = cpBar();
    if (!l || l[0] !== String(cp)) say(`open panel: CP ladder should mark ${cp}, marks ${l && l[0]}`);
    else if (!b.textContent.endsWith(l[1])) say(`open panel at CP=${cp}: bar says ${b.textContent}, panel says ${l[1]}`); }
  set("trn.cp", 1); if (panels().length) say("CP=1 should close the CP panel, its bar is gone");
  set("trn.cp", 4); if (panels().length !== 1) say("back to CP=4: the panel should reopen"); cpBar().click();

  // The layer and pass selectors
  for (const id of ["x-t-n", "x-t-show"]) for (const b of Array.from(d.getElementById(id).querySelectorAll("button")).reverse()) { const v = b.dataset.v, before = chart(), on = b.getAttribute("aria-pressed") === "true";
    d.querySelector(`#${id} button[data-v="${v}"]`).click(); if (!on && chart() === before) say(`${id}=${v} does not move the timeline`); checkTrain(w, d, `${id}=${v}`); }
  d.querySelector('#x-t-n button[data-v="2"]').click(); d.querySelector('#x-t-show button[data-v="both"]').click();

  // Every model, framework, topology and preset
  const opts = id => Array.from(d.getElementById(id).options).map(o => o.value);
  let seen = 0;
  for (const model of opts("g-model")) { sel("g-model", model);
    for (const fw of ["torchtitan", "megatron"]) { set("trn.fw", fw);
      for (const preset of opts("t-preset")) { sel("t-preset", preset); seen++; checkTrain(w, d, `${model}/${fw}/${preset}`); } } }
  if (seen < 10) say(`only ${seen} presets swept`);
  sel("g-model", "qwen35"); set("trn.fw", "torchtitan");
  for (const hw of opts("g-hw")) { sel("g-hw", hw); for (const topo of opts("g-topo")) { sel("g-topo", topo); checkTrain(w, d, `${hw}/${topo}`); } }
  // Every training control, on both frameworks
  sel("g-hw", "GB300"); sel("g-topo", "nvl64");
  const trnSig = () => { const T = JSON.parse(w.eval(TRN_TRUTH)); return JSON.stringify([T.layers.map(l => [l.i, l.comp.fwd.toPrecision(3), l.comp.bwd.toPrecision(3), l.calls]), T.fsdp, T.want, T.step, T.pp, T.tOpt.toPrecision(3)]); };
  for (const fw of ["torchtitan", "megatron"]) { set("trn.fw", fw); everyControl(w, d, "trn.", "x-t-grid", trnSig, fw); }
  if (errs.length) say(`training page errors: ${errs.slice(0, 3).join(" | ")}`);

  // ---- inference ----
  const inf = await load(page, "#infer/deploy", null, "?lang=zh"), iw = inf.w, id = inf.d, iset = bind(iw, id), isel = pick(iw, id);
  const ichart = () => print(read(id, "x-i-grid")), itruth = () => JSON.parse(iw.eval(INF_TRUTH)), iSig = () => { const T = itruth(); return JSON.stringify([T.calls, T.work, T.preset]); };
  const cfg = () => id.getElementById("x-i-cfg"), lede = () => id.getElementById("x-i-lede").textContent;
  const imove = (k, v, where) => { const a = ichart(); iset(k, v); if (a === ichart()) say(`${where}: ${k}=${v} does not move the timeline`); return checkInfer(iw, id, `${where} ${k}=${v}`); };
  // One GPU per instance: a preset lends only its layout, and the rest still follows the rack
  let T = checkInfer(iw, id, "inference default");
  if (!T.auto) say("inference default should borrow a preset's layout, N=1 has no comm");
  if (!lede().includes(iw.eval('t("comm.inf.srcAuto")'))) say("the borrowed layout has to be stated");
  T = imove("inf.B", 32, "borrowed"); if (T.B !== 32) say(`borrowed layout: rack B=32 should reach the chart, got B=${T.B}`);
  imove("inf.n", 4096, "borrowed"); imove("inf.phase", "prefill", "borrowed"); imove("inf.C", 4096, "borrowed"); imove("inf.phase", "decode", "borrowed");
  iset("inf.engine", "vllm"); T = checkInfer(iw, id, "vLLM on one GPU"); if (T.auto && T.engine !== "vllm") say(`rack engine is vLLM, borrowed layout is from ${T.engine}`); iset("inf.engine", "sglang");
  // The rack's own layout
  for (const N of [2, 8, 16, 64]) { iset("inf.dpA", 1); T = imove("inf.N", N, "rack"); if (T.auto || T.N !== N || cfg().value !== "cur") say(`N=${N}: the chart should be on the rack, it is on ${T.preset || cfg().value}`); }
  for (const dp of [2, 4, 8]) imove("inf.dpA", dp, "rack N=64");
  for (const mp of ["deepep", "fused", "tp"]) imove("inf.moePar", mp, "rack N=64");
  iset("inf.moePar", "deepep"); for (const B of [16, 128]) imove("inf.B", B, "rack N=64"); for (const n of [1024, 32768]) imove("inf.n", n, "rack N=64");
  imove("inf.tbo", true, "rack DeepEP"); imove("inf.tbo", false, "rack DeepEP"); imove("inf.dispFp8", false, "rack DeepEP"); iset("inf.dispFp8", true);
  iset("inf.phase", "prefill"); checkInfer(iw, id, "rack prefill"); for (const C of [2048, 16384]) imove("inf.C", C, "rack prefill"); iset("inf.phase", "decode");
  // A preset from the dropdown holds until a control it overrides moves
  const presets = () => Array.from(cfg().options).map(o => o.value).filter(v => v !== "cur");
  if (!presets().length) say("no multi-GPU preset to pick");
  isel("x-i-cfg", presets()[0]); T = checkInfer(iw, id, "picked preset"); if (T.preset !== presets()[0]) say("the picked preset should be on the chart");
  const held = ichart(); iset("inf.n", 16384); if (itruth().preset !== presets()[0]) say("n is not the preset's, moving it should keep the preset"); if (held === ichart()) say("n should still move the picked preset's chart");
  iset("inf.N", 32); T = checkInfer(iw, id, "after moving N"); if (T.preset || T.N !== 32 || cfg().value !== "cur") say(`moving N should bring the chart back to the rack, it is on ${T.preset || cfg().value} N=${T.N}`);
  // The panel's ladder follows the rack
  const iBar = () => id.querySelector('#x-i-grid button.xtb[data-xk^="x|tp|AR"]'); iBar().click();
  for (const B of [32, 64]) { iset("inf.B", B); const fig = id.querySelector("#x-i-grid .xp .xm"), cur = fig && fig.querySelector("text.xx.cur");
    if (!cur || cur.textContent !== String(B)) say(`inference panel should mark B=${B}, marks ${cur && cur.textContent}`); }
  iBar().click();
  iset("inf.N", 64); iset("inf.dpA", 8); iset("inf.moePar", "deepep");
  everyControl(iw, id, "inf.", "x-i-grid", iSig, "inference");
  // Every model and topology, with every preset the dropdown offers
  let iseen = 0;
  for (const model of opts2(id, "g-model")) { isel("g-model", model);
    for (const topo of opts2(id, "g-topo")) { isel("g-topo", topo); checkInfer(iw, id, `${model}/${topo}`);
      for (const p of presets()) { isel("x-i-cfg", p); iseen++; checkInfer(iw, id, `${model}/${topo}/${p}`); } isel("x-i-cfg", "cur"); } }
  if (iseen < 10) say(`only ${iseen} inference presets swept`);
  if (errs.length) say(`page errors: ${errs.slice(0, 3).join(" | ")}`);
  if (inf.errs.length) say(`inference page errors: ${inf.errs.slice(0, 3).join(" | ")}`);

  if (problems.length) { console.error(problems.slice(0, 40).join("\n") + (problems.length > 40 ? `\n… ${problems.length - 40} more` : "")); process.exit(1); }
  console.log(`comm dynamic ok (${seen} training presets, ${iseen} inference presets)`);
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
