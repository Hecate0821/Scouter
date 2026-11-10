/* ================= HuggingArch：真实模型的仿真（推理 · 一步、训练 · 一步的第一块） =================
   真实模型（Hga.REAL，模型没改过）的中间那个口径不再是 Scouter 自己的建模，而是 HuggingArch 的仿真（shell.js 的 hgaOn）。
   对照的数字全部来自数据平面 Scouter.hga / hgaTrain（src/engine/hga.js 算，界面不另算）：这里只放控件和画图——
   签名（每个引擎 5 个，所有模型共用）、滑块（在导出网格上按对数插值，所以能实时拖）、瀑布、按类别的时间、资源表、曲线和读图。
   导出的数据按模型懒加载（src/data/hga/<model>.js），加载完再画一次 */
S.hgaI = { engine:"sglang", sig:"tp8", isl:8192, b:64, phase:"decode", axis:"b", metric:"step" };
S.hgaT = { engine:"megatron", sig:"tp8sp", seq:8192, mbs:1, metric:"step" };
const HGA_ENG = { sglang:"SGLang", vllm:"vLLM", megatron:"Megatron-Core", torchtitan:"torchtitan" };
const HGA_ROLEC = { attn:"c-attn", moe:"c-moe", dense:"c-lin", router:"c-cp", other:"c-lm", comm:"c-a2a" };
/* 滑块是对数刻度：每档 2^(1/4)，两端是导出网格的两端 */
const LOGS = {
  isl:{ lo:1024, n:28, round:64 }, b:{ lo:1, n:36, round:1 }, seq:{ lo:4096, n:12, round:256 } };
const logV = (k, i) => { const L=LOGS[k]; return Math.max(L.lo, Math.round(L.lo*Math.pow(2, i/4)/L.round)*L.round); };
const logI = (k, v) => Math.max(0, Math.min(LOGS[k].n, Math.round(4*Math.log2(v/LOGS[k].lo))));

/* ---- 懒加载 ---- */
const hgaLoad = {};
function hgaReady(mk){
  if(HGA_DATA.has(mk)) return "ok";
  if(!hgaLoad[mk]){
    hgaLoad[mk] = "loading";
    const sc = document.createElement("script"); sc.src = "src/data/hga/" + mk + ".js";
    const done = ok => { hgaLoad[mk] = ok && HGA_DATA.has(mk) ? "ok" : "error"; if(hgaOn()) render(); };
    sc.onload = () => done(true); sc.onerror = () => done(false);
    document.head.appendChild(sc);
  }
  return hgaLoad[mk];
}

/* ---- 共用的控件和外框 ---- */
function hgaSeg(el, items, cur, onPick){
  el.innerHTML = items.map(([k, label]) => '<button type="button" data-k="' + esc(k) + '" aria-pressed="' + (k === cur) + '">' + esc(label) + '</button>').join("");
  el.onclick = e => { const b = e.target.closest("button[data-k]"); if(b) onPick(b.dataset.k); };
}
function hgaSigSel(el, engine, cur, onPick){
  const sigs = Hga.sigsFor(engine, S.modelKey);
  el.innerHTML = sigs.map(s => '<option value="' + s.id + '">' + esc(t("hga.sig." + engine + "." + s.id)) + '</option>').join("");
  el.value = cur; el.onchange = e => onPick(e.target.value);
}
/* 这一块能不能画：卡在 HuggingArch 的卡表里、数据加载了、格子算过；不能画时把原因写进 lede，返回 false */
function hgaGate(pre, body){
  const mk = S.modelKey, lede = $(pre + "-lede");
  if(!(S.hw in Hga.CARD)){ lede.innerHTML = t("hga.noCard", { hw:esc(HW[S.hw].name), cards:Object.keys(Hga.CARD).map(k => HW[k].name).join(" / ") }); body.hidden = true; return false; }
  const st = hgaReady(mk);
  if(st !== "ok"){ lede.innerHTML = t(st === "error" ? "hga.loadError" : "hga.loading", { model:esc(S.model.short) }); body.hidden = true; return false; }
  return true;
}
function hgaFail(pre, body, r){
  body.hidden = true;
  $(pre + "-lede").innerHTML = r.reason === "missing" ? t("hga.missing")
    : r.reason === "capacity" ? t("hga.capacity")
    : t("hga.refused") + '<br><code class="hga-why">' + esc(r.reason) + '</code>';
}
const hgaRatio = (a, b) => a > 0 && isFinite(b) ? fI(b/a) + "×" : "—";
const hgaSrc = r => t("hga.src", { commit:r.source ? r.source.hga : "?", date:r.source ? r.source.generated : "?" }) + " " + (r.exact ? t("hga.exact") : t("hga.interp"));
const hgaWarnings = list => list && list.length ? '<p class="hint">' + esc(t("hga.says")) + '</p>' + list.map(w => '<code class="hga-why">' + esc(typeof w === "string" ? w : JSON.stringify(w)) + '</code>').join("") : "";
const IDEAL_SER = { name:"", col:"s1" }, HGA_SER = { name:"HuggingArch", col:"s2" };
/* 总差距那一句：HuggingArch 比理想还低时，理想只是 Scouter 自己逐算子记账的下限，指出 HuggingArch 记得更少的那一类（roles 是 [{role, ideal, hga}]） */
function hgaGapLine(ideal, hga, roles, more){
  const under = hga < ideal ? roles.filter(x => x.ideal > x.hga).sort((a, b) => (b.ideal - b.hga) - (a.ideal - a.hga))[0] : null;
  return under ? t("hga.read.under", { x:hgaRatio(ideal, hga), hga:V(fT(hga)), ideal:V(fT(ideal)), role:t("hga.role." + under.role), a:V(fT(under.ideal)), b:V(fT(under.hga)) })
    : t("hga.read.gap", { x:hgaRatio(ideal, hga), hga:V(fT(hga)), ideal:V(fT(ideal)) }) + (more ? " " + more : "");
}

/* ================= 推理 ================= */
function hgaInferQuery(){ const h=S.hgaI; return { model:S.modelKey, engine:h.engine, sig:h.sig, hw:S.hw, isl:h.isl, b:h.b }; }
function renderHgaInfer(){
  const h = S.hgaI, body = $("ih-body");
  if(!Hga.sigOf(h.engine, h.sig) || !Hga.sigsFor(h.engine, S.modelKey).some(s => s.id === h.sig)) h.sig = Hga.sigsFor(h.engine, S.modelKey)[0].id;
  hgaSeg($("ih-eng"), Hga.INFER.map(e => [e, HGA_ENG[e]]), h.engine, k => { h.engine = k; renderHgaInfer(); });
  hgaSigSel($("ih-sig"), h.engine, h.sig, v => { h.sig = v; renderHgaInfer(); });
  const isl = $("ih-isl"), bb = $("ih-b");
  isl.max = LOGS.isl.n; isl.value = logI("isl", h.isl); bb.max = LOGS.b.n; bb.value = logI("b", h.b);
  $("ih-isl-o").textContent = fInt(h.isl); $("ih-b-o").textContent = fInt(h.b);
  isl.oninput = e => { h.isl = logV("isl", +e.target.value); renderHgaInfer(); };
  bb.oninput = e => { h.b = logV("b", +e.target.value); renderHgaInfer(); };
  const sig = Hga.sigOf(h.engine, h.sig, S.modelKey);
  $("ih-flags").innerHTML = '<code>' + esc((h.engine === "sglang" ? "python -m sglang.launch_server " : "vllm serve ") + sig.flags) + '</code>';
  if(!hgaGate("ih", body)) return;
  const r = Scouter.hga(hgaInferQuery());
  if(!r.ok){ hgaFail("ih", body, r); return; }
  body.hidden = false;
  IDEAL_SER.name = t("hga.ideal");
  const dp = sig.hga.dp_attention || 1;
  $("ih-lede").innerHTML = t("hga.inf.lede", { model:esc(S.model.short), hw:esc(HW[S.hw].name), eng:HGA_ENG[h.engine], gpus:sig.gpus, dp, isl:fInt(h.isl), osl:fInt(r.osl), b:fInt(h.b), ctx:fInt(h.isl + r.osl/2) });
  const d = r.decode, p = r.prefill;
  kpis($("ih-kpis"), [
    [t("hga.kpi.dec"), fT(d.hga.t), t("hga.kpi.vsIdeal", { v:fT(d.ideal.t), x:hgaRatio(d.ideal.t, d.hga.t) })],
    [t("hga.kpi.ttft"), fT(p.hga.t), t("hga.kpi.vsIdeal", { v:fT(p.ideal.t), x:hgaRatio(p.ideal.t, p.hga.t) })],
    [t("hga.kpi.thr"), fN(r.throughput.hga) + " tok/s/GPU", t("hga.kpi.thrFoot", { v:fN(r.throughput.ideal) })],
    [t("hga.kpi.maxB"), fInt(r.maxB.hga), t("hga.kpi.maxBFoot", { v:fInt(r.maxB.ideal) }), !r.fits.hga] ]);
  hgaSeg($("ih-phase"), [["decode", t("common.phase.decode")], ["prefill", t("common.phase.prefill")]], h.phase, k => { h.phase = k; renderHgaInfer(); });
  const ph = h.phase === "prefill" ? p : d;
  waterfall("ih-wf", ph.waterfall.map(x => Object.assign({}, x, { label:t("hga.wf." + x.key), note:t("hga.wfTip." + x.key), col:x.key === "hga" ? "s2" : "s1" })), { fmt:fT, up:"c-a2a", down:"c-ar", accLabel:t("hga.wfAcc") });
  pairBars("ih-roles", ph.roles.map(x => ({ label:t("hga.role." + x.role), v:[x.ideal, x.hga] })), [IDEAL_SER, HGA_SER], { fmt:fT, ratio:t("hga.ratio") });
  legendInto("ih-roles-leg", [{ name:IDEAL_SER.name, col:"s1" }, { name:HGA_SER.name, col:"s2" }]);
  const R = ph.resources, row = (label, k, f, hgaOnly) => [label, hgaOnly ? "—" : f(R[k][0]), f(R[k][1]), hgaOnly ? "" : hgaRatio(R[k][0], R[k][1])];
  table($("ih-res"), ["", t("hga.ideal"), "HuggingArch", t("hga.ratio")], [
    row(t("hga.res.compute"), "compute", fT), row(t("hga.res.memory"), "memory", fT), row(t("hga.res.comm"), "comm", fT),
    [t("hga.res.launch"), "—", fT(ph.hga.launch), ""],
    row(t("hga.res.flops"), "flops", fF), row(t("hga.res.bytes"), "bytes", fB),
    [t("hga.res.wBytes"), "—", fB(ph.hga.weightBytes), ""], [t("hga.res.kvBytes"), "—", fB(ph.hga.kvBytes), ""],
    row(t("hga.res.commBytes"), "commBytes", fB), row(t("hga.res.kernels"), "kernels", fInt),
    { cls:"hl", cells:["<b>" + esc(t("hga.res.step")) + "</b>", fT(ph.ideal.t), fT(ph.hga.t), hgaRatio(ph.ideal.t, ph.hga.t)] }], [1, 2, 3]);
  /* 读图：差距最大的一段、差距最大的类别和它在 HuggingArch 里卡在哪、launch 和容量 */
  const parts = ph.waterfall.filter(x => x.d != null), gap = ph.hga.t - ph.ideal.t, big = parts.slice().sort((a, b) => Math.abs(b.d) - Math.abs(a.d))[0];
  const rg = ph.roles.slice().sort((a, b) => Math.abs(b.hga - b.ideal) - Math.abs(a.hga - a.ideal))[0], rb = rg && ph.hga.byRoleBound[rg.role];
  const bound = rb ? ["compute", "memory", "comm"].sort((a, b) => rb[b] - rb[a])[0] : null;
  $("ih-read").innerHTML = ul([
    hgaGapLine(ph.ideal.t, ph.hga.t, ph.roles, big && gap > 0 ? t("hga.read.big", { part:t("hga.wf." + big.key), v:V(fT(big.d)), pct:pct(Math.max(0, big.d), gap) }) : ""),
    rg ? t("hga.read.role", { role:t("hga.role." + rg.role), a:V(fT(rg.ideal)), b:V(fT(rg.hga)) }) + (bound && rg.role !== "comm" ? " " + t("hga.read.bound." + bound, { c:fT(rb.compute), m:fT(rb.memory), n:fT(rb.comm) }) : "") : "",
    ph.hga.launch > 0.05*ph.hga.t ? t("hga.read.launch", { n:fInt(ph.hga.kernels), v:V(fT(ph.hga.launch)), pct:pct(ph.hga.launch, ph.hga.t) }) : "",
    h.phase === "decode" ? t(r.fits.hga ? "hga.read.fits" : "hga.read.over", { b:fInt(h.b), hga:fInt(r.maxB.hga), ideal:fInt(r.maxB.ideal) }) : "" ]);
  $("ih-warn").innerHTML = hgaWarnings(r.hgaWarnings);
  $("ih-src").textContent = hgaSrc(r);
  renderHgaInferCurve();
}
function renderHgaInferCurve(){
  const h = S.hgaI, q = Object.assign(hgaInferQuery(), { axis:h.axis }), cv = Scouter.hga(q).curve;
  hgaSeg($("ih-axis"), [["b", t("hga.axis.b")], ["isl", t("hga.axis.isl")]], h.axis, k => { h.axis = k; renderHgaInferCurve(); });
  hgaSeg($("ih-metric"), [["step", t("hga.metric.step")], ["thr", t("hga.metric.thr")]], h.metric, k => { h.metric = k; renderHgaInferCurve(); });
  const pts = cv ? cv.points.filter(x => x.ok !== false) : [], thr = h.metric === "thr";
  const val = (x, side) => thr ? x.tok[side] : x.decode[side], note = (x, side) => x.fits && !x.fits[side] ? { hollow:true, note:t("hga.overCap") } : null;
  const series = [{ name:t("hga.ideal"), col:"s1", pts:pts.map(x => [x.x, val(x, "ideal"), note(x, "ideal")]) }, { name:"HuggingArch", col:"s2", pts:pts.map(x => [x.x, val(x, "hga"), note(x, "hga")]) }];
  lineChart("ih-curve", { series, labelEnds:false, xLog:true, xBase:2, xFmt:fxPow, yFmt:thr ? fN : fT, yMin:0, xLab:t("hga.axis." + h.axis), yLab:t("hga.metric." + h.metric),
    vlines:[{ x:h.axis === "b" ? h.b : h.isl, label:t("hga.now") }], empty:t("hga.curveEmpty") });
  legendInto("ih-curve-leg", series.map(x => ({ name:x.name, col:x.col, line:true })));
}

/* ================= 训练 ================= */
function hgaTrainQuery(){ const h=S.hgaT; return { model:S.modelKey, engine:h.engine, sig:h.sig, hw:S.hw, seq:h.seq, mbs:h.mbs }; }
function renderHgaTrain(){
  const h = S.hgaT, body = $("th-body");
  if(!Hga.sigsFor(h.engine, S.modelKey).some(s => s.id === h.sig)) h.sig = Hga.sigsFor(h.engine, S.modelKey)[0].id;
  hgaSeg($("th-eng"), Hga.TRAIN.map(e => [e, HGA_ENG[e]]), h.engine, k => { h.engine = k; renderHgaTrain(); });
  hgaSigSel($("th-sig"), h.engine, h.sig, v => { h.sig = v; renderHgaTrain(); });
  const sq = $("th-seq"); sq.max = LOGS.seq.n; sq.value = logI("seq", h.seq); $("th-seq-o").textContent = fInt(h.seq);
  sq.oninput = e => { h.seq = logV("seq", +e.target.value); renderHgaTrain(); };
  hgaSeg($("th-mbs"), Hga.GRID.mbs.map(v => [String(v), String(v)]), String(h.mbs), k => { h.mbs = +k; renderHgaTrain(); });
  const sig = Hga.sigOf(h.engine, h.sig, S.modelKey);
  $("th-flags").innerHTML = '<code>' + esc(sig.flags) + '</code>';
  if(!hgaGate("th", body)) return;
  const r = Scouter.hgaTrain(hgaTrainQuery());
  if(!r.ok){ hgaFail("th", body, r); return; }
  body.hidden = false;
  IDEAL_SER.name = t("hga.ideal");
  const I = r.ideal, H = r.hga, c = r.config || {};
  $("th-lede").innerHTML = t("hga.trn.lede", { model:esc(S.model.short), hw:esc(HW[S.hw].name), fw:HGA_ENG[h.engine], world:r.world, seq:fInt(h.seq), mbs:h.mbs, gtok:fN(r.gtok),
    m:fInt(H.m), rc:t("hga.rc." + (c.recompute || "none")) });
  kpis($("th-kpis"), [
    [t("hga.kpi.step"), fT(H.t), t("hga.kpi.vsIdeal", { v:fT(I.t), x:hgaRatio(I.t, H.t) })],
    [t("hga.kpi.thr"), fN(H.tokPerGpu) + " tok/s/GPU", t("hga.kpi.thrFoot", { v:fN(I.tokPerGpu) })],
    ["MFU", (H.mfu*100).toFixed(1) + "%", t("hga.kpi.mfuFoot", { v:(I.mfu*100).toFixed(1) + "%" })],
    [t("hga.kpi.mem"), fB(H.mem), t("hga.kpi.memFoot", { v:fB(I.mem), hbm:HW[S.hw].mem }), !H.fits] ]);
  waterfall("th-wf", r.waterfall.map(x => Object.assign({}, x, { label:t("hga.twf." + x.key), note:t("hga.twfTip." + x.key), col:x.key === "hga" ? "s2" : "s1" })), { fmt:fT, up:"c-a2a", down:"c-ar", accLabel:t("hga.wfAcc") });
  pairBars("th-pass", r.passes.filter(x => x.ideal > 0 || x.hga > 0).map(x => ({ label:t("hga.pass." + x.key), v:[x.ideal, x.hga] })), [IDEAL_SER, HGA_SER], { fmt:fT, ratio:t("hga.ratio") });
  legendInto("th-pass-leg", [{ name:IDEAL_SER.name, col:"s1" }, { name:HGA_SER.name, col:"s2" }]);
  const roles = r.roles.map(x => ({ role:x.role, ideal:x.ideal.t, hga:x.hga.t }));
  pairBars("th-roles", roles.map(x => ({ label:t("hga.role." + x.role), v:[x.ideal, x.hga] })), [IDEAL_SER, HGA_SER], { fmt:fT, ratio:t("hga.ratio") });
  legendInto("th-roles-leg", [{ name:IDEAL_SER.name, col:"s1" }, { name:HGA_SER.name, col:"s2" }]);
  table($("th-dp"), [t("hga.dp.name"), t("hga.dp.op"), t("hga.dp.world"), t("hga.dp.bytes"), t("hga.dp.t")],
    H.dpParts.map(x => ['<code>' + esc(x[0]) + '</code>', '<code>' + esc(x[1]) + '</code>', fInt(x[2]), fB(x[3]), fT(x[4])])
      .concat([{ cls:"hl", cells:["<b>" + esc(t("hga.dp.total")) + "</b>", "", "", "", t("hga.dp.exposed", { serial:fT(H.dpSerial), exposed:fT(H.dp) })] }]), [2, 3, 4]);
  const M = H.memParts;
  table($("th-mem"), ["", "HuggingArch", t("hga.ideal")], [
    [t("hga.mem.weights"), fB(M.weights), ""], [t("hga.mem.grads"), fB(M.grads), ""], [t("hga.mem.optimizer"), fB(M.optimizer), ""], [t("hga.mem.act"), fB(M.activations), ""],
    { cls:"hl", cells:["<b>" + esc(t("hga.mem.total")) + "</b>", fB(H.mem), fB(I.mem)] }], [1, 2]);
  const parts = r.waterfall.filter(x => x.d != null), gap = H.t - I.t, big = parts.slice().sort((a, b) => Math.abs(b.d) - Math.abs(a.d))[0];
  const rg = roles.slice().sort((a, b) => Math.abs(b.hga - b.ideal) - Math.abs(a.hga - a.ideal))[0];
  $("th-read").innerHTML = ul([
    hgaGapLine(I.t, H.t, roles, big && gap > 0 ? t("hga.read.big", { part:t("hga.twf." + big.key), v:V(fT(big.d)), pct:pct(Math.max(0, big.d), gap) }) : ""),
    t("hga.read.micro", { m:fInt(H.m), hf:V(fT(H.fwd)), hb:V(fT(H.bwd)), hr:V(fT(H.rc)), f:fT(I.fwd), b:fT(I.bwd), r:fT(I.rc) }),
    rg && H.t >= I.t ? t("hga.read.trole", { role:t("hga.role." + rg.role), a:V(fT(rg.ideal)), b:V(fT(rg.hga)) }) : "",
    H.dpSerial > 0 ? t("hga.read.dp", { serial:V(fT(H.dpSerial)), exposed:V(fT(H.dp)), ideal:fT(I.dp) }) : "",
    !H.fits ? t("hga.read.memOver", { v:V(fB(H.mem)), hbm:HW[S.hw].mem }) : "" ]);
  $("th-warn").innerHTML = hgaWarnings(r.hgaWarnings);
  $("th-src").textContent = hgaSrc(r);
  hgaSeg($("th-metric"), [["step", t("hga.metric.tstep")], ["thr", t("hga.metric.thr")], ["mfu", "MFU"]], h.metric, k => { h.metric = k; renderHgaTrain(); });
  const cv = Scouter.hgaTrain(Object.assign(hgaTrainQuery(), { axis:"seq" })).curve, pts = cv ? cv.points.filter(x => x.ok !== false) : [];
  const val = (x, side) => h.metric === "thr" ? x.tok[side] : h.metric === "mfu" ? x.mfu[side] : x[side];
  const series = [{ name:t("hga.ideal"), col:"s1", pts:pts.map(x => [x.x, val(x, "ideal")]) }, { name:"HuggingArch", col:"s2", pts:pts.map(x => [x.x, val(x, "hga")]) }];
  lineChart("th-curve", { series, labelEnds:false, xLog:true, xBase:2, xFmt:fxPow, yFmt:h.metric === "mfu" ? (v => (v*100).toFixed(0) + "%") : h.metric === "thr" ? fN : fT, yMin:0,
    xLab:t("hga.axis.seq"), yLab:h.metric === "mfu" ? "MFU" : t("hga.metric." + (h.metric === "step" ? "tstep" : h.metric)), vlines:[{ x:h.seq, label:t("hga.now") }], empty:t("hga.curveEmpty") });
  legendInto("th-curve-leg", series.map(x => ({ name:x.name, col:x.col, line:true })));
}
