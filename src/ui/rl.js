/* ================= RL：训推不一致、低精度 rollout、QAT。rollout 照搬推理核心，训练照搬后训练核心 ================= */
const MM_PUB = [
  ["lmsys","2025-11-25-fp8-rl",t("rl.mismatch.pub.fp8Rl")],
  ["lmsys","2026-04-25-deepseek-v4",t("rl.mismatch.pub.deepseekV4")],
  ["lmsys","2026-06-04-nvidia-run-nemotron-3-ultra",t("rl.mismatch.pub.nemotron3Ultra")],
  ["lmsys","2026-07-10-rocm-miles-dsv4",t("rl.mismatch.pub.rocmMilesDsv4")],
  ["lmsys","2026-07-15-inkling-day0-support",t("rl.mismatch.pub.inklingDay0")],
  ["lmsys","2026-08-18-miles-v0-1",t("rl.mismatch.pub.milesV01")],
  ["lmsys","2026-09-10-deepseek-v41",t("rl.mismatch.pub.deepseekV41")],
  {ev:/train-inference-consistency/,t:t("rl.mismatch.pub.vllmTrainInferConsistency")},
  ["lmsys","2026-10-15-meta-muse-spark-open",t("rl.mismatch.pub.museSparkOpen")],
  ["lmsys","2026-10-16-beam",t("rl.mismatch.pub.beam")],
  ["lmsys","2026-10-25-minite",t("rl.mismatch.pub.minite")],
  ["lmsys","2026-10-22-miles-mi355x-agentic-rl",t("rl.mismatch.pub.mi355xAgenticRl")]];
const MM_OSS = [
  ["slime","v0.2.0",t("rl.mismatch.oss.slimeV020")],
  ["sglang","v0.5.6",t("rl.mismatch.oss.sglangV056")],
  ["slime","v0.2.1",t("rl.mismatch.oss.slimeV021")],
  ["sglang","v0.5.7",t("rl.mismatch.oss.sglangV057")],
  ["slime","v0.2.2",t("rl.mismatch.oss.slimeV022")],
  ["areal","v1.0.0",t("rl.mismatch.oss.arealV100")],
  ["areal","v1.0.1",t("rl.mismatch.oss.arealV101")],
  ["areal","v1.0.4",t("rl.mismatch.oss.arealV104")],
  ["vllm","v0.21.0",t("rl.mismatch.oss.vllmV0210")],
  ["lmsys","2026-05-13-no-token-left-behind",t("rl.mismatch.oss.noTokenLeftBehind")],
  ["areal","v2.0.0",t("rl.mismatch.oss.arealV200")],
  ["slime","v0.3.1",t("rl.mismatch.oss.slimeV031")],
  ["areal","v2.1.0",t("rl.mismatch.oss.arealV210")],
  ["sglang","v0.5.20",t("rl.mismatch.oss.sglangV0520")],
  ["miles","v0.1.2",t("rl.mismatch.oss.milesV012")],
  ["slime","v0.4.0",t("rl.mismatch.oss.slimeV040")]];
const DET_OSS = [
  ["sglang","v0.5.5",t("rl.determinism.oss.sglangV055")],
  ["vllm","v0.11.1",t("rl.determinism.oss.vllmV0111")],
  ["lmsys","2025-11-19-miles",t("rl.determinism.oss.miles")],
  ["slime","v0.2.0",t("rl.determinism.oss.slimeV020")],
  ["sglang","v0.5.6",t("rl.determinism.oss.sglangV056")],
  ["vllm","v0.12.0",t("rl.determinism.oss.vllmV0120")],
  ["slime","v0.2.1",t("rl.determinism.oss.slimeV021")],
  ["vllm","v0.13.0",t("rl.determinism.oss.vllmV0130")],
  ["vllm","v0.20.0",t("rl.determinism.oss.vllmV0200")],
  ["vllm","v0.22.0",t("rl.determinism.oss.vllmV0220")],
  ["sglang","v0.5.18",t("rl.determinism.oss.sglangV0518")],
  ["areal","v2.1.0",t("rl.determinism.oss.arealV210")],
  ["vllm","v0.28.0",t("rl.determinism.oss.vllmV0280")],
  ["slime","v0.3.2",t("rl.determinism.oss.slimeV032")],
  ["vllm","v0.29.0",t("rl.determinism.oss.vllmV0290")],
  ["sglang","v0.5.21",t("rl.determinism.oss.sglangV0521")],
  ["vllm","v0.31.0",t("rl.determinism.oss.vllmV0310")],
  ["vllm","v0.32.0",t("rl.determinism.oss.vllmV0320")]];
const LP_PUB = [
  ["lmsys","2025-11-25-fp8-rl",t("rl.lowprec.pub.fp8Rl")],
  ["lmsys","2026-01-26-int4-qat",t("rl.lowprec.pub.int4Qat")],
  ["lmsys","2026-07-29-mxfp8-nvfp4-rl",t("rl.lowprec.pub.mxfp8Nvfp4Rl")],
  ["lmsys","2026-08-12-qwen3-8-day0-support",t("rl.lowprec.pub.qwen38Day0")],
  ["lmsys","2026-10-25-minite",t("rl.lowprec.pub.minite")]];
const WS_PUB = [
  ["sglang","v0.5.6",t("rl.lowprec.ws.oss.sglangV056")],
  ["vllm","v0.16.0",t("rl.lowprec.ws.oss.vllmV0160")],
  ["vllm","v0.17.0",t("rl.lowprec.ws.oss.vllmV0170")],
  ["areal","v1.0.3",t("rl.lowprec.ws.oss.arealV103")],
  ["lmsys","2026-04-29-p2p-update",t("rl.lowprec.ws.oss.p2pUpdate")],
  ["areal","v1.0.4",t("rl.lowprec.ws.oss.arealV104")],
  ["vllm","v0.21.0",t("rl.lowprec.ws.oss.vllmV0210")],
  ["slime","v0.3.0",t("rl.lowprec.ws.oss.slimeV030")],
  ["slime","v0.3.1",t("rl.lowprec.ws.oss.slimeV031")],
  ["miles","v0.1.0",t("rl.lowprec.ws.oss.milesV010")],
  ["areal","v2.1.0",t("rl.lowprec.ws.oss.arealV210")],
  ["vllm","v0.29.0",t("rl.lowprec.ws.oss.vllmV0290")],
  ["miles","v0.1.1",t("rl.lowprec.ws.oss.milesV011")],
  ["sglang","v0.5.21",t("rl.lowprec.ws.oss.sglangV0521")],
  ["miles","v0.1.3",t("rl.lowprec.ws.oss.milesV013")],
  {ev:/minite/,t:t("rl.lowprec.ws.oss.minite")}];
const QAT_OSS = [
  ["slime","v0.2.2",t("rl.qat.oss.slimeV022")],
  ["lmsys","2026-01-26-int4-qat",t("rl.qat.oss.int4Qat")],
  ["slime","v0.2.3",t("rl.qat.oss.slimeV023")],
  {ev:/humansand/,t:t("rl.qat.oss.fourBitterLesson")},
  ["lmsys","2026-07-29-mxfp8-nvfp4-rl",t("rl.qat.oss.mxfp8Nvfp4Rl")],
  ["miles","v0.1.0",t("rl.qat.oss.milesV010")],
  ["slime","v0.3.2",t("rl.qat.oss.slimeV032")],
  ["miles","v0.1.1",t("rl.qat.oss.milesV011")],
  {ev:/train-inference-consistency/,t:t("rl.qat.oss.vllmTrainInferConsistency")},
  ["lmsys","2026-10-25-minite",t("rl.qat.oss.minite")]];

function renderMismatch(){
  const {m,d}=ctx(), Sq=S.trn.S, e=m.moe, f=m.full, nL=e.E?d.nMoe:m.L, idB=e.E>256?2:1;
  const kvTok=Math.max(0,f.n-(f.swa||0))*d.kvTokElems*2, r3=e.E?d.nMoe*e.k*idB:0;
  const R=[[t("rl.mismatch.carry.tito"),4,t("rl.mismatch.carry.titoNote")],
    [t("rl.mismatch.carry.logprob"),4,t("rl.mismatch.carry.logprobNote")],
    ...(e.E?[[t("rl.mismatch.carry.r3", {layers:d.nMoe, k:e.k}),r3,t("rl.mismatch.carry.r3Note", {bits:idB*8, x:4/idB})]]:[]),
    [t("rl.mismatch.carry.mask"),160,t("rl.mismatch.carry.maskNote")]];
  const tot=sum(R,x=>x[1]);
  $("r-mm-lede").innerHTML=t("rl.mismatch.carry.lede", {model:esc(m.short), S:fInt(Sq), r3:e.E ? t("rl.mismatch.carry.ledeR3", {pct:V(pct(r3,kvTok))}) : ""});
  table($("r-mm"),[t("rl.mismatch.carry.head.what"),t("rl.mismatch.carry.head.perTok"),t("rl.mismatch.carry.head.perSeq"),t("rl.mismatch.carry.head.note")],R.map(x=>[esc(x[0]),fB(x[1]),fB(x[1]*Sq),esc(x[2])]).concat([
    {cls:"hl",cells:["<b>"+t("rl.mismatch.carry.total")+"</b>",fB(tot),fB(tot*Sq),""]},
    [t("rl.mismatch.carry.kv"),fB(kvTok),fB(kvTok*Sq),t("rl.mismatch.carry.kvNote")],
    [e.E ? t("rl.mismatch.carry.hiddenMoe") : t("rl.mismatch.carry.hidden"),fB(nL*m.D*2),fB(nL*m.D*2*Sq),t("rl.mismatch.carry.hiddenNote")]]),[1,2]);
  ossTable($("r-mm-pub"),MM_PUB);
  ossTable($("r-mm-oss"),MM_OSS);
}

/* 确定性 rollout：哪些地方随 batch 变、四个层级、代价、和低精度的关系。数字都从 determinism_public / quant_evidence / releases 取 */
function renderDeterminism(){
  const {m,hw,topo,g}=ctx(), f=Math.min(1,Math.max(0,S.q.roll)), D=m.D, rAR=Math.floor(512*1024/(2*D)), e=m.moe;
  const u=detOf("tml-unique"), tc=detOf("tml-cost"), lc=detOf("lmsys-det-cost"), ev=evOf(/train-inference-consistency/), vt=detOf("vllm-tt-bitwise");
  $("r-det-lede").innerHTML=t("rl.determinism.lede", {tml:detLink("tml-unique","Thinking Machines Lab"), nDefault:V(u.nDefault), nInvariant:u.nInvariant, model:esc(m.short), D:fInt(D)});
  table($("r-det-ops"),[t("rl.determinism.ops.head.where"),t("rl.determinism.ops.head.why"),t("rl.determinism.ops.head.fix"),t("rl.determinism.ops.head.src")],[
    [t("rl.determinism.ops.norm.where"),t("rl.determinism.ops.norm.why"),t("rl.determinism.ops.norm.fix"),detLink("tml-unique","TML")],
    ["GEMM",t("rl.determinism.ops.gemm.why"),t("rl.determinism.ops.gemm.fix"),t("rl.determinism.ops.gemm.src", {tml:detLink("tml-unique","TML"), v0111:relLink("vllm","v0.11.1"), v0120:relLink("vllm","v0.12.0")})],
    [t("rl.determinism.ops.attn.where"),t("rl.determinism.ops.attn.why"),t("rl.determinism.ops.attn.fix"),t("rl.determinism.ops.attn.src", {tml:detLink("tml-unique","TML"), sglang:detLink("lmsys-det-cost","SGLang")})],
    [t("rl.determinism.ops.ar.where"),t("rl.determinism.ops.ar.why", {tok:V(fInt(rAR))}),t("rl.determinism.ops.ar.fix"),
      t("rl.determinism.ops.ar.src", {v0518:relLink("sglang","v0.5.18"), v0310:relLink("vllm","v0.31.0"), v0320:relLink("vllm","v0.32.0"), tbik:detLink("tbik","TBIK")})],
    ["MoE",e.E ? t("rl.determinism.ops.moe.why", {E:e.E, k:e.k}) : t("rl.determinism.ops.moe.whyDense"),
      t("rl.determinism.ops.moe.fix"),t("rl.determinism.ops.moe.src", {sglang:relLink("sglang","v0.5.21"), vllm:relLink("vllm","v0.28.0"), lmsys:detLink("lmsys-det-limit","LMSYS")})],
    [t("rl.determinism.ops.actQuant.where"),t("rl.determinism.ops.actQuant.why"),t("rl.determinism.ops.actQuant.fix"),t("rl.determinism.ops.actQuant.src")],
    [t("rl.determinism.ops.spec.where"),t("rl.determinism.ops.spec.why"),t("rl.determinism.ops.spec.fix"),t("rl.determinism.ops.spec.src")],
    [t("rl.determinism.ops.sampling.where"),t("rl.determinism.ops.sampling.why"),t("rl.determinism.ops.sampling.fix"),t("rl.determinism.ops.sampling.src", {sglang:detLink("lmsys-det-cost","SGLang"), v0518:relLink("sglang","v0.5.18")})]]);
  table($("r-det-lv"),[t("rl.determinism.lv.head.level"),t("rl.determinism.lv.head.what"),t("rl.determinism.lv.head.extra"),t("rl.determinism.lv.head.pub")],[
    [t("rl.determinism.lv.l1.name"),t("rl.determinism.lv.l1.what"),t("rl.determinism.lv.l1.extra"),
      t("rl.determinism.lv.l1.pub", {tml:detLink("tml-unique","TML"), nDefault:u.nDefault, nInvariant:u.nInvariant, sglang:relLink("sglang","v0.5.5"), vllm:relLink("vllm","v0.11.1")})],
    [t("rl.determinism.lv.l2.name"),t("rl.determinism.lv.l2.what"),t("rl.determinism.lv.l2.extra"),t("rl.determinism.lv.l2.pub", {repro:detLink("lmsys-slime-repro","slime + SGLang"), slime:relLink("slime","v0.3.2")})],
    [t("rl.determinism.lv.l3.name"),t("rl.determinism.lv.l3.what"),t("rl.determinism.lv.l3.extra"),
      t("rl.determinism.lv.l3.pub", {tt:detLink("vllm-tt-bitwise","vLLM + TorchTitan"), slow:vt.slow, vllm:aOut(ev.url,"vLLM 2026-10"), miles:relLink("lmsys","2025-11-19-miles","Miles"), sglang:relLink("sglang","v0.5.5")})],
    [t("rl.determinism.lv.l4.name"),t("rl.determinism.lv.l4.what"),t("rl.determinism.lv.l4.extra"),t("rl.determinism.lv.l4.pub", {tbik:detLink("tbik","TBIK")})]]);
  // 代价
  const sp=x=>Array.isArray(x)?x.map(v=>f*v+(1-f)):[f*x+(1-f)], fx=a=>{ const lo=Math.min(...a), hi=Math.max(...a); return "×"+(lo===hi?lo.toFixed(2):lo.toFixed(2)+"–"+hi.toFixed(2)); };
  const dec=bi=>Infer.decode(m,hw,topo,Object.assign({},S.inf,{phase:"decode",bi,specK:0}),g), sBi=dec(true).stepT/dec(false).stepT;
  $("r-det-cost-lede").innerHTML=t("rl.determinism.cost.lede", {f, v0120:relLink("vllm","v0.12.0"), v0220:relLink("vllm","v0.22.0"), v0290:relLink("vllm","v0.29.0"),
    graph:detLink("lmsys-det-graph",t("rl.determinism.cost.graphLink"))});
  const C=[
    [t("rl.determinism.cost.sglang.name"),detLink("lmsys-det-cost","LMSYS 2025-09"),t("rl.determinism.cost.sglang.value", {pct:((lc.slow-1)*100).toFixed(2)}),t("rl.determinism.cost.rollout"),sp(lc.slow)],
    [t("rl.determinism.cost.tml.name"),detLink("tml-cost","TML 2025-09"),t("rl.determinism.cost.tml.value", {base:tc.tBase, det:tc.tDet, opt:tc.tDetOpt}),t("rl.determinism.cost.rollout"),sp([tc.tDet/tc.tBase,tc.tDetOpt/tc.tBase])],
    [t("rl.determinism.cost.fwd.name"),aOut(ev.url,"vLLM 2026-10"),t("rl.determinism.cost.fwd.value", {thr:ev.thr.map(x=>(x*100).toFixed(1)+"%").join(" / ")}),t("rl.determinism.cost.fwd.side"),sp(ev.thr.map(x=>1/x))],
    [t("rl.determinism.cost.tt.name"),detLink("vllm-tt-bitwise","vLLM 2025-11"),t("rl.determinism.cost.tt.value", {x:vt.slow}),t("rl.determinism.cost.tt.side"),[vt.slow]],
    [t("rl.determinism.cost.page.name", {f:S.inf.biFactor}),t("rl.determinism.cost.page.src"),
      t("rl.determinism.cost.page.value", {x:sBi.toFixed(2), tail:sBi<S.inf.biFactor-0.01 ? t("rl.determinism.cost.page.commSame") : t("rl.determinism.cost.page.noComm")}),t("rl.determinism.cost.rollout"),sp(sBi)]];
  table($("r-det-cost"),[t("rl.determinism.cost.head.how"),t("rl.determinism.cost.head.src"),t("rl.determinism.cost.head.cost"),t("rl.determinism.cost.head.side"),t("rl.determinism.cost.head.step", {f})],C.map(x=>[esc(x[0]),x[1],esc(x[2]),esc(x[3]),fx(x[4])]),[4]);
  // 和低精度的关系
  const qh=hw, lk=Quant.schemeOK(qh,"nvfp4")?"nvfp4":"fp8", r16=qInferE2E("bf16"), r4=qInferE2E(lk), sr=r16.stepT/r4.stepT;   // Hopper 没有 NVFP4：换成 block FP8
  $("r-det-lp").innerHTML=ul([
    t("rl.determinism.lp.necessary", {vllm:aOut(ev.url,"vLLM 2026-10")}),
    t("rl.determinism.lp.granularity"),
    t("rl.determinism.lp.kernels", {v0200:relLink("vllm","v0.20.0"), v0220:relLink("vllm","v0.22.0"), v0280:relLink("vllm","v0.28.0"), v0521:relLink("sglang","v0.5.21")}),
    t("rl.determinism.lp.both", {hw:esc(qh.name), scheme:DTN[lk], x:V(fX(sr)), slow:lc.slow.toFixed(2), net:V(fX(sr/lc.slow))})]);
  table($("r-det-pub"),[t("rl.determinism.pub.head.date"),t("rl.determinism.pub.head.src"),t("rl.determinism.pub.head.what"),t("rl.determinism.pub.head.model"),t("rl.determinism.pub.head.setting"),t("rl.determinism.pub.head.value")],DB.set("determinism_public").slice().sort((x,y)=>x.date.localeCompare(y.date))
    .map(p=>['<span class="v">'+esc(p.date)+'</span>','<span class="nw">'+aOut(p.url,p.src)+'</span>'+(p.code?"<br>"+aOut(p.code,t("rl.determinism.pub.code")):""),'<span class="nw">'+esc(p.what)+'</span>',esc(p.model)+(p.hw&&p.hw!=="—"?"<br>"+'<span class="tag">'+esc(p.hw)+'</span>':""),esc(p.setting),esc(p.value)]));
  ossTable($("r-det-oss"),DET_OSS);
}

function renderLowprec(){
  const {hw,f}=qRL(), {m,d,topo}=ctx(), hwS=HW[S.hw];
  $("r-lp-lede").innerHTML=t("rl.lowprec.lede", {engine:esc(ENGN[S.inf.engine]||S.inf.engine), N:S.inf.N, B:S.inf.B, hw:esc(hw.name), formula:V("1 / (f / s_rollout + (1 − f) / s_train)"), f});
  ossTable($("r-lp-pub"),LP_PUB);
  // 权重同步：一次全量要搬的字节
  const lin=d.total-d.pEmb-d.pHead, ex=d.pExpertsAll, rest=d.total-ex, emb=(d.pEmb+d.pHead)*2;
  const F=[["BF16",d.total*2,t("rl.lowprec.ws.who.bf16")],[t("rl.lowprec.ws.fmt.fp8"),lin*Common.dtB("fp8")+emb,t("rl.lowprec.ws.who.fp8")],[t("rl.lowprec.ws.fmt.mxfp8"),lin*Common.dtB("mxfp8")+emb,t("rl.lowprec.ws.who.mxfp8")],
    ...(ex?[[t("rl.lowprec.ws.fmt.nvfp4"),ex*Common.dtB("nvfp4")+rest*2,t("rl.lowprec.ws.who.nvfp4")],[t("rl.lowprec.ws.fmt.int4"),ex*(0.5+2/32)+rest*2,t("rl.lowprec.ws.who.int4")]]:[])];
  const ib=hwS.ib*1e9, N=topo.gpus;
  $("r-ws-lede").innerHTML=t("rl.lowprec.ws.lede", {params:fN(d.total), hw:hwS.name, ib:hwS.ib});
  table($("r-ws"),[t("rl.lowprec.ws.head.fmt"),t("rl.lowprec.ws.head.bytes"),t("rl.lowprec.ws.head.vsBf16"),t("rl.lowprec.ws.head.one"),t("rl.lowprec.ws.head.sharded", {N}),t("rl.lowprec.ws.head.who")],F.map(([n,b,w])=>[esc(n),fB(b),pct(b,F[0][1]),fT(b/ib),fT(b/(N*ib)),esc(w)]),[1,2,3,4]);
  ossTable($("r-ws-pub"),WS_PUB);
}

function renderQat(){
  const {q,hw,g,Q,bw}=qHead(), N=q.N, K=q.K, Mt=Math.max(16,q.Mt), to={nMicro:q.nMicro,tokFused:q.tokFused};
  const G=(k,o)=>Q.trainGemm(hw,g,k,Mt,N,K,Object.assign({},to,o)), b=G("bf16");
  const fq=Q.fakeQuantT(hw,g,N,K,q.nMicro);
  $("r-qat-lede").innerHTML=t("rl.qat.lede", {N:fInt(N), K:fInt(K), M:fInt(Mt), hw:esc(hw.name)});
  const rows=[
    [t("rl.qat.row.bf16Ptq"),"BF16","BF16",t("rl.qat.grad.unquant"),t("rl.qat.match.no"),b,t("rl.qat.who.bf16Ptq"),"bf16"],
    [t("rl.qat.row.fakeQuant"),t("rl.qat.fwd.fakeQuant"),t("rl.qat.bwd.ste"),t("rl.qat.grad.quant"),t("rl.qat.match.weights"),{t:b.t+fq,extra:b.extra},t("rl.qat.who.fakeQuant"),"bf16"],
    [t("rl.qat.row.mxfp8"),"MXFP8","MXFP8",t("rl.qat.grad.quantNoise"),t("rl.qat.match.rowOnly"),G("mxfp8"),"Miles --fp8-recipe mxfp8","mxfp8"],
    [t("rl.qat.row.mxfp8hp"),"MXFP8",t("rl.qat.bwd.orig"),t("rl.qat.grad.unquantMismatch"),t("rl.qat.match.yes"),G("mxfp8hp"),"TE NVTE_BACKWARD_OVERRIDE=high_precision","mxfp8hp"],
    [t("rl.qat.row.nvfp4hp"),t("rl.qat.fwd.nvfp4Tok"),t("rl.qat.bwd.orig"),t("rl.qat.grad.unquantMismatch"),t("rl.qat.match.yes"),G("nvfp4hp"),t("rl.qat.who.nvfp4hp"),"nvfp4hp"],
    [t("rl.qat.row.nvfp4dq"),t("rl.qat.fwd.nvfp4Tok"),t("rl.qat.bwd.deq"),t("rl.qat.grad.quantSte"),t("rl.qat.match.yes"),G("nvfp4dq"),t("rl.qat.who.nvfp4dq"),"nvfp4dq"],
    [t("rl.qat.row.nvfp4"),t("rl.qat.fwd.nvfp4W2d"),t("rl.qat.bwd.nvfp4"),t("rl.qat.grad.quantNoise"),t("rl.qat.match.te"),G("nvfp4"),t("rl.qat.who.nvfp4"),"nvfp4"]].filter(x=>Q.recipeOK(hw,x[7]));   // Hopper 只剩前两行
  table($("r-qat"),[t("rl.qat.head.how"),t("rl.qat.head.fwd"),t("rl.qat.head.bwd"),t("rl.qat.head.grad"),t("rl.qat.head.match"),t("rl.qat.head.time"),t("rl.qat.head.vsBf16"),t("rl.qat.head.saved"),t("rl.qat.head.who")],
    rows.map(x=>[esc(x[0]),esc(x[1]),esc(x[2]),esc(x[3]),esc(x[4]),fT(x[5].t),fX(b.t/x[5].t),fB(x[5].extra),esc(x[6])]),[5,6,7]);
  const ideal=(f,bk)=>Q.idealSpeedup(hw,f,bk);
  const hp=G("nvfp4hp"), hpU=G("nvfp4hp",{tokFused:false}), hpF=G("nvfp4hp",{tokFused:true}), dq=G("nvfp4dq"), tDeq=(dq.q.find(z=>z[2]==="deq")||[0,0])[1];
  const mem=evOf(/mxfp8-nvfp4-rl/), A=[];
  if(bw){
  A.push(t("rl.qat.an.peak", {hw:esc(hw.name), mx:V(fX(ideal("mxfp8","bf16"))), nv:V(fX(ideal("nvfp4","bf16"))), nvAll:fX(ideal("nvfp4","nvfp4"))}));
  A.push(t("rl.qat.an.tok", {state:q.tokFused ? t("rl.qat.an.tokFused") : t("rl.qat.an.tokSeparate"), fused:fT(hpF.t), unfused:V(fT(hpU.t)), xFused:fX(b.t/hpF.t), xUnfused:fX(b.t/hpU.t),
    lmsys:aOut(mem.url,t("rl.qat.an.tokLink"))}));
  A.push(t("rl.qat.an.deq", {t:V(fT(tDeq)), pct:pct(tDeq,dq.t), dq:fB(dq.saveX), hp:fB(hp.saveX), same:aOut(mem.url,t("rl.qat.an.deqLink"))})); }
  A.push(t("rl.qat.an.e2e"));
  $("r-qat-notes").innerHTML=ul(A);
  ossTable($("r-qat-oss"),QAT_OSS);
}
