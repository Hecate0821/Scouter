/* 量化核心：只描述量化本身带来的差别。单个 GEMM 的权重 / scale / 激活字节，激活量化 kernel，W4A16 的反量化，
   MoE grouped GEMM 的每 expert 行数与对齐补齐，训练三遍 GEMM 的精度与转置副本。统一按 Blackwell（SM100 / SM103）的块缩放 MMA 算。 */
const Quant = (function(){
const { dtB, peakOf, roof, memT } = Common;

/* 推理方案：w / a 是 Common.DT 里的格式；pk 是 MMA 实际跑在 HW 表的哪一列峰值上（取两个操作数里精度高的那个） */
const SCHEMES = {
  bf16:  { name:Msg.of("engine_quant.scheme.bf16"), short:"W16A16", w:"bf16",  a:"bf16",  pk:"bf16" },
  fp8:   { name:Msg.of("engine_quant.scheme.fp8"), short:"W8A8 block", w:"fp8", a:"fp8a", pk:"fp8" },
  mxfp8: { name:Msg.of("engine_quant.scheme.mxfp8"), short:"W8A8 MX", w:"mxfp8", a:"mxfp8", pk:"fp8" },
  w4a16: { name:Msg.of("engine_quant.scheme.w4a16"), short:"W4A16", w:"mxfp4", a:"bf16",  pk:"bf16", deq:true },
  w4a8:  { name:Msg.of("engine_quant.scheme.w4a8"), short:"W4A8", w:"mxfp4", a:"mxfp8", pk:"fp8" },
  mxfp4: { name:Msg.of("engine_quant.scheme.mxfp4"), short:"W4A4 MX", w:"mxfp4", a:"mxfp4", pk:"fp4" },
  nvfp4: { name:Msg.of("engine_quant.scheme.nvfp4"), short:"W4A4 NV", w:"nvfp4", a:"nvfp4", pk:"fp4", tok:true }
};
const ORDER = ["bf16","fp8","mxfp8","w4a16","w4a8","mxfp4","nvfp4"];

/* 单个 GEMM：Y[M,N] = X[M,K]·W[N,K]ᵀ。
   o.fuse：激活量化融合进上一个算子（RMSNorm / SwiGLU / dispatch 前的量化）还是单独一个 kernel；
   o.align：M 按多少行补齐（MoE 的每个 expert、或 MMA 的 tile）；o.BM、o.deqRate：W4A16 每个 BM 行的 CTA 反量化一遍整块权重 */
function gemm(hw, g, key, M, N, K, o){
  const S=SCHEMES[key], bw=dtB(S.w), ba=dtB(S.a), pk=hw[S.pk]*1e12;
  const Mp = M>0 && o.align>1 ? Math.ceil(M/o.align)*o.align : M;
  const fl = 2*Mp*N*K;
  const wB = N*K*bw, wScale = N*K*(bw-(S.w==="bf16"?2:S.w==="fp8"||S.w==="mxfp8"?1:0.5));
  const aB = M*K*ba + (S.tok ? M*4 : 0), oB = M*N*2;
  const nk = o.floor===false ? 0 : 1;
  const r = roof(fl, wB+aB+oB, pk, hw, g, nk);
  const tDeq = S.deq ? N*K*Math.ceil(Math.max(M,1)/o.BM)/(o.deqRate*1e12) : 0;
  let tQ=0, qB=0;
  if(S.a!=="bf16" && o.fuse==="separate"){ qB = M*K*(2+ba) + (S.tok?M*4:0); tQ = roof(5*M*K, qB, peakOf(hw,"bf16"), hw, g, nk).t; }
  const tG = Math.max(r.tc+tDeq, r.tm, r.tf);
  const bound = tG===r.tf&&r.tf>0 ? "launch" : r.tc+tDeq>=r.tm ? "compute" : "memory";
  return { t:tG+tQ, tG, tc:r.tc, tm:r.tm, tDeq, tQ, fl, flReal:2*M*N*K, wB, wScale, aB, oB, qB, bytes:wB+aB+oB+qB, I:fl/(wB+aB+oB), bound, Mp, pk };
}
/* 计算侧追上访存侧的 M（tile 内不补齐时） */
function crossover(hw, g, key, N, K, o){
  let lo=1, hi=1<<20; const side=M=>{ const x=gemm(hw,g,key,M,N,K,Object.assign({},o,{align:1})); return x.tc+x.tDeq>=x.tm; };
  if(!side(hi)) return Infinity; if(side(lo)) return 1;
  for(let i=0;i<40;i++){ const mid=Math.sqrt(lo*hi); if(side(mid)) hi=mid; else lo=mid; } return hi;
}

/* MoE：均匀路由时每个 expert 的行数 X ~ Poisson(λ)，λ = T·k/E。返回 E[⌈X/a⌉·a]、E[⌈X/BM⌉]、P(X>0) */
function poisson(lam, a, BM){
  if(lam<=0) return { pad:0, tiles:0, nz:0 };
  if(lam>600){ const sd=Math.sqrt(lam), up=(x,b)=> b>sd ? Math.ceil(x/b)*b : x+(b-1)/2;   // 正态近似：块比波动大时落在同一个台阶
    return { pad: a>1 ? up(lam,a) : lam, tiles: up(lam,BM)/BM, nz:1 }; }
  let p=Math.exp(-lam), pad=0, tiles=0; const xmax=Math.ceil(lam+12*Math.sqrt(lam)+30);
  for(let x=1;x<=xmax;x++){ p*=lam/x; pad+=p*(a>1?Math.ceil(x/a)*a:x); tiles+=p*Math.ceil(x/BM); }
  return { pad, tiles, nz:1-Math.exp(-lam) };
}
/* FlashInfer TRTLLM MoE 按平均每 expert 行数挑 tile：next_pow2(λ)，夹在 8 到 64 之间（近似） */
const autoAlign = lam => Math.min(64, Math.max(8, Math.pow(2, Math.ceil(Math.log2(Math.max(lam,1))))));

/* 一层 routed expert（gate_up + SwiGLU + down），本 EP rank：T 是整个 EP 组这一层看到的 token */
function moeLayer(hw, g, m, key, T, ep, o){
  const S=SCHEMES[key], e=m.moe, De=e.Dlat||m.D, F=e.F, bw=dtB(S.w), ba=dtB(S.a), pk=hw[S.pk]*1e12;
  const lam=T*e.k/e.E, a = o.align==="auto" ? autoAlign(lam) : +o.align, P=poisson(lam, a, o.BM), El=e.E/ep;
  const rows=El*lam, rowsPad=El*P.pad, touched=El*P.nz, Tloc=T/ep;
  const fl = 6*rowsPad*De*F, wB = touched*3*De*F*bw;
  const aB = rows*(De*ba + 2*F*2 + F*ba + De*2) + (S.tok? rows*4*2 : 0);
  const r = roof(fl, wB+aB, pk, hw, g, 3);
  const tDeq = S.deq ? El*P.tiles*3*De*F/(o.deqRate*1e12) : 0;
  let tQ=0; if(S.a!=="bf16" && o.fuse==="separate") tQ = roof(5*Tloc*De, Tloc*De*(2+ba), peakOf(hw,"bf16"), hw, g, 1).t;
  const tG = Math.max(r.tc+tDeq, r.tm, r.tf);
  return { t:tG+tQ, tG, tc:r.tc, tm:r.tm, tDeq, tQ, lam, a, rows, rowsPad, waste: rowsPad>0?1-rows/rowsPad:0, touched, El, wB, aB,
    bound: tG===r.tf ? "launch" : r.tc+tDeq>=r.tm ? "compute" : "memory" };
}

/* 训练配方：f 前向，b 反向（dgrad + wgrad）；w2d 权重方块（一份 qdata 两个方向共用）；
   hp：反向用原始 BF16 操作数（orig）或前向量化值反量化后的 BF16（deq）；tok：NVFP4 激活按 token 的 FP32 scale；rht：wgrad 输入做随机 Hadamard */
const RECIPES = {
  bf16:    { name:Msg.of("engine_quant.recipe.bf16.name"), f:"bf16", b:"bf16", who:Msg.of("engine_quant.recipe.bf16.who") },
  fp8:     { name:Msg.of("engine_quant.recipe.fp8.name"), f:"fp8", b:"fp8", w2d:true, who:Msg.of("engine_quant.recipe.fp8.who") },
  mxfp8:   { name:Msg.of("engine_quant.recipe.mxfp8.name"), f:"mxfp8", b:"mxfp8", w2d:false, who:Msg.of("engine_quant.recipe.mxfp8.who") },
  mxfp8tt: { name:Msg.of("engine_quant.recipe.mxfp8tt.name"), f:"mxfp8", b:"mxfp8", w2d:true, who:Msg.of("engine_quant.recipe.mxfp8tt.who") },
  nvfp4:   { name:Msg.of("engine_quant.recipe.nvfp4.name"), f:"nvfp4", b:"nvfp4", w2d:true, rht:true, who:Msg.of("engine_quant.recipe.nvfp4.who") },
  // 下面三种是 RL 的配方（前向低精度、反向 BF16），只在 RL · 低精度训练与 QAT 一节用
  mxfp8hp: { name:Msg.of("engine_quant.recipe.mxfp8hp.name"), f:"mxfp8", b:"bf16", hp:"orig", who:Msg.of("engine_quant.recipe.mxfp8hp.who") },
  nvfp4hp: { name:Msg.of("engine_quant.recipe.nvfp4hp.name"), f:"nvfp4", b:"bf16", hp:"orig", tok:true, who:Msg.of("engine_quant.recipe.nvfp4hp.who") },
  nvfp4dq: { name:Msg.of("engine_quant.recipe.nvfp4dq.name"), f:"nvfp4", b:"bf16", hp:"deq", tok:true, who:Msg.of("engine_quant.recipe.nvfp4dq.who") }
};
const RORDER = ["bf16","fp8","mxfp8","mxfp8tt","nvfp4"];   // 后训练页只列 Megatron / TE / torchtitan 的默认配方

/* 一个线性层一个 micro-batch 的三遍 GEMM + 量化 kernel；权重量化每个 step 做一次，按 micro-batch 数摊开 */
function trainGemm(hw, g, key, M, N, K, o){
  const R=RECIPES[key], fq=R.f!=="bf16", bq=R.b!=="bf16", bf=dtB(R.f), bb=dtB(R.b), pkA=peakOf(hw,"bf16");
  const pass=(fl,by,pk)=>roof(fl,by,pk,hw,g,1).t;
  const fprop = pass(2*M*N*K, N*K*bf + M*K*bf + M*N*2, peakOf(hw,R.f));
  const dgrad = pass(2*M*N*K, N*K*bb + M*N*bb + M*K*2, peakOf(hw,R.b));
  const wgrad = pass(2*M*N*K, M*N*bb + M*K*bb + N*K*4*2, peakOf(hw,R.b));
  const q = [];   // [名字, 时间, 种类]：种类 x / w / tok / dy / deq 供界面按种类取，不按名字匹配
  const nm = Math.max(1,o.nMicro);
  if(fq){
    // X：一次读 BF16，写行方向（fprop）；反向低精度时同一个 kernel 再写列方向（wgrad）
    q.push([bq ? Msg.of("engine_quant.q.xRowCol") : Msg.of("engine_quant.q.xRow"), pass(M*K*(5 + (R.rht&&bq?32:0)), M*K*(2 + bf + (bq?bb:0)), pkA), "x"]);
    // W：1D 块反向要从 BF16 再量化一份列方向；2D 块一份 qdata，两种 scale 布局
    const wq = N*K*(2 + bf + (bq && !R.w2d ? bb : 0));
    q.push([bq ? (R.w2d ? Msg.of("engine_quant.q.w2d", {n:nm}) : Msg.of("engine_quant.q.w1dRowCol", {n:nm})) : Msg.of("engine_quant.q.wRow", {n:nm}), pass(5*N*K, wq, pkA)/nm, "w"]);
    if(R.tok && !o.tokFused) q.push([Msg.of("engine_quant.q.tokScale"), pass(2*M*K+2*M*N, 2*M*K*2 + 2*M*N*2, pkA), "tok"]);
  }
  if(bq) q.push([R.rht ? Msg.of("engine_quant.q.dyRht") : Msg.of("engine_quant.q.dy"), pass(M*N*(5 + (R.rht?32:0)), M*N*(2 + 2*bb), pkA), "dy"]);
  if(R.hp==="deq") q.push([Msg.of("engine_quant.q.deq"), pass(M*K+N*K, M*K*(bf+2) + N*K*(bf+2)/nm, pkA), "deq"]);
  const tq = q.reduce((s,x)=>s+x[1],0);
  // 反向要用到的东西：保存的激活 + 低精度权重缓存
  const saveX = !fq ? M*K*2 : bq ? M*K*bb : R.hp==="deq" ? M*K*bf : M*K*2;
  const wCache = !fq ? 0 : N*K*(bf + (bq && !R.w2d ? bb : 0));
  return { R, fprop, dgrad, wgrad, q, tq, t:fprop+dgrad+wgrad+tq, saveX, wCache, extra:saveX+wCache };
}

/* 训练三遍 GEMM（fprop 一遍、dgrad + wgrad 两遍）只看峰值能快多少：fprop 用格式 f，反向用格式 b */
const idealSpeedup = (hw, f, b) => { const pk = k => peakOf(hw,k)/peakOf(hw,"bf16"); return 3/(1/pk(f)+2/pk(b)); };
/* QAT 的 fake quant：读 BF16 权重、写回量化再反量化的 BF16（N×K 各 2+2 字节），按 micro-batch 摊；至少一个 kernel */
const fakeQuantT = (hw, g, N, K, nMicro) => Math.max(g.kernelUs*1e-6, memT(N*K*(2+2), hw, g))/Math.max(1, nMicro);

return { SCHEMES, ORDER, RECIPES, RORDER, gemm, crossover, poisson, autoAlign, moeLayer, trainGemm, idealSpeedup, fakeQuantT };
})();
