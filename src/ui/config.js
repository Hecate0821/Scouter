/* ---------- 基础配置 ---------- */
/* 默认值属于计算核心：效率系数 Common.G0、推理 / 训练的基础配置 Infer.BASE / Train.BASE */
const GLOB = Common.G0, BASE_INF = Infer.BASE, BASE_TRN = Train.BASE;

const MOEPAR = {
  sglang:[["tp",t("config.moePar.sglang.tp")],["replicated",t("config.moePar.sglang.replicated")],["deepep",t("config.moePar.sglang.deepep")],["fused",t("config.moePar.sglang.fused")]],
  vllm:[["tp",t("config.moePar.vllm.tp")],["ag",t("config.moePar.vllm.ag")],["deepep",t("config.moePar.vllm.deepep")],["fused",t("config.moePar.vllm.fused")]],
  tokenspeed:[["replicated",t("config.moePar.tokenspeed.replicated")],["tp",t("config.moePar.tokenspeed.tp")],["deepep",t("config.moePar.tokenspeed.deepep")],["fused",t("config.moePar.tokenspeed.fused")]],
  titan:[["standard",t("config.moePar.titan.standard")],["fused",t("config.moePar.titan.fused")],["tp",t("config.moePar.titan.tp")]] };
const MOENAME = { tp:t("config.moeName.tp"), ag:t("config.moeName.ag"), replicated:t("config.moeName.replicated"), deepep:t("config.moeName.deepep"), fused:t("config.moeName.fused"), standard:t("config.moeName.standard") };
function compatMoe(eng, mp){ const ok=MOEPAR[eng].map(x=>x[0]); if(ok.includes(mp)) return mp;
  if(mp==="replicated"||mp==="ag"||mp==="standard") return Infer.ENGINES[eng].defaultMoe;
  if(mp==="deepep") return eng==="titan"?"standard":"deepep"; return ok[0]; }

const blackwell = () => Common.sm10x(HW[S.hw]);   // 预设里的 Blackwell 分支用的是 SM100 的 kernel（FP4 KV、Dist-MoE、trtllm-gen），SM120 走另一支
const hwOK = (h,t) => Common.homeTopo(h)===t;
const prefHW = (mk,t) => t==="pcie8" ? "RTXP6000" : /^nvl/.test(t) ? (mk==="kimik3"||mk==="musespark"||mk==="dsv41f"?"GB300":"GB200") : (mk==="kimik3"?"B300":mk==="glm53f"||mk==="musespark"||mk==="dsv41f"?"B200":"H200");

/* ---------- 推理预设：引擎 × 模型 × 拓扑 ---------- */
function infPresets(eng, mk, topoKey){
  const big=TOPO[topoKey].gpus>8, G=TOPO[topoKey].gpus, P=[], add=(id,label,src,cfg)=>P.push({id,label,src,cfg});
  const m=MODELS[mk], f8 = m.wDefault==="fp8";
  /* 模型配置里写了 dense / KV 的默认精度（dDefault / kvDefault）就用它，否则按权重是不是 FP8 推 */
  const X = o => Object.assign({ wDt:m.wDefault, dDt:m.dDefault||(f8?"fp8":"bf16"), kvDt:m.kvDefault||(f8?"fp8":"bf16"), stateDt:"fp32" }, o);
  const hk=S.hw, bw=blackwell();
  if(eng==="sglang"){
    if(mk==="qwen35"){
      add("rec",t("config.inf.sglang.qwen35.rec.label",{n:G}),t("config.inf.sglang.qwen35.rec.src"),X({N:1,dpA:1,moePar:"tp",B:128}));
      add("mtp",t("config.inf.sglang.qwen35.mtp.label"),t("config.inf.sglang.qwen35.mtp.src"),X({N:1,dpA:1,moePar:"tp",B:48,specK:3,accept:1.8}));
      add("dep",t("config.inf.sglang.qwen35.dep.label"),t("config.inf.sglang.qwen35.dep.src"),X({N:8,dpA:8,moePar:"deepep",B:64}));
    } else if(mk==="kimik3"){
      if(!big){
        add("rec",t("config.inf.sglang.kimik3.hgx.rec.label"),t("config.inf.sglang.kimik3.hgx.rec.src"),X({N:8,dpA:1,moePar:"tp",B:32}));
        add("dcp",t("config.inf.sglang.kimik3.hgx.dcp.label"),t("config.inf.sglang.kimik3.hgx.dcp.src"),X({N:8,dpA:1,moePar:"tp",B:96,dcp:8}));
        add("spec",t("config.inf.sglang.kimik3.hgx.spec.label"),t("config.inf.sglang.kimik3.hgx.spec.src"),X({N:8,dpA:1,moePar:"tp",B:32,specK:7,specMethod:"dspark",accept:1.7}));
      } else {
        add("rec",t("config.inf.sglang.kimik3.nvl.rec.label"),t("config.inf.sglang.kimik3.nvl.rec.src"),X({N:64,dpA:8,moePar:"fused",B:64,kvDt:"fp8",stateDt:"bf16"}));
        add("dp64",t("config.inf.sglang.kimik3.nvl.dp64.label"),t("config.inf.sglang.kimik3.nvl.dp64.src"),X({N:64,dpA:64,moePar:"fused",B:32,kvDt:"fp8",stateDt:"bf16"}));
        add("tp8",t("config.inf.sglang.kimik3.nvl.tp8.label"),t("config.inf.sglang.kimik3.nvl.tp8.src"),X({N:8,dpA:1,moePar:"tp",B:32}));
      }
    } else if(mk==="glm53f"){
      if(!big){
        add("rec",t("config.inf.sglang.glm53f.hgx.rec.label"),t("config.inf.sglang.glm53f.hgx.rec.src"),X({N:8,dpA:1,moePar:"tp",B:64}));
        add("mtp",t("config.inf.sglang.glm53f.hgx.mtp.label"),t("config.inf.sglang.glm53f.hgx.mtp.src"),X({N:8,dpA:1,moePar:"tp",B:48,specK:5,accept:1.6}));
        add("dep",t("config.inf.sglang.glm53f.hgx.dep.label"),t("config.inf.sglang.glm53f.hgx.dep.src"),X({N:8,dpA:8,moePar:"deepep",B:64}));
      } else {
        add("rec",t("config.inf.sglang.glm53f.nvl.rec.label"),t("config.inf.sglang.glm53f.nvl.rec.src"),X({N:4,dpA:1,moePar:"tp",B:64}));
        add("dep",t("config.inf.sglang.glm53f.nvl.dep.label"),t("config.inf.sglang.glm53f.nvl.dep.src"),X({N:64,dpA:64,moePar:"deepep",B:32}));
      }
    } else if(mk==="musespark"){
      if(!big){
        add("rec",t("config.inf.sglang.musespark.hgx.rec.label"),t("config.inf.sglang.musespark.hgx.rec.src"),X({N:4,dpA:1,moePar:"replicated",B:64}));
        add("bf16",t("config.inf.sglang.musespark.hgx.bf16.label"),t("config.inf.sglang.musespark.hgx.bf16.src"),X({N:8,dpA:1,moePar:"replicated",B:32,wDt:"bf16"}));
      } else {
        add("rec",t("config.inf.sglang.musespark.nvl.rec.label"),t("config.inf.sglang.musespark.nvl.rec.src"),X({N:8,dpA:1,moePar:"replicated",B:32,kvDt:"fp8"}));
        add("tp4",t("config.inf.sglang.musespark.nvl.tp4.label"),t("config.inf.sglang.musespark.nvl.tp4.src"),X({N:4,dpA:1,moePar:"replicated",B:64}));
      }
    } else if(mk==="dsv41f"){
      /* cookbook 单节点：B200 / B300 / GB300 是 TP4 + EP4，H200 是 TP8 + EP8；KV 布局按 SGLang 的默认（Blackwell NVFP4 压缩 KV，Hopper 回落 V4 的 FP8 布局） */
      const n=bw?4:8, kv=bw?"nvfp4":"fp8";
      add("ht",t("config.inf.sglang.dsv41f.ht.label",{n}),t(bw?"config.inf.sglang.dsv41f.ht.src":"config.inf.sglang.dsv41f.ht.srcHopper"),X({N:n,dpA:1,moePar:"replicated",B:bw?256:64,kvDt:kv,memFrac:bw?0.85:0.8}));
      if(bw) add("ll",t("config.inf.sglang.dsv41f.ll.label",{n}),t("config.inf.sglang.dsv41f.ll.src"),X({N:n,dpA:1,moePar:"replicated",B:48,kvDt:kv,memFrac:0.8,specK:5,specMethod:"dspark",accept:2}));
      add("full",t("config.inf.sglang.dsv41f.full.label"),t("config.inf.sglang.dsv41f.full.src"),X({N:n,dpA:1,moePar:"replicated",B:bw?256:64,kvDt:kv,memFrac:bw?0.85:0.8,replay:false}));
    } else {
      add("rec",t("config.inf.sglang.qwen38_27b.rec.label"),t("config.inf.sglang.qwen38_27b.rec.src"),X({N:1,dpA:1,moePar:"tp",B:96,kvDt:"fp8"}));
      add("mtp",t("config.inf.sglang.qwen38_27b.mtp.label"),t("config.inf.sglang.qwen38_27b.mtp.src"),X({N:1,dpA:1,moePar:"tp",B:48,kvDt:"fp8",specK:3,accept:1.8}));
      add("tp2",t("config.inf.sglang.qwen38_27b.tp2.label"),t("config.inf.sglang.qwen38_27b.tp2.src"),X({N:2,dpA:1,moePar:"tp",B:128,kvDt:"fp8"}));
    }
  } else if(eng==="vllm"){
    if(mk==="qwen35"){
      add("rec",t("config.inf.vllm.qwen35.rec.label"),t("config.inf.vllm.qwen35.rec.src"),X({N:1,dpA:1,moePar:"tp",B:128}));
      add("mtp",t("config.inf.vllm.qwen35.mtp.label"),t("config.inf.vllm.qwen35.mtp.src"),X({N:1,dpA:1,moePar:"tp",B:96,specK:3,accept:1.8}));
      add("dep",t("config.inf.vllm.qwen35.dep.label"),t("config.inf.vllm.qwen35.dep.src"),X({N:8,dpA:8,moePar:"ag",B:64}));
    } else if(mk==="kimik3"){
      if(!big){
        add("rec",t("config.inf.vllm.kimik3.hgx.rec.label"),t("config.inf.vllm.kimik3.hgx.rec.src"),X({N:8,dpA:1,moePar:"tp",B:32,kvDt:"fp8"}));
        add("dep",t("config.inf.vllm.kimik3.hgx.dep.label"),t("config.inf.vllm.kimik3.hgx.dep.src"),X({N:8,dpA:8,moePar:"ag",B:32,kvDt:"fp8"}));
      } else {
        add("rec",t("config.inf.vllm.kimik3.nvl.rec.label"),t("config.inf.vllm.kimik3.nvl.rec.src"),X({N:64,dpA:64,moePar:"deepep",B:32,kvDt:"fp8"}));
        add("tp8",t("config.inf.vllm.kimik3.nvl.tp8.label"),t("config.inf.vllm.kimik3.nvl.tp8.src"),X({N:8,dpA:1,moePar:"tp",B:32,kvDt:"fp8"}));
      }
    } else if(mk==="glm53f"){
      if(!big) add("rec",t("config.inf.vllm.glm53f.hgx.rec.label"),t("config.inf.vllm.glm53f.hgx.rec.src"),X({N:8,dpA:1,moePar:"tp",B:64}));
      else add("rec",t("config.inf.vllm.glm53f.nvl.rec.label"),t("config.inf.vllm.glm53f.nvl.rec.src"),X({N:4,dpA:1,moePar:"tp",B:64}));
      add("dep",t("config.inf.vllm.glm53f.dep.label",{n:big?64:8}),t("config.inf.vllm.glm53f.dep.src"),X({N:big?64:8,dpA:big?64:8,moePar:"ag",B:64}));
    } else if(mk==="musespark"){
      add("rec",t("config.inf.vllm.musespark.rec.label"),t("config.inf.vllm.musespark.rec.src"),X({N:8,dpA:1,moePar:"ag",B:64}));
      add("dep",t("config.inf.vllm.musespark.dep.label"),t("config.inf.vllm.musespark.dep.src"),X({N:8,dpA:8,moePar:"ag",B:32}));
      if(big) add("tp8",t("config.inf.vllm.musespark.tp8.label"),t("config.inf.vllm.musespark.tp8.src"),X({N:8,dpA:1,moePar:"ag",B:64}));
    } else if(mk==="dsv41f"){
      /* recipe single_node_tp：tp 默认 4、H100 8、Blackwell 2；Blackwell 显式 --kv-cache-dtype fp8；NVIDIA 各卡都把 Engram 表放主机内存（cpu_offload）。
         max-num-seqs：Blackwell 128，B200 8，B300 256 */
      /* --gpu-memory-utilization：vLLM 默认 0.9，recipe 在 H100 上调到 0.92 */
      const n=hk==="H100"?8:bw?2:4, Bt=hk==="B200"?8:hk==="B300"?256:128, mf=hk==="H100"?0.92:0.9;
      add("rec",t("config.inf.vllm.dsv41f.rec.label",{n}),t("config.inf.vllm.dsv41f.rec.src"),X({N:n,dpA:1,moePar:"tp",B:Bt,kvDt:"fp8",engramAt:"host",memFrac:mf}));
      add("spec",t("config.inf.vllm.dsv41f.spec.label",{n}),t("config.inf.vllm.dsv41f.spec.src"),X({N:n,dpA:1,moePar:"tp",B:Math.min(Bt,64),kvDt:"fp8",engramAt:"host",memFrac:mf,specK:5,specMethod:"dspark",accept:2}));
      /* single_node_dep：Blackwell 用 deep_gemm_mega_moe（a2a 和 expert GEMM 融合，这里按 fused 算），Engram embedding_across_dp（表按 DP rank 切开放显存） */
      add("dep",t("config.inf.vllm.dsv41f.dep.label"),t("config.inf.vllm.dsv41f.dep.src"),X({N:8,dpA:8,moePar:bw?"fused":"ag",B:hk==="B200"?32:bw?128:64,kvDt:"fp8",engramAt:"gpu",memFrac:0.9}));
    } else {
      add("rec",t("config.inf.vllm.qwen38_27b.rec.label"),t("config.inf.vllm.qwen38_27b.rec.src"),X({N:1,dpA:1,moePar:"tp",B:128}));
      add("mtp",t("config.inf.vllm.qwen38_27b.mtp.label"),t("config.inf.vllm.qwen38_27b.mtp.src"),X({N:1,dpA:1,moePar:"tp",B:96,specK:3,accept:1.8}));
    }
  } else if(eng==="tokenspeed"){
    if(mk==="kimik3"){
      add("rec",t("config.inf.tokenspeed.kimik3.rec.label"),t("config.inf.tokenspeed.kimik3.rec.src"),X({N:8,dpA:1,moePar:"replicated",B:32,kvDt:"fp8"}));
      add("spec",t("config.inf.tokenspeed.kimik3.spec.label"),t("config.inf.tokenspeed.kimik3.spec.src"),X({N:8,dpA:1,moePar:"replicated",B:32,kvDt:"fp8",specK:3,accept:1.7}));
      if(big) add("adp",t("config.inf.tokenspeed.kimik3.adp.label"),t("config.inf.tokenspeed.kimik3.adp.src"),X({N:8,dpA:8,moePar:"fused",B:32,kvDt:"fp8"}));
    } else if(mk==="glm53f"){
      add("rec",t("config.inf.tokenspeed.glm53f.rec.label"),t("config.inf.tokenspeed.glm53f.rec.src"),X({N:8,dpA:1,moePar:"replicated",B:64}));
      add("spec",t("config.inf.tokenspeed.glm53f.spec.label"),t("config.inf.tokenspeed.glm53f.spec.src"),X({N:8,dpA:1,moePar:"replicated",B:64,specK:3,accept:1.6}));
    } else if(mk==="musespark"){
      add("rec",t("config.inf.tokenspeed.musespark.rec.label"),t("config.inf.tokenspeed.musespark.rec.src"),X({N:4,dpA:1,moePar:"replicated",B:64}));
    } else if(mk==="dsv41f"){
      /* TokenSpeed 没有 DeepSeek-V4.1 的 recipe，只留自定义 */
    } else {
      const n2 = mk==="qwen38_27b" ? 1 : 2;
      add("rec",mk==="qwen38_27b"?t("config.inf.tokenspeed.qwen38_27b.rec.label",{n:n2}):t("config.inf.tokenspeed.qwen35.rec.label",{n:n2}),
        mk==="qwen38_27b"?t("config.inf.tokenspeed.qwen38_27b.rec.src"):t("config.inf.tokenspeed.qwen35.rec.src"),X({N:n2,dpA:1,moePar:"replicated",B:128}));
      add("spec",t("config.inf.tokenspeed.qwen.spec.label",{n:n2}),t("config.inf.tokenspeed.qwen.spec.src"),X({N:n2,dpA:1,moePar:"replicated",B:80,specK:3,accept:1.8}));
      if(n2>1) add("tp1",t("config.inf.tokenspeed.qwen35.tp1.label"),t("config.inf.tokenspeed.qwen35.tp1.src"),X({N:1,dpA:1,moePar:"replicated",B:128}));
    }
  } else {
    const T = o => Object.assign({ wDt:"bf16", dDt:"bf16", kvDt:"bf16", stateDt:"fp32", moePar: blackwell()?"fused":"standard", B:64 }, o);
    const st = { qwen35:t("config.inf.titan.status.qwen35"), qwen38_27b:t("config.inf.titan.status.qwen38_27b"), musespark:t("config.inf.titan.status.musespark"),
      kimik3:t("config.inf.titan.status.kimik3"), glm53f:t("config.inf.titan.status.glm53f"), dsv41f:t("config.inf.titan.status.dsv41f") }[mk];
    const be = blackwell() ? t("config.inf.titan.ep.blackwell") : t("config.inf.titan.ep.hopper");
    add("rec", big?t("config.inf.titan.nvl.rec.label"):t("config.inf.titan.hgx.rec.label"),
      t("config.inf.titan.rec.src",{status:st, shape:big?t("config.inf.titan.nvl.rec.shape"):t("config.inf.titan.hgx.rec.shape"), ep:be}),
      T(big?{N:16,dpA:4}:{N:4,dpA:4}));
    add("tp1",t("config.inf.titan.tp1.label"),t("config.inf.titan.tp1.src"),T({N:1,dpA:1,moePar:"tp",B:128}));
  }
  P.push({id:"custom",label:t("config.inf.custom"),src:"",cfg:null});
  return P;
}

/* ---------- 训练预设：框架 × 模型 × 拓扑 ---------- */
function trnPresets(fw, mk, topoKey){
  const big=TOPO[topoKey].gpus>8, P=[], add=(id,label,src,cfg,need)=>P.push({id,label,src,cfg,need});
  if(fw==="megatron"){
    if(mk==="qwen38_27b"){
      add("miles",t("config.trn.megatron.qwen38_27b.miles.label"),t("config.trn.megatron.qwen38_27b.miles.src"),{tp:4,cp:1,ep:1,tok:8192});
    } else if(mk==="musespark"){
      add("miles",t("config.trn.megatron.musespark.miles.label"),t("config.trn.megatron.musespark.miles.src"),{tp:8,pp:4,cp:1,ep:8,tok:8192},{topo:"nvl64",hw:"GB300"});
    } else if(mk==="qwen35"){
      add("miles",t("config.trn.megatron.qwen35.miles.label"),t("config.trn.megatron.qwen35.miles.src"),{tp:2,cp:2,ep:8,tok:8192});
      add("miles1",t("config.trn.megatron.qwen35.miles1.label"),t("config.trn.megatron.qwen35.miles1.src"),{tp:1,cp:1,ep:8,tok:8192});
    } else if(mk==="glm53f"){
      add("slime",t("config.trn.megatron.glm53f.slime.label"),t("config.trn.megatron.glm53f.slime.src"),{tp:1,pp:4,cp:8,ep:16,tok:2048,cpType:"allgather"},{topo:"nvl64"});
      add("miles",t("config.trn.megatron.glm53f.miles.label"),t("config.trn.megatron.glm53f.miles.src"),{tp:8,pp:4,cp:1,ep:16,tok:4096,offload:false},{topo:"nvl64"});
    } else if(mk==="kimik3"){
      add("miles",t("config.trn.megatron.kimik3.miles.label"),t("config.trn.megatron.kimik3.miles.src"),{tp:4,pp:8,cp:2,ep:8,tok:8192},{topo:"nvl64",hw:"GB300"});
    }
    const dense=!MODELS[mk].moe.E;
    add("generic",dense?t("config.trn.megatron.generic.dense.label"):big?t("config.trn.megatron.generic.nvl.label"):t("config.trn.megatron.generic.hgx.label"),t("config.trn.megatron.generic.src"),dense?{tp:2,cp:1,ep:1,tok:8192}:big?{tp:4,cp:2,ep:16,tok:8192}:{tp:2,cp:1,ep:8,tok:8192});
  } else {
    const bw = blackwell(), disp = bw?"distmoe":"standard";
    const st = { kimik3:t("config.trn.torchtitan.status.kimik3"), glm53f:t("config.trn.torchtitan.status.glm53f"), musespark:t("config.trn.torchtitan.status.musespark"), qwen38_27b:t("config.trn.torchtitan.status.qwen38_27b"), dsv41f:t("config.trn.torchtitan.status.dsv41f") }[mk]||"";
    const isMoE=!!MODELS[mk].moe.E;
    if(!big) add("dapo",t(isMoE?"config.trn.torchtitan.dapo.label":"config.trn.torchtitan.dapo.labelDense"),t(isMoE?"config.trn.torchtitan.dapo.src":"config.trn.torchtitan.dapo.srcDense",{ep:bw?t("config.trn.torchtitan.dapo.epBlackwell"):t("config.trn.torchtitan.dapo.epHopper"), status:st}),
      {tp:1,cp:1,ep:isMoE?4:1,tok:8192,dispatcher:disp,cpType:"allgather",linCP:"state",recompute:"default",offload:false});
    else add("big",t(isMoE?"config.trn.torchtitan.big.label":"config.trn.torchtitan.big.labelDense"),t("config.trn.torchtitan.big.src",{status:st}),
      {tp:4,cp:4,ep:isMoE?16:1,tok:16384,dispatcher:disp,cpType:"ulysses",linCP:"ulysses",recompute:"default",offload:false});
    if(MODELS[mk].moe.E) add("ep8",t("config.trn.torchtitan.ep8.label"),t("config.trn.torchtitan.ep8.src"),{tp:1,cp:1,ep:8,tok:16384,dispatcher:disp,cpType:"allgather",linCP:"state",recompute:"full",offload:false});
    if(mk==="qwen35") add("h200cp8",t("config.trn.torchtitan.qwen35.h200cp8.label"),t("config.trn.torchtitan.qwen35.h200cp8.src"),
      {tp:1,cp:8,ep:4,tok:32768,S:262144,Gtok:262144,dispatcher:"standard",cpType:"allgather",cpLB:false,linCP:"state",recompute:"full",offload:false},{topo:"hgx8",hw:"H200"});
    if(mk==="qwen38_27b") add("uly256",t("config.trn.torchtitan.qwen38_27b.uly256.label"),t("config.trn.torchtitan.qwen38_27b.uly256.src"),
      {tp:1,cp:8,ep:1,tok:32768,S:262144,Gtok:262144,cpType:"ulysses",linCP:"ulysses",recompute:"full",offload:false});
    if(mk==="kimik3") add("mok64",t("config.trn.torchtitan.kimik3.mok64.label"),t("config.trn.torchtitan.kimik3.mok64.src"),
      {tp:1,cp:4,ep:64,tok:16384,S:65536,Gtok:2097152,dispatcher:"mok",cpType:"allgather",linCP:"state",prec:"mxfp8",recompute:"full",offload:true},{topo:"nvl64",hw:"GB300"});
    if(mk==="musespark") add("dm64",t("config.trn.torchtitan.musespark.dm64.label"),t("config.trn.torchtitan.musespark.dm64.src"),
      {tp:1,cp:4,ep:16,tok:16384,S:65536,Gtok:4194304,dispatcher:"distmoe",cpType:"allgather",linCP:"state",recompute:"full",offload:false},{topo:"nvl64",hw:"GB300"});
  }
  P.push({id:"custom",label:t("config.trn.custom"),src:"",cfg:null});
  return P;
}
