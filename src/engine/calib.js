/* 和实测对照：每个数据集怎样用计算核心复现——筛哪些点、实测是哪个字段、预测怎么算。
   数据点由调用方传进来（界面从数据点库取，测试直接读 data/db.json）；这里不读数据库、不出文字。
   所有实测都是单台 H200 HGX 上的 Qwen3.5-35B-A3B。 */
const Calib = (function(){
const { MODELS, HW, TOPO, derive, compT, memT, peakOf } = Common;

/* 实测的训练设置：torchtitan，CP8 / EP8，不重算；etaA 是实测的 attention 核心效率（273 / 989 TFLOP/s） */
const BASE = Common.freeze({ N:8, fw:"torchtitan", tp:1, pp:1, vpp:1, cp:8, ep:8, etp:1, linCP:"state", prec:"bf16", recompute:"none", cpLB:false,
  dispatcher:"standard", imb:1, etaA:273/989, offload:false, hsdp:1, gradDt:"fp32", reshard:true, hostSyncUs:30, mokF:0.25, workspaceGB:6,
  cpuAdamGBs:30, rcModules:[] });
/* 实测里的 EP 后端 → Par.EP 里的实现 */
const BACKEND = Common.freeze({ local:"standard", nccl_a2a:"standard", deepep_legacy:"deepep", deepep_v2:"deepep" });
const run = (g, o) => Train.run(MODELS.qwen35, HW.H200, TOPO.hgx8, Object.assign({}, BASE, o), g);

/* where：能复现的点（数据点库的筛选条件，null = 整个数据集）；meas：实测（秒）；predict：本模型的预测（秒） */
const SETS = {
  cp_attention: { where:{ model:"qwen35", head_chunks:1 }, meas:p=>p.fb_ms/1e3,
    predict:(p,g)=>{ const C=p.cp, r=run(g,{S:p.seq,tok:p.seq/C,Gtok:p.seq*8/C,cp:C,cpType:C>1?p.method:"allgather"}), o=r.par.coreFullOp;
      return (o.fwd.t+o.dgrad.t)/o.layers+(r.par.cpFull?r.par.docs*r.par.cpFull.exposed:0); } },
  cp_full_model: { where:{ model:"qwen35" }, meas:p=>p.fb_s,
    predict:(p,g)=>{ const r=run(g,{S:p.seq,tok:p.seq/p.cp,Gtok:p.seq*8/p.cp,cp:p.cp,cpType:p.method,recompute:"full",ep:p.cp}); return r.stepT-r.tOpt; } },
  efsdp_full_model: { where:null, meas:p=>p.step_s,
    predict:(p,g)=>{ const r=run(g,{S:p.seq,tok:p.seq/8,Gtok:p.seq,cpType:"allgather",recompute:"full",ep:p.ep,dispatcher:BACKEND[p.backend]}); return r.stepT-r.tOpt; } },
  efsdp_comm: { where:null, meas:p=>p.s,
    predict:(p,g)=>{ const m=MODELS.qwen35, hw=HW.H200, n=p.seq/8, De=derive(m).De;
      const o=Par.epLayer("deepep",{n,K:8,Z:256,E:p.ep,De,hw,topo:TOPO.hgx8,g,ac:false,prec:"bf16",imb:1,syncUs:30});
      return o.tw+o.tAlpha+o.tSync+memT(4*n*8*De*2, hw, g); } },
  h200_kernels: { where:null, meas:p=>p.ms/1e3,
    predict:(p,g)=>{ const hw=HW.H200; return p.op==="bf16_gemm" ? compT(2*p.m*p.n*p.k, peakOf(hw,"bf16"), g) : memT(p.rw_bytes, hw, g); } } };

/* 公开数据的 replay：同一层 MoE 分别用 DeepEP 和 MoK 算一遍，返回 DeepEP / MoK 的时间比 */
function mokRatio(R, g){
  const h=HW[R.hw], tF=compT(2*R.n*R.K*3*R.De*R.F, peakOf(h,"fp8"), g);
  const mk=be=>Par.epLayer(be,{n:R.n,K:R.K,Z:R.Z,E:R.E,De:R.De,hw:h,topo:TOPO[R.topo],g,ac:false,prec:R.prec,imb:1.1,tExpF:tF*1.1,tExpB:2*tF*1.1,syncUs:60,mokF:0.25}).t;
  return mk("deepep")/mk("mok");
}

return { BASE, BACKEND, SETS, mokRatio };
})();
