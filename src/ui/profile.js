/* ================= 其他 · Profile（other/profile）：当前推理配置的一步按线上 trace 的分法拆开、DP / EP 负载不均、观测本身的开销、工具对照 =================
   数都来自 Profile（src/engine/profile.js），它只拆 Infer.step 的结果；开销的"每个节点多少 µs"取 profile_public 里的实测行 */
const PF = {
  pub: () => DB.set("profile_public"),
  row: id => DB.set("profile_public").find(p => p.id === "profile_public:"+id),
  col: { moe:"c-moe", eplb:"c-oth", attn:"c-attn", proj:"c-cp", idx:"s4", mhc:"s6", lin:"c-lin", mlp:"c-fsdp", other:"muted", draft:"c-lm", comm:"c-a2a", wait:"c-dp", idle:"c-free" },
  link: p => p.url ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.src)+'</a>' : esc(p.src)
};

function renderProfile(){
  const { m, hw, topo, g } = ctx(), c = S.inf, sp = Profile.split(m, hw, topo, c, g, S.prof);
  markInert("prof", Profile.inert(m, c));
  const share = k => sp.kern[k] ? sp.kern[k].frac : 0;
  $("pf-lede").innerHTML = t("profile.b1.lede", { model:esc(m.short||m.name), hw:esc(String(hw.name)), phase:t("profile.phase."+c.phase), n:c.N, dp:sp.r.dpA, step:V(fT(sp.total)) });
  kpis($("pf-kpis"), [
    [t("profile.kpi.step"), fT(sp.total), sp.total>sp.stepT ? t("profile.kpi.stepFoot", { t:fT(sp.stepT) }) : ""],
    [t("profile.kpi.compute"), pct(share("compute"), 1), t("profile.kpi.computeFoot")],
    [t("profile.kpi.comm"), pct(share("comm"), 1), t("profile.kpi.commFoot")],
    [t("profile.kpi.idle"), pct(share("idle"), 1), t("profile.kpi.idleFoot")],
    [t("profile.kpi.kernels"), fInt(sp.nKern), sp.graph ? t("profile.kpi.kernelsGraph") : t("profile.kpi.kernelsEager")]]);
  stackBar($("pf-bar"), $("pf-barlist"), sp.cats.map(x => ({ label:t("profile.cat."+x.key), v:x.t, color:css(PF.col[x.key]), tag:pill(x.kern==="idle" ? "launch" : x.kern) })), sp.total, fT);
  $("pf-read").innerHTML = ul([t("profile.b1.readKern"), t("profile.b1.readBlog"), t("profile.b1.readTrace")]);

  // ---- DP / EP 负载不均 ----
  const dpOn = sp.dp, epOn = sp.ep;
  kpis($("pf-lb-kpis"), [
    [t("profile.lb.dp"), dpOn ? pct(sp.dpLoss, 1) : "—", dpOn ? t("profile.lb.dpFoot", { s:S.prof.dpSkew }) : t("profile.lb.dpOff")],
    [t("profile.lb.ep"), epOn ? pct(sp.eplbLoss, 1) : "—", epOn ? t("profile.lb.epFoot", { imb:c.imb }) : t("profile.lb.epOff")]]);
  const ex = PF.row("hes-dplb");
  $("pf-lb-read").innerHTML = ul([t("profile.lb.readDp"), t("profile.lb.readEp"), ex ? t("profile.lb.readBlog", { src:PF.link(ex) }) : ""]);

  // ---- 观测本身的开销 ----
  const tools = [["hes", PF.row("hes-node-cost")], ["cupti", PF.row("cupti-node-cost")]].filter(([, p]) => p && p.usPerNode>0);
  table($("pf-obs"), [t("profile.obs.head.tool"), t("profile.obs.head.us"), t("profile.obs.head.nodes"), t("profile.obs.head.t"), t("profile.obs.head.frac"), t("profile.obs.head.src")],
    tools.map(([k, p]) => { const o = Profile.observer(sp, p.usPerNode);
      return [t("profile.obs.tool."+k), p.usPerNode.toFixed(1)+" µs", fInt(o.nodes), fT(o.t), pct(o.frac, 1), PF.link(p)]; }), [1, 2, 3, 4]);
  const e1 = PF.row("hes-e2e-small"), e2 = PF.row("hes-e2e-dsv41f");
  $("pf-obs-read").innerHTML = ul([t("profile.obs.readBound"), e1 ? t("profile.obs.readSmall", { v:esc(e1.value), src:PF.link(e1) }) : "", e2 ? t("profile.obs.readBig", { v:esc(e2.value), src:PF.link(e2) }) : ""]);

  // ---- 工具和公开数字 ----
  table($("pf-tools"), [t("profile.tools.head.what"), t("profile.tools.head.value"), t("profile.tools.head.src")],
    PF.pub().map(p => [esc(p.what)+(p.platform ? '<br><span class="tag">'+esc(p.platform)+'</span>' : ''),
      esc(p.value)+(p.caveat ? '<br><span style="color:var(--warn)">'+esc(p.caveat)+'</span>' : ''), PF.link(p)+'<br><span class="tag">'+esc(p.date)+'</span>']));
}
