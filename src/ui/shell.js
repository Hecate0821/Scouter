/* ---------- 外壳：侧栏、顶栏（模型 / GPU / 拓扑 / 口径）、主题、命令行与 JSON、地址里的状态 ---------- */
/* 口径：ideal = 硬件屋顶，modeled = 默认效率，measured = 只看实测数据点。前两个换的是计算用的效率参数，
   第三个只换显示：页面里标了 data-src="measured" 的部分是实测，data-src="both" 是实测和模型对照，其余都是计算。 */
const SRC = Scouter.SOURCES;
S.src = "modeled"; S.globModeled = S.glob;

function setSource(src, quiet){
  if(!SRC.includes(src)) src = "modeled";
  S.src = src; S.glob = src === "ideal" ? clone(Common.IDEAL) : S.globModeled;
  document.body.dataset.src = src;
  qa("#g-src button").forEach(b => b.setAttribute("aria-checked", b.dataset.src === src ? "true" : "false"));
  qa("#g-adv input").forEach(el => { el.disabled = src === "ideal"; });
  syncInputs();
  if(!quiet){ saveUrl(); render(); applySource(); }
}
/* 当前页里按口径显示 / 隐藏；整页都被藏掉时给一个空状态，指向另一种口径或有实测的页 */
const paneOf = key => document.querySelector('section[data-pane="' + key + '"]');
const hasMeasured = key => { const p = paneOf(key); return !!(p && p.querySelector('[data-src="measured"],[data-src="both"]')); };
const modelOnly = key => { const p = paneOf(key); return !!p && Array.from(p.querySelectorAll(".block")).every(b => b.dataset.src === "measured"); };
function applySource(){
  const meas = S.src === "measured", key = viewKey(), pane = paneOf(key);
  qa("main [data-src]").forEach(el => el.classList.toggle("src-off", el.dataset.src === "measured" && !meas));
  qa("main .block").forEach(b => b.classList.toggle("src-off", meas ? !b.dataset.src && !b.querySelector("[data-src]") : b.dataset.src === "measured"));
  qa("#g-side a[data-route]").forEach(a => a.classList.toggle("dim", meas ? !hasMeasured(a.dataset.route) : modelOnly(a.dataset.route)));
  const empty = $("g-empty"), none = pane && !Array.from(pane.querySelectorAll(".block")).some(b => !b.classList.contains("src-off"));
  empty.hidden = !none;
  if(none){
    const pages = Object.keys(VIEW).filter(hasMeasured).map(k => '<a href="#' + k + '">' + esc(navLabel(k)) + '</a>').join(" · ");
    empty.innerHTML = meas
      ? '<h2>' + esc(t("shell.empty.noMeasured.h")) + '</h2><p>' + esc(t("shell.empty.noMeasured.p")) + '</p><p>' + pages + '</p>' +
        '<p><button class="act" type="button" data-set-src="modeled">' + esc(t("shell.empty.toModeled")) + '</button></p>'
      : '<h2>' + esc(t("shell.empty.onlyMeasured.h")) + '</h2><p>' + esc(t("shell.empty.onlyMeasured.p")) + '</p>' +
        '<p><button class="act" type="button" data-set-src="measured">' + esc(t("shell.empty.toMeasured")) + '</button></p>';
  }
  $("g-srcnote").textContent = t("shell.src.note." + S.src);
}
document.addEventListener("click", e => { const b = e.target.closest("[data-set-src]"); if(b) setSource(b.dataset.setSrc); });
$("g-src").addEventListener("click", e => { const b = e.target.closest("button[data-src]"); if(b && b.dataset.src !== S.src) setSource(b.dataset.src); });
$("g-src").addEventListener("keydown", e => {      // 方向键在三个口径之间移动（radiogroup 的约定）
  const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0; if(!d) return;
  e.preventDefault(); const i = (SRC.indexOf(S.src) + d + SRC.length) % SRC.length; setSource(SRC[i]);
  const b = $("g-src").querySelector('[data-src="' + SRC[i] + '"]'); if(b && b.focus) b.focus(); });

/* ---------- 侧栏与页头 ---------- */
const navLink = key => document.querySelector('#g-side a[data-route="' + key + '"]');
const navLabel = key => { const a = navLink(key); return a ? a.textContent : key; };
function pageHead(key){
  qa("#g-side a[data-route]").forEach(a => a.setAttribute("aria-current", a.dataset.route === key ? "page" : "false"));
  const a = navLink(key), grp = a && a.closest(".grp"), h = grp && grp.querySelector("h4");
  $("g-crumb").textContent = h ? h.textContent : "Scouter";
  $("g-title").textContent = navLabel(key);
  $("g-lede").hidden = key !== "overview";
  document.body.classList.remove("nav-open"); $("g-scrim").hidden = true; $("g-menu").setAttribute("aria-expanded", "false");
}
$("g-find").addEventListener("input", e => {
  const q = e.target.value.trim().toLowerCase();
  qa("#g-side .grp").forEach(g => { let any = false;
    g.querySelectorAll("a[data-route]").forEach(a => { const on = !q || a.textContent.toLowerCase().includes(q) || a.dataset.route.includes(q); a.hidden = !on; any = any || on; });
    g.hidden = !any; });
});
$("g-find").addEventListener("keydown", e => {
  if(e.key === "Enter"){ const a = Array.from(qa("#g-side a[data-route]")).find(x => !x.hidden); if(a){ location.hash = a.getAttribute("href"); e.target.blur(); } }
  else if(e.key === "Escape"){ e.target.value = ""; e.target.dispatchEvent(new Event("input")); e.target.blur(); }
});
document.addEventListener("keydown", e => {        // "/" 跳到侧栏搜索；在输入框里不抢
  if(e.key !== "/" || e.metaKey || e.ctrlKey || /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement && document.activeElement.tagName)) return;
  e.preventDefault(); document.body.classList.add("nav-open"); $("g-find").focus();
});
const setNav = open => { document.body.classList.toggle("nav-open", open); $("g-scrim").hidden = !open; $("g-menu").setAttribute("aria-expanded", String(open)); };
$("g-menu").addEventListener("click", () => setNav(!document.body.classList.contains("nav-open")));
$("g-scrim").addEventListener("click", () => setNav(false));

/* ---------- 主题：和主页共用 localStorage 的 theme ---------- */
function themeLabel(){ const th = document.documentElement.getAttribute("data-theme") || "dark"; $("g-theme").textContent = t(th === "dark" ? "shell.theme.toLight" : "shell.theme.toDark"); }
$("g-theme").addEventListener("click", () => {
  const th = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", th); try{ localStorage.setItem("theme", th); }catch(e){}
  themeLabel(); render();        // 图里的颜色是渲染时从 CSS 变量取的
});

/* ---------- 这一页对应的数据平面调用：命令行和 JSON 用同一份 ---------- */
const diffOf = (o, base) => Object.fromEntries(Object.keys(o).filter(k => JSON.stringify(o[k]) !== JSON.stringify(base[k])).map(k => [k, o[k]]));
function planeCall(){
  const key = viewKey(), q = { model: S.modelKey, hw: S.hw, topo: S.topo };
  if(JSON.stringify(S.model) !== JSON.stringify(MODELS[S.modelKey])) q.model = S.model;       // 改过的模型整个带上
  const basis = S.src === "ideal" ? "ideal" : "modeled", g = diffOf(S.glob, Common.BASIS[basis]);
  const calc = o => Object.assign(q, basis === "ideal" ? { basis } : {}, Object.keys(g).length ? { g } : {}, o);
  if(key === "train/calib") return ["calib", Object.assign({ set: calibSel }, basis === "ideal" ? { basis } : {}, Object.keys(g).length ? { g } : {})];
  if(S.src === "measured" || key === "appendix/data") return ["sets", {}];
  if(key.startsWith("arch/")){ const a = Object.assign({}, S.arch); delete a.mod; return ["module", calc(diffOf(a, Arch.defaults(S.model)))]; }
  if(key.startsWith("train/")) return ["train", calc(diffOf(S.trn, Object.assign({}, Train.BASE, { N: TOPO[S.topo].gpus })))];
  if(key.startsWith("infer/") || key === "overview") return ["infer", calc(diffOf(S.inf, Infer.BASE))];
  return ["roofline", { hw: S.hw, ...(basis === "ideal" ? { basis } : {}) }];
}
const shq = v => { const s = typeof v === "string" ? v : JSON.stringify(v); return /^[\w.,:+-]+$/.test(s) ? s : "'" + s.replace(/'/g, "'\\''") + "'"; };
function cliOf(cmd, q){
  const flat = Object.keys(q).every(k => k === "g" ? Object.values(q.g).every(v => typeof v !== "object") : q[k] === null || typeof q[k] !== "object");
  if(!flat) return "node tools/scouter.js " + cmd + " --json " + shq(q);
  const args = [];
  for(const k of Object.keys(q)){ if(k === "g") for(const gk of Object.keys(q.g)) args.push("--g." + gk + " " + shq(q.g[gk])); else args.push("--" + k + " " + shq(q[k])); }
  return ["node tools/scouter.js", cmd].concat(args).join(" ");
}
let dlgText = "", dlgName = "scouter.json";
function openDlg(title, hint, text, name){
  dlgText = text; dlgName = name;
  $("g-dlg-t").textContent = title; $("g-dlg-hint").textContent = hint; $("g-dlg-pre").textContent = text;
  $("g-dlg").hidden = false; $("g-dlg-x").focus();
}
const closeDlg = () => { $("g-dlg").hidden = true; };
$("g-dlg-x").addEventListener("click", closeDlg);
$("g-dlg").addEventListener("click", e => { if(e.target === $("g-dlg")) closeDlg(); });
document.addEventListener("keydown", e => { if(e.key === "Escape" && !$("g-dlg").hidden) closeDlg(); });
function copyText(text, btn){
  const done = () => { const old = btn.textContent; btn.textContent = t("shell.dlg.copied"); setTimeout(() => { btn.textContent = old; }, 1200); };
  if(navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, () => {}); }
$("g-dlg-copy").addEventListener("click", e => copyText(dlgText, e.currentTarget));
$("g-dlg-dl").addEventListener("click", () => {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([dlgText], { type: "application/json" })); a.download = dlgName;
  document.body.appendChild(a); a.click(); a.remove(); });
$("g-cli").addEventListener("click", () => { const [cmd, q] = planeCall();
  openDlg(t("shell.dlg.cli.t"), t("shell.dlg.cli.hint"), cliOf(cmd, q) + "\n\n" + "Scouter." + cmd + "(" + JSON.stringify(q) + ")", "scouter-" + cmd + ".sh"); });
$("g-json").addEventListener("click", () => { const [cmd, q] = planeCall(); let out;
  try { out = JSON.stringify(Scouter[cmd](q), (k, v) => typeof v === "function" ? undefined : v === Infinity ? "Infinity" : v, 1); }
  catch(e) { out = String(e.message); }
  openDlg(t("shell.dlg.json.t", { cmd }), t("shell.dlg.json.hint"), out, "scouter-" + cmd + ".json"); });
$("g-link").addEventListener("click", e => { saveUrl(); copyText(location.href, e.currentTarget); });

/* ---------- 地址：?model=&hw=&topo=&src= 加上 #页/子页，复制下来别人打开是同一个视图 ---------- */
function saveUrl(){
  try {
    const u = new URL(location.href), p = u.searchParams;
    p.set("model", S.modelKey); p.set("hw", S.hw); p.set("topo", S.topo);
    if(S.src === "modeled") p.delete("src"); else p.set("src", S.src);
    history.replaceState(null, "", u.pathname + "?" + p.toString() + location.hash);
  } catch(e) {}        // file:// 下有的浏览器不让改地址，不影响使用
}
["g-model", "g-hw", "g-topo"].forEach(id => $(id).addEventListener("change", () => { saveUrl(); applySource(); }));
function loadUrl(){
  let p; try { p = new URL(location.href).searchParams; } catch(e) { return; }
  setSource(p.get("src") || "modeled", true);        // 先定口径：下面换模型时会写回地址
  const pickSel = (id, v, table) => { if(v && v in table && $(id).value !== v){ $(id).value = v; $(id).dispatchEvent(new Event("change")); } };
  pickSel("g-model", p.get("model"), MODELS); pickSel("g-topo", p.get("topo"), TOPO); pickSel("g-hw", p.get("hw"), HW);
}
window.Scouter = Scouter;        // 浏览器控制台和驱动浏览器的 agent 直接调数据平面：Scouter.infer({...})
