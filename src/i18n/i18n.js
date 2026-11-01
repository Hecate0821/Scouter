/* 界面文字与代码分开：代码里只写 key，文字在 locales/<语言>/<命名空间>.js 里登记。
   - t("infer.step.title", {n: 3})：取当前语言的文字，{n} 换成参数；当前语言没有这个 key 时退回 zh，再没有就原样返回 key（并记进 missing）。
   - 值可以是函数 p => ...，用于不好用占位符表达的语法（复数、条件）；能用占位符就别用函数。
   - HTML 里的静态文字用属性标出来：data-i18n（纯文本）、data-i18n-html（带标签）、data-i18n-attr="aria-label=key;title=key2"，I18N.apply() 填进去。
   - 换语言 = 带 ?lang=<代码> 重新加载；选择记在 localStorage。zh 的文件由 index.html 直接加载，其他语言在这里按 FILES 追加。
   加新语言：在 LOCALES 里登记，复制 locales/zh/ 为 locales/<代码>/ 后逐条翻译；漏翻的 key 会退回 zh。详见 docs/i18n.md。 */
const I18N = (function(){
const FALLBACK = "zh";
const LOCALES = { zh: { name: "中文", html: "zh-CN" }, en: { name: "English", html: "en" } };
/* 每种语言都按这个顺序加载同名文件；zh 的 <script> 写在 index.html 里 */
const FILES = ["common", "shell", "overview", "arch", "train", "par", "calib", "comm", "infer", "tradeoff", "spec", "config",
  "quant", "rl", "appendix", "releases", "engine_common", "engine_infer", "engine_train", "engine_par", "engine_quant", "data"];
const dict = {}, missing = new Set();

function pick(){
  let want = null;
  try { want = new URLSearchParams(location.search).get("lang"); } catch(e) {}
  if(want && LOCALES[want]){ try { localStorage.setItem("scouter-lang", want); } catch(e) {} return want; }
  try { const s = localStorage.getItem("scouter-lang"); if(s && LOCALES[s]) return s; } catch(e) {}
  const nav = (typeof navigator !== "undefined" && (navigator.languages || [navigator.language])) || [];
  for(const l of nav){ const c = String(l || "").toLowerCase().split("-")[0]; if(LOCALES[c]) return c; }
  return FALLBACK;
}
const locale = pick();

/* 登记一批文字：I18N.add("zh", "infer", {"step.title": "…"}) → key "infer.step.title" */
function add(loc, ns, entries){
  const d = dict[loc] || (dict[loc] = {});
  for(const k in entries) d[ns ? ns + "." + k : k] = entries[k];
}
function lookup(key){
  const a = dict[locale]; if(a && Object.prototype.hasOwnProperty.call(a, key)) return a[key];
  const b = dict[FALLBACK]; if(b && Object.prototype.hasOwnProperty.call(b, key)) return b[key];
  return undefined;
}
function fill(s, p){ return p ? s.replace(/\{(\w+)\}/g, (m, k) => (p[k] !== undefined && p[k] !== null ? String(p[k]) : m)) : s; }
function t(key, p){
  const v = lookup(key);
  if(v === undefined){ missing.add(key); return key; }
  return typeof v === "function" ? v(p || {}) : fill(v, p);
}
const has = key => lookup(key) !== undefined;

/* 填静态 HTML 里标出来的文字 */
function apply(root){
  (root || document).querySelectorAll("[data-i18n],[data-i18n-html],[data-i18n-attr]").forEach(el => {
    if(el.dataset.i18n) el.textContent = t(el.dataset.i18n);
    if(el.dataset.i18nHtml) el.innerHTML = t(el.dataset.i18nHtml);
    if(el.dataset.i18nAttr) el.dataset.i18nAttr.split(";").forEach(pair => {
      const i = pair.indexOf("="); if(i > 0) el.setAttribute(pair.slice(0, i).trim(), t(pair.slice(i + 1).trim())); });
  });
  if(typeof document !== "undefined" && document.documentElement) document.documentElement.lang = LOCALES[locale].html;
}

/* 非 zh：解析到这里时同步追加该语言的文件（file:// 下也能用） */
if(locale !== FALLBACK && typeof document !== "undefined" && document.readyState === "loading"){
  const base = (document.currentScript && document.currentScript.src || "").replace(/src\/i18n\/i18n\.js(\?.*)?$/, "");
  for(const f of FILES) document.write('<script src="' + base + "locales/" + locale + "/" + f + '.js"><\/script>');
}

function setLocale(code){
  if(!LOCALES[code]) return;
  try { localStorage.setItem("scouter-lang", code); } catch(e) {}
  const u = new URL(location.href); u.searchParams.set("lang", code); location.href = u.toString();
}

return { add, t, has, apply, setLocale, missing, LOCALES, FILES, FALLBACK, get locale(){ return locale; }, get keys(){ return Object.keys(dict[FALLBACK] || {}); } };
})();
const t = I18N.t;
