// i18n checks: (1) no human-language text left in code — every string / template literal in src/ (generated db.js excepted)
// and every text node / attribute in index.html must be free of CJK; (2) code never branches on display text — no regex literal
// with CJK / full-width punctuation / ×, and no such string compared with === or passed to split, replace, includes, test …
// (language-specific rules such as how to shorten a name are function values in the locale files); (3) every view renders with
// no missing key; (4) pseudo-localization: the views are rendered with every locale value wrapped in ⟦ ⟧ and every prose string
// of the data store in ⟪ ⟫, so a word left outside the markers was written by code and no translation can reach it. Outside the
// markers only numbers, units, code identifiers (anything with a digit or _ / . =), abbreviations (no lowercase letter), CamelCase
// names, the names in the engine's tables (models, GPUs, engines, kernels, formats, CP / EP methods) and NAMES may appear.
// Usage: node i18n.js [--static-only]
const fs = require("fs");
const path = require("path");
const acorn = require("acorn");
const ROOT = path.resolve(__dirname, "..");
const CJK = /[　-〿㐀-鿿＀-￯]/;
const problems = [];

// zh 的全部 key：用一个假的 I18N.add 执行 locales/zh/*.js
const KEYS = new Set();
for (const f of fs.readdirSync(path.join(ROOT, "locales", "zh")).filter(f => f.endsWith(".js"))) {
  try { new Function("I18N", fs.readFileSync(path.join(ROOT, "locales", "zh", f), "utf8"))({ add(loc, ns, e) { for (const k in e) KEYS.add(ns ? ns + "." + k : k); } }); }
  catch (e) { problems.push(`locales/zh/${f}: ${e.message}`); }
}
function scanJs(file) {
  const src = fs.readFileSync(file, "utf8");
  const rel = path.relative(ROOT, file);
  try {
    const toks = Array.from(acorn.tokenizer(src, { ecmaVersion: "latest", locations: true }));
    toks.forEach((tok, i) => {
      const isStr = tok.type === acorn.tokTypes.string || tok.type === acorn.tokTypes.template;
      if (isStr && CJK.test(String(tok.value))) problems.push(`${rel}:${tok.loc.start.line}  ${String(tok.value).slice(0, 70)}`);
      // t("literal") / I18N.t("literal")：key 必须存在（t("prefix." + x) 这种拼出来的 key 查不了，只能靠渲染时的 missing）
      const nx = toks[i + 3];
      if (tok.type === acorn.tokTypes.name && tok.value === "t" && toks[i + 1] && toks[i + 1].type === acorn.tokTypes.parenL && toks[i + 2] && toks[i + 2].type === acorn.tokTypes.string
          && nx && (nx.type === acorn.tokTypes.parenR || nx.type === acorn.tokTypes.comma)
          && !(toks[i - 1] && toks[i - 1].type === acorn.tokTypes.dot && !(toks[i - 2] && toks[i - 2].value === "I18N"))
          && !KEYS.has(toks[i + 2].value)) problems.push(`${rel}:${tok.loc.start.line}  missing key ${toks[i + 2].value}`);
    });
  } catch (e) { problems.push(`${rel}: parse error ${e.message}`); }
}
// 按显示文字分支：正则里带中文 / 全角标点 / ×，或者拿这类字符串去比较、split、replace、includes……
const TEXTY = /[　-〿㐀-鿿＀-￯×]/;
const MATCHERS = new Set(["includes", "startsWith", "endsWith", "indexOf", "lastIndexOf", "split", "replace", "replaceAll", "match", "matchAll", "test", "search"]);
const astWalk = (n, f) => { if (!n || typeof n.type !== "string") return; f(n);
  for (const k in n) { const v = n[k]; if (Array.isArray(v)) v.forEach(c => astWalk(c, f)); else if (v && typeof v.type === "string") astWalk(v, f); } };
function scanLogic(file) {
  const rel = path.relative(ROOT, file);
  let ast; try { ast = acorn.parse(fs.readFileSync(file, "utf8"), { ecmaVersion: "latest", locations: true }); } catch (e) { return; }
  const texty = n => n && n.type === "Literal" && typeof n.value === "string" && TEXTY.test(n.value);
  astWalk(ast, n => {
    const at = `${rel}:${n.loc.start.line}  `;
    if (n.type === "Literal" && n.regex && TEXTY.test(n.regex.pattern)) problems.push(at + `matches display text: /${n.regex.pattern}/`);
    if (n.type === "BinaryExpression" && /^[!=]==?$/.test(n.operator) && (texty(n.left) || texty(n.right))) problems.push(at + "compares with display text");
    if (n.type === "CallExpression" && n.callee.type === "MemberExpression" && !n.callee.computed && MATCHERS.has(n.callee.property.name) && n.arguments.some(texty))
      problems.push(at + `matches display text: .${n.callee.property.name}(${JSON.stringify(n.arguments.find(texty).value)})`);
  });
}
function walk(dir, out = []) {
  for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, f.name);
    if (f.isDirectory()) walk(p, out); else if (f.name.endsWith(".js")) out.push(p);
  }
  return out;
}
for (const f of walk(path.join(ROOT, "src"))) if (!f.endsWith(path.join("data", "db.js")) && !f.endsWith(path.join("i18n", "i18n.js"))) { scanJs(f); scanLogic(f); }
// 数据点库的译文（locales/<语言>/data.js）：每个 key 都要对得上一条记录的文字字段，写错的 key 会悄悄不生效
{ const db = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "db.json"), "utf8")), recs = new Map();
  for (const [id, s] of Object.entries(db.sets)) recs.set(id, s);
  for (const p of db.points) recs.set(p.id, p);
  const textAt = key => { for (const [id, r] of recs) { if (!key.startsWith(id + ".")) continue;
      const [f, i, ...more] = key.slice(id.length + 1).split(".");
      if (f === "metrics" && i && !more.length && r.metrics && r.metrics[i]) return true;
      const v = r[f]; if (i === undefined ? typeof v === "string" : Array.isArray(v) && typeof v[+i] === "string" && !more.length) return true; }
    return false; };
  for (const loc of fs.readdirSync(path.join(ROOT, "locales"))) {
    const f = path.join(ROOT, "locales", loc, "data.js"); if (!fs.existsSync(f)) continue;
    const keys = []; new Function("I18N", fs.readFileSync(f, "utf8"))({ add(l, ns, e) { keys.push(...Object.keys(e)); } });
    for (const k of keys) if (!textAt(k)) problems.push(`locales/${loc}/data.js  ${k} matches no text field in data/db.json`);
  } }   // i18n.js 里只有各语言的自称（LOCALES.name）
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8").replace(/<!--[\s\S]*?-->/g, "");
html.split("\n").forEach((line, i) => { if (CJK.test(line)) problems.push(`index.html:${i + 1}  ${line.trim().slice(0, 70)}`); });
for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"|data-i18n-attr="([^"]+)"/g)) {
  const ks = m[1] ? [m[1]] : m[2].split(";").map(p => p.slice(p.indexOf("=") + 1).trim());
  for (const k of ks) if (!KEYS.has(k)) problems.push(`index.html  missing key ${k}`);
}

(async () => {
  if (!process.argv.includes("--static-only")) {
    const load = require("./load");
    // 伪本地化：locale 的每个值包进 ⟦ ⟧，数据点库里的文字包进 ⟪ ⟫：含中文或空格的，或者不在引用字段里、带大小写的词；
    // 引用字段里的单词（B200、qwen35、rl……）是查表用的 key，不动
    const WRAP = `<script>(function(){ const add = I18N.add; I18N.add = function(loc, ns, e){ const o = {};
      for (const k in e) { const v = e[k]; o[k] = typeof v === "function" ? function(p){ return "⟦" + v(p) + "⟧"; } : "⟦" + v + "⟧"; }
      return add(loc, ns, o); }; })();</script>`;
    const MARK = `<script>(function(){ const REF = /^(id|set|url|href|hw|topo|model|engine|fw|kind|stage|origin|unit|project|version|date|phase|dtype)$/;
      const prose = (v, k) => /[\\u3000-\\u9fff\\uff00-\\uffef ]/.test(v) || !REF.test(k) && /[A-Z]/.test(v) && /[a-z]{2}/.test(v);
      const mark = (v, k) => typeof v === "string" ? (prose(v, k) && !/^https?:/.test(v) ? "⟪" + v + "⟫" : v)
        : Array.isArray(v) ? v.map(x => mark(x, k)) : v && typeof v === "object" ? (Object.keys(v).forEach(f => { v[f] = mark(v[f], f); }), v) : v;
      mark(DB_RAW, ""); })();</script>`;
    const { w, d, errs } = await load(path.join(ROOT, "index.html"), "#overview", h => h
      .replace('<script src="src/i18n/i18n.js"></script>', m => m + WRAP).replace('<script src="src/data/db.js"></script>', m => m + MARK));
    const go = h => { w.location.hash = h; w.dispatchEvent(new w.HashChangeEvent("hashchange")); scan(h, d.querySelector(`section[data-pane="${h}"]`)); };
    const setSel = (sel, v) => { const el = d.querySelector(sel); if (!el) return; el.value = v; el.dispatchEvent(new w.Event("change", { bubbles: true })); };

    // 标记外面剩下的词：数字、单位、代码标识符、缩写、CamelCase、引擎表里的名字、NAMES 都和语言无关，其余的词就是代码写死的文字
    const Msg = w.eval("Msg"), table = new Set();
    const names = v => { if (typeof v === "string") { if (/[A-Za-z]{2}/.test(v)) table.add(v); } else if (v && typeof v === "object" && !(v instanceof Msg)) Object.values(v).forEach(names); };
    names(w.eval("[Common.MODELS, Common.HW, Common.TOPO, Infer.ENGINES, Infer.SPEC, Train.PREC, Par.EP, Par.CP, Quant.SCHEMES, Quant.RECIPES]"));
    const TABLE = new RegExp("(?<![A-Za-z0-9])(" + [...table].sort((a, b) => b.length - a.length).map(s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")(?![A-Za-z0-9])", "g");
    const UNITS = new Set(["ms", "us", "s", "min", "h", "tok", "x"]);
    // 代码里会出现的专名和代码词：框架、GPU 架构、GEMM 的三遍、计数同步的 op 名、命令行参数；再加上模型名里的每个词
    const NAMES = new Set(["torchtitan", "Megatron", "Megatron-Core", "Miles", "slime", "Blackwell", "Hopper", "fprop", "dgrad", "wgrad", "sync", "top-k", "dp-size"]);
    for (const m of Object.values(w.eval("Common.MODELS"))) for (const v of [m.name, m.short]) if (typeof v === "string") v.split(/\s+/).forEach(x => NAMES.add(x));
    for (const id of Object.keys(w.eval("DB.sets"))) NAMES.add(id);   // 数据集的 id（数据浏览里照印）
    const plain = p => p.length > 1 && /[a-z]/.test(p) && !/^[A-Za-z][a-z]*[A-Z]/.test(p) && !UNITS.has(p) && !NAMES.has(p);
    const words = s => (s.match(/[A-Za-z0-9_./=-]*[A-Za-z][A-Za-z0-9_./=-]*/g) || [])
      .filter(tok => !NAMES.has(tok) && !/[\d_/.=]|^-/.test(tok) && tok.split("-").some(plain));
    const strip = s => { let depth = 0, out = ""; for (const ch of s) {
      if (ch === "⟦" || ch === "⟪") { depth++; out += "\u0001"; } else if (ch === "⟧" || ch === "⟫") { depth = Math.max(0, depth - 1); out += "\u0001"; } else if (!depth) out += ch; }
      return out; };
    const coded = new Map(), seen = new Set();
    function scan(where, root) {
      if (!root) return;
      const nodes = [], it = d.createTreeWalker(root, w.NodeFilter.SHOW_TEXT);
      for (let n; (n = it.nextNode()); ) if (!n.parentElement.closest("code,script,style,#m-db-pts")) nodes.push(n.data);   // #m-db-pts：数据浏览表，印的是记录的原始字段
      const attrs = Array.from(root.querySelectorAll("[title],[aria-label],[placeholder],[alt]")).flatMap(el => ["title", "aria-label", "placeholder", "alt"].map(a => el.getAttribute(a) || ""));
      for (let seg of [strip(nodes.join("\u0001")), ...attrs.map(strip)].join("\u0001").split("\u0001")) {
        seg = seg.replace(/\s+/g, " ").trim();
        if (!seg || seen.has(seg)) continue; seen.add(seg);
        for (const tok of words(seg.replace(TABLE, " "))) if (!coded.has(tok)) coded.set(tok, { where, seg });
      }
    }

    scan("shell", d.body);
    const views = Array.from(d.querySelectorAll("section[data-pane]")).map(s => s.dataset.pane);
    for (const m of ["qwen35", "qwen38_27b", "glm53f", "musespark", "kimik3"]) {
      setSel("#g-model", m);
      for (const v of views) go(v);
      for (const eng of ["sglang", "vllm", "tokenspeed", "titan"]) { setSel("[data-bind='inf.engine']", eng);
        for (const ph of ["decode", "prefill"]) { setSel("[data-bind='inf.phase']", ph); go("infer/step"); } go("infer/deploy"); go("infer/spec"); }
      for (const fw of ["megatron", "torchtitan"]) { go("train/step"); setSel("[data-bind='trn.fw']", fw); go("train/par"); }
      setSel("[data-bind='inf.engine']", "sglang"); setSel("[data-bind='inf.phase']", "decode");
      scan("racks · " + m, d.body);
    }
    if (coded.size) problems.push(`${coded.size} words written by code rather than a locale file or the data:` +
      [...coded].map(([tok, o]) => `\n  ${tok.padEnd(16)} ${o.where}: ${JSON.stringify(o.seg.slice(0, 90))}`).join(""));
    // 数据点库的译文真的换得进去：登记三条（数据集标题、指标名、数组字段里的一项），再跑一遍 localizeData
    w.eval(`I18N.add("zh", "data", { "cp_attention.title": "@T", "cp_attention.metrics.fb_ms": "@M", "${"releases:lmsys:2025-11-03-gpt-oss-on-nvidia-dgx-spark"}.highlights.1": "@H" }); localizeData();`);
    const got = w.eval(`[DB.meta("cp_attention").title, DB.meta("cp_attention").metrics.fb_ms.label, DB.one("releases", { id: "releases:lmsys:2025-11-03-gpt-oss-on-nvidia-dgx-spark" }).highlights[1]].join()`);
    if (got !== "⟦@T⟧,⟦@M⟧,⟦@H⟧") problems.push("data translations are not applied: " + got);
    const miss = Array.from(w.eval("I18N").missing);
    if (miss.length) problems.push("missing keys: " + miss.slice(0, 40).join(", ") + (miss.length > 40 ? " …" : ""));
    if (errs.length) problems.push("script errors: " + errs.slice(0, 3).join("\n"));
  }
  if (problems.length) { console.log(problems.slice(0, process.env.ALL ? Infinity : 200).join("\n")); console.log(problems.length + " i18n problems"); process.exitCode = 1; return; }
  console.log("i18n ok");
})();
