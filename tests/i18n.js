// i18n checks: (1) no human-language text left in code — every string / template literal in src/ (generated db.js excepted)
// and every text node / attribute in index.html must be free of CJK; (2) every view renders with no missing key.
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
function walk(dir, out = []) {
  for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, f.name);
    if (f.isDirectory()) walk(p, out); else if (f.name.endsWith(".js")) out.push(p);
  }
  return out;
}
for (const f of walk(path.join(ROOT, "src"))) if (!f.endsWith(path.join("data", "db.js")) && !f.endsWith(path.join("i18n", "i18n.js"))) scanJs(f);   // i18n.js 里只有各语言的自称（LOCALES.name）
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8").replace(/<!--[\s\S]*?-->/g, "");
html.split("\n").forEach((line, i) => { if (CJK.test(line)) problems.push(`index.html:${i + 1}  ${line.trim().slice(0, 70)}`); });
for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"|data-i18n-attr="([^"]+)"/g)) {
  const ks = m[1] ? [m[1]] : m[2].split(";").map(p => p.slice(p.indexOf("=") + 1).trim());
  for (const k of ks) if (!KEYS.has(k)) problems.push(`index.html  missing key ${k}`);
}

(async () => {
  if (!process.argv.includes("--static-only")) {
    const load = require("./load");
    const { w, d, errs } = await load(path.join(ROOT, "index.html"), "#overview");
    const go = h => { w.location.hash = h; w.dispatchEvent(new w.HashChangeEvent("hashchange")); };
    const setSel = (sel, v) => { const el = d.querySelector(sel); if (!el) return; el.value = v; el.dispatchEvent(new w.Event("change", { bubbles: true })); };
    const views = Array.from(d.querySelectorAll("section[data-pane]")).map(s => s.dataset.pane);
    for (const m of ["qwen35", "qwen38_27b", "glm53f", "musespark", "kimik3"]) {
      setSel("#g-model", m);
      for (const v of views) go(v);
      for (const eng of ["sglang", "vllm", "tokenspeed", "titan"]) { setSel("[data-bind='inf.engine']", eng);
        for (const ph of ["decode", "prefill"]) { setSel("[data-bind='inf.phase']", ph); go("infer/step"); } go("infer/deploy"); go("infer/spec"); }
      for (const fw of ["megatron", "torchtitan"]) { go("train/step"); setSel("[data-bind='trn.fw']", fw); go("train/par"); }
      setSel("[data-bind='inf.engine']", "sglang"); setSel("[data-bind='inf.phase']", "decode");
    }
    const miss = Array.from(w.eval("I18N").missing);
    if (miss.length) problems.push("missing keys: " + miss.slice(0, 40).join(", ") + (miss.length > 40 ? " …" : ""));
    if (errs.length) problems.push("script errors: " + errs.slice(0, 3).join("\n"));
  }
  if (problems.length) { console.log(problems.slice(0, process.env.ALL ? Infinity : 200).join("\n")); console.log(problems.length + " i18n problems"); process.exitCode = 1; return; }
  console.log("i18n ok");
})();
