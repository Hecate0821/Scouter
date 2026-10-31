// Load the multi-file page in jsdom with its local scripts; resolves once every script has run.
// edit(html) may rewrite the page before it is parsed; scripts still load from next to the page. search ("?a=b") goes in the URL.
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { JSDOM, VirtualConsole } = require("jsdom");
module.exports = async function load(page, hash, edit, search) {
  const errs = [];
  const vc = new VirtualConsole();
  vc.on("error", e => errs.push(String(e && e.stack || e)));
  vc.on("jsdomError", e => errs.push("jsdomError: " + (e.stack || e.message)));
  const opts = { runScripts: "dangerously", resources: "usable", virtualConsole: vc, pretendToBeVisual: true,
    beforeParse(w) { w.Element.prototype.scrollIntoView = function () {}; if (hash) w.location.hash = hash; } };
  const file = path.resolve(page);
  if (search && !edit) edit = h => h;
  const dom = edit ? new JSDOM(edit(fs.readFileSync(file, "utf8")), Object.assign({ url: pathToFileURL(file).href + (search || "") }, opts))
    : await JSDOM.fromFile(file, opts);
  await new Promise(r => dom.window.addEventListener("load", r));
  return { dom, w: dom.window, d: dom.window.document, errs };
};
