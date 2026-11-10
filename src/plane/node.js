/* 数据平面的 Node 入口（浏览器不加载这个文件）：在计算核心的干净上下文里再加载数据点库和 scouter.js，返回 Scouter 和计算核心的接口。
   上下文里没有 window / document / t() / 界面状态，所以数据平面一旦碰了界面就会直接报错（tests/plane.js 靠这个）。
     const P = require("./src/plane/node.js")();
     P.Scouter.infer({ model: "kimik3", hw: "B200", basis: "ideal", B: 64 }).stepT */
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const loadEngine = require("../engine/node.js");

/* HuggingArch 的导出（src/data/hga/<model>.js）：浏览器按需加载，这里全部加载 */
const HGA_DIR = path.join(__dirname, "../data/hga");
const hgaFiles = () => fs.readdirSync(HGA_DIR).filter(f => f.endsWith(".js") && f !== "index.js").sort().map(f => "../data/hga/" + f);
const FILES = ["../data/db.js", "../data/query.js", "../data/hga/index.js", "scouter.js"];

function load(ctx){
  const E = loadEngine(ctx);
  const files = FILES.slice(0, 3).concat(hgaFiles(), FILES.slice(3));
  for(const f of files){ const file = path.join(__dirname, f); vm.runInContext(fs.readFileSync(file, "utf8"), E.context, { filename: file }); }
  return Object.assign(E, { DB: vm.runInContext("DB", E.context), Scouter: vm.runInContext("Scouter", E.context) });
}
load.FILES = FILES;
module.exports = load;
