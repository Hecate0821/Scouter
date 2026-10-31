/* 数据平面的 Node 入口（浏览器不加载这个文件）：在计算核心的干净上下文里再加载数据点库和 scouter.js，返回 Scouter 和计算核心的接口。
   上下文里没有 window / document / t() / 界面状态，所以数据平面一旦碰了界面就会直接报错（tests/plane.js 靠这个）。
     const P = require("./src/plane/node.js")();
     P.Scouter.infer({ model: "kimik3", hw: "B200", basis: "ideal", B: 64 }).stepT */
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const loadEngine = require("../engine/node.js");

const FILES = ["../data/db.js", "../data/query.js", "scouter.js"];

function load(ctx){
  const E = loadEngine(ctx);
  for(const f of FILES){ const file = path.join(__dirname, f); vm.runInContext(fs.readFileSync(file, "utf8"), E.context, { filename: file }); }
  return Object.assign(E, { DB: vm.runInContext("DB", E.context), Scouter: vm.runInContext("Scouter", E.context) });
}
load.FILES = FILES;
module.exports = load;
