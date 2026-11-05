/* Node 入口（浏览器不加载这个文件）：在一个只有 JS 内置对象的干净上下文里，按 index.html 的顺序加载计算核心，返回它的接口。
   上下文里没有 window / document / t() / I18N / 数据点库，所以计算核心一旦碰了界面、文字或数据就会直接报错（tests/engine.js 就靠这个）。
   消息默认渲染成 key；要出中文，把 locales 和 src/i18n/i18n.js 也加载进来再设置 Msg.render（见 tools/scouter.js）。
     const E = require("./src/engine/node.js")();
     const r = E.Infer.step(E.MODELS.qwen35, E.HW.H200, E.TOPO.hgx8, Object.assign({}, E.Infer.BASE, { B: 32 }), E.G0); */
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");

const FILES = ["msg.js", "common.js", "par.js", "infer.js", "train.js", "quant.js", "arch.js", "calib.js", "trace.js", "kv.js", "fabric.js"];
const MODULES = ["Msg", "Common", "Par", "Infer", "Train", "Quant", "Arch", "Calib", "Trace", "KV", "Fabric"];

function load(ctx){
  ctx = ctx || vm.createContext({});
  for(const f of FILES){ const file = path.join(__dirname, f); vm.runInContext(fs.readFileSync(file, "utf8"), ctx, { filename: file }); }
  const api = { context: ctx };
  for(const n of MODULES) api[n] = vm.runInContext(n, ctx);
  const { MODELS, HW, TOPO, G0 } = api.Common;
  return Object.assign(api, { MODELS, HW, TOPO, G0 });
}
load.FILES = FILES;
load.MODULES = MODULES;
module.exports = load;
