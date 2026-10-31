#!/usr/bin/env node
// Run the Scouter engine without a browser and print the result as JSON.
//   node tools/sim.js infer --model qwen35 --hw H200 --topo hgx8 --engine vllm --B 32 --n 16384
//   node tools/sim.js train --model kimik3 --hw B200 --topo nvl64 --fw torchtitan --ep 16 --cp 2
//   node tools/sim.js arch  --model glm53f --hw H200 --phase prefill --n 32768
//   node tools/sim.js list                        models, GPUs, topologies, engines, precision recipes
// Every other --key value overrides the default config (Infer.BASE / Train.BASE / Arch.defaults); --N defaults to the topology's GPUs
// for training. Values are parsed as JSON when they can be (numbers, true / false), otherwise kept as strings.
//   --lang zh    render messages as text with locales/zh (default: messages stay { key, p })
//   --g.etaC 0.6 override an efficiency parameter (Common.G0)
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const load = require("../src/engine/node.js");
const ROOT = path.resolve(__dirname, "..");

const [cmd, ...rest] = process.argv.slice(2);
const opt = {}, g = {};
for (let i = 0; i < rest.length; i += 2) {
  const k = rest[i].replace(/^--/, ""), raw = rest[i + 1];
  let v; try { v = JSON.parse(raw); } catch (e) { v = raw; }
  if (k.startsWith("g.")) g[k.slice(2)] = v; else opt[k] = v;
}
const E = load();
if (opt.lang) {
  // Plug the language layer into the same context: i18n.js, then every file it lists for this language.
  const run = f => vm.runInContext(fs.readFileSync(path.join(ROOT, f), "utf8"), E.context, { filename: f });
  run("src/i18n/i18n.js");
  for (const f of vm.runInContext("I18N.FILES", E.context)) run(`locales/${opt.lang}/${f}.js`);
  vm.runInContext("Msg.render = m => t(m.key, m.p); Msg.prototype.toJSON = function(){ return String(this); };", E.context);
  delete opt.lang;
}
const take = (k, table, dflt) => { const v = opt[k] !== undefined ? opt[k] : dflt; delete opt[k];
  if (!(v in table)) { console.error(`unknown ${k} "${v}"; one of: ${Object.keys(table).join(", ")}`); process.exit(2); } return v; };
const G = Object.assign({}, E.G0, g);
let out;
if (cmd === "list") {
  out = { models: Object.keys(E.MODELS), hw: Object.keys(E.HW), topo: Object.keys(E.TOPO), engines: Object.keys(E.Infer.ENGINES),
    spec: Object.keys(E.Infer.SPEC), prec: Object.keys(E.Train.PREC), epBackends: Object.keys(E.Par.EP), quant: E.Quant.ORDER, g: E.G0 };
} else if (cmd === "infer" || cmd === "train" || cmd === "arch") {
  const m = E.MODELS[take("model", E.MODELS, "qwen35")], hw = E.HW[take("hw", E.HW, "H200")], topo = E.TOPO[take("topo", E.TOPO, "hgx8")];
  if (cmd === "infer") out = E.Infer.step(m, hw, topo, Object.assign({}, E.Infer.BASE, opt), G);
  else if (cmd === "train") out = E.Train.run(m, hw, topo, Object.assign({}, E.Train.BASE, { N: topo.gpus }, opt), G);
  else { const wl = Object.assign(E.Arch.defaults(m), opt); out = { workload: wl, modules: E.Arch.present(m), measure: E.Arch.measure(m, hw, G, wl), facts: E.Arch.facts(m, wl) }; }
} else {
  console.error(fs.readFileSync(__filename, "utf8").split("\n").filter(l => l.startsWith("//")).map(l => l.slice(3)).join("\n"));
  process.exit(cmd ? 2 : 0);
}
process.stdout.write(JSON.stringify(out, (k, v) => typeof v === "function" ? undefined : v === Infinity ? "Infinity" : v, 1) + "\n");
