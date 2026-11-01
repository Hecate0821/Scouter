#!/usr/bin/env node
// Scouter's data plane from the command line: the same Scouter.* calls the page makes, printed as JSON (docs/api.md).
//   node tools/scouter.js catalog                                   models, GPUs, topologies, bases, engines, defaults, data sets
//   node tools/scouter.js roofline --hw B200 --basis ideal --flops 4e12 --bytes 2e9 --dtype fp8
//   node tools/scouter.js infer    --model kimik3 --hw B200 --engine vllm --phase prefill --n 32768 --pick stepT,tComp
//   node tools/scouter.js train    --model qwen35 --fw torchtitan --ep 8 --basis ideal --pick stepT,mfu
//   node tools/scouter.js module   --model glm53f --phase decode --B 64 --pick measure.mods.full.t
//   node tools/scouter.js sweep    --run infer --model qwen35 --over.B 1,8,64,256 --pick stepT
//   node tools/scouter.js sets                                      measured data sets (origin, stage, hardware, size)
//   node tools/scouter.js points   --set inference_public --where.engine SGLang
//   node tools/scouter.js kernels                                   measured GEMM / copy points against the ideal and modeled rooflines, SOL η
//   node tools/scouter.js eff                                       per-kernel-class efficiency table and the data point behind each cell
//   node tools/scouter.js calib    --set h200_kernels --basis ideal  model against measured points (also cookbook_tpot, cookbook_ttft)
//   node tools/scouter.js serve    --model kimik3 --hw B300 --N 8 --conc 16 --isl 8192 --osl 1024   closed-loop benchmark estimate
//   node tools/scouter.js kv       --model qwen35 --hw H200 --N 8 --sessions 128 --hostGB 200   KV cache: bytes, hit rate, TTFT per turn
//   node tools/scouter.js trace    --model qwen35 --hw H200 --B 64 --steps 20 --file.trace trace.json   measured kernels by category
// Options: --model --hw --topo pick the tables; --basis ideal|modeled (default modeled); --g.<key> overrides one efficiency
// parameter; --pick a,b.c keeps only those fields; --over.<key> v1,v2 and --where.<key> v are lists and filters. Every other
// --key value overrides the run's default config. Values are parsed as JSON when they can be, otherwise kept as strings.
//   --json '{...}'   the whole query as one JSON object (or "-" to read it from stdin); flags given after it still apply
//   --file.<key> p   read the JSON file p (or p.gz) into <key> (for example --file.trace for a torch profiler export;
//                    a trace too big to load whole: uv run --no-project tools/reduce_trace.py trace.json.gz > kernels.json first)
//   --lang zh        render messages as text with locales/zh (default: messages stay { key, p })
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const load = require("../src/plane/node.js");
const ROOT = path.resolve(__dirname, "..");

const help = () => fs.readFileSync(__filename, "utf8").split("\n").filter(l => l.startsWith("//")).map(l => l.slice(3)).join("\n");
const [cmd, ...args] = process.argv.slice(2);
if(!cmd || cmd === "help" || cmd === "--help" || cmd === "-h"){ console.log(help()); process.exit(0); }

const parse = raw => { try { return JSON.parse(raw); } catch(e) { return raw; } };
const set = (o, key, v) => { const ks = key.split("."); let x = o; for(const k of ks.slice(0, -1)) x = x[k] = x[k] && typeof x[k] === "object" ? x[k] : {}; x[ks[ks.length - 1]] = v; };
let q = {}, lang = null;
for(let i = 0; i < args.length; i += 2){
  const k = args[i].replace(/^--/, ""), raw = args[i + 1];
  if(raw === undefined){ console.error(`missing value for --${k}`); process.exit(2); }
  if(k === "json") q = Object.assign(q, JSON.parse(raw === "-" ? fs.readFileSync(0, "utf8") : raw));
  else if(k === "lang") lang = raw;
  else if(k === "pick") q.pick = raw.split(",");
  else if(k.startsWith("file.")) set(q, k.slice(5), JSON.parse((raw.endsWith(".gz") ? require("zlib").gunzipSync(fs.readFileSync(raw)) : fs.readFileSync(raw)).toString("utf8")));
  else if(k.startsWith("over.")) set(q, k, String(raw).split(",").map(parse));
  else set(q, k, parse(raw));
}

const P = load();
if(lang){
  // Plug the language layer into the same context: i18n.js, then every file it lists for this language.
  const run = f => vm.runInContext(fs.readFileSync(path.join(ROOT, f), "utf8"), P.context, { filename: f });
  run("src/i18n/i18n.js");
  for(const f of vm.runInContext("I18N.FILES", P.context)) run(`locales/${lang}/${f}.js`);
  vm.runInContext("Msg.render = m => t(m.key, m.p); Msg.prototype.toJSON = function(){ return String(this); };", P.context);
}
const S = P.Scouter;
if(typeof S[cmd] !== "function" || !/^[a-z]+$/.test(cmd) || cmd === "gOf"){ console.error(`unknown command "${cmd}"\n\n${help()}`); process.exit(2); }
let out;
try { out = S[cmd](q); }
catch(e) { console.error(e.message); process.exit(2); }
process.stdout.write(JSON.stringify(out, (k, v) => typeof v === "function" ? undefined : v === Infinity ? "Infinity" : v, 1) + "\n");
