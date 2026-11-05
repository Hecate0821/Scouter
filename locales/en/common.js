/* common: shared charts, state, error messages (src/ui/base.js, state.js, charts.js, app.js, src/data/query.js) */
I18N.add("en", "common", {
  /* Language-specific formatting rules (the code knows no language's punctuation): names drop parenthesised notes; presets take the part before the first separator */
  "fmt.shortName": p => String(p.s).replace(/\s*\([^()]*\)/g, "").trim(),
  "fmt.presetShort": p => String(p.s).split(/[,:;(]/)[0].trim(),
  "calcError": "Calculation error: {m}",
  "chart.error": "Chart error: {m}",
  "chart.empty": "No points to plot",
  "chart.noPoint": "No point at this position",
  "pill.compute": "compute",
  "pill.memory": "memory",
  "pill.launch": "launch",
  "pill.comm": "comm",
  "phase.decode": "decode",
  "phase.prefill": "prefill",
  "ctl.kvCache": "KV cache",
  "dt.fp8Block": "FP8 block",
  "dt.blockFp8": "block FP8",
  "gantt.sm": "SM (compute)",
  "anonPoint": "Anonymized data point",
  "preset.need": " (needs {n}{hw} GPUs; selecting it switches the topology)"
});
