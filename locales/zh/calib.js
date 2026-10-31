/* calib：index.html 里的静态文字（由迁移脚本从 index.html 抽出） */
I18N.add("zh", "calib", {
  "b1.h2": "和实测对一下",
  "b1.lede": "数据点库里的匿名实测（8 × H200，一个 NVLink 域，BF16，Qwen3.5-35B-A3B 的形状），凡是这一页能表达的设置，都原样喂给后训练核心的 Train.run / Par：torchtitan、full AC 与否照实测，只把 attention 核心效率换成单卡校准值（FA2 在 256 维头上 273 / 989），其余都是左侧“效率系数”的当前值，没有对这些点做拟合，改效率系数这里会一起变。小形状（每卡几 K token、几十 µs 的 kernel）本页明显偏快：固定开销只按 launch 和 α 计。在汇总表里点一行，下面列出这个数据集的每个点。",
  "attr.p-calib-plot.aria-label": "本页预测对实测的散点图，对数坐标",
});
