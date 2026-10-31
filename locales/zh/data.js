/* 数据点库里的文字（数据集的 title / conditions / 指标名，公开数据点的 claim、value、setting、caveat、highlights 等）是内容，
   原文就在 data/*.json 里，用中文写成，所以 zh 不需要条目。别的语言在 locales/<语言>/data.js 里按记录 id 给译文：
     "<数据点 id>.<字段>"            例 "spec_public:eagle3-paper.setting"
     "<数据点 id>.<字段>.<序号>"     数组字段，例 "releases:lmsys:2025-11-03-gpt-oss-on-nvidia-dgx-spark.highlights.0"
     "<数据集 id>.<字段>"            例 "cp_attention.conditions"
     "<数据集 id>.metrics.<指标>"    指标名，例 "cp_attention.metrics.fb_ms"
   界面启动时换进去（src/ui/base.js localizeData）；没给译文的字段显示原文。tests/i18n.js 检查每个 key 都对得上一个文字字段。 */
I18N.add("zh", "data", {});
