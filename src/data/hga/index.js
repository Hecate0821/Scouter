/* HuggingArch 导出的注册表：src/data/hga/<model>.js 由 tools/hga/export.py 生成，加载后调用 register。
   浏览器里按需插 <script>（选到这个模型时才加载），Node 里 src/plane/node.js 一次读完 */
const HGA_DATA = { models:{}, register(key, data){ this.models[key] = data; }, has(key){ return key in this.models; } };
