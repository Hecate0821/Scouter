/* 计算核心不产出任何语言的文字：名字、警告、说明都写成 Msg.of(key, 参数)，key 指向 locales/<语言>/engine_*.js。
   消息被当成字符串用时（拼接、模板、String()、textContent）才渲染：界面层启动时设置 Msg.render（用 t() 翻译）；
   不设置时（例如在 Node 里单独跑计算核心）渲染成 key 本身。参数里可以再嵌消息。
   invalid()：这条警告表示配置跑不起来（不只是提醒），界面据此把配置判为不可行。 */
const Msg = (function(){
class Msg {
  constructor(key, p){ this.key = key; if(p) this.p = p; }
  invalid(){ this.bad = true; return this; }
  toString(){ return Msg.render(this); }
  toJSON(){ const o = { key: this.key }; if(this.p) o.p = this.p; if(this.bad) o.bad = true; return o; }
}
Msg.render = m => m.key;
Msg.of = (key, p) => new Msg(key, p);
/* 两条消息是否相同：同一个 key、同样的参数；不依赖渲染出来的文字 */
Msg.id = m => m instanceof Msg ? JSON.stringify(m) : String(m);
return Msg;
})();
