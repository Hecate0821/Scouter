/* 在 <head> 里同步执行，先于样式生效，避免闪一下另一个主题。主题和 hecate0821.github.io 共用 localStorage 的 theme：
   没选过就跟系统，系统没说就是深色（和主页一样）。切换按钮在 shell.js。 */
(function(){
  var th; try { th = localStorage.getItem("theme"); } catch(e) {}
  if(th !== "light" && th !== "dark") th = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", th);
})();
