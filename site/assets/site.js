// 跳转链接在部分浏览器里只滚动不移动焦点，这里补一次焦点移动，便于键盘继续操作。
const skip = document.querySelector(".skip");
const main = document.getElementById("main");

if (skip && main) {
  skip.addEventListener("click", () => {
    main.focus({ preventScroll: true });
  });
}

// 打印时展开回测证据区，打印结束恢复到打印前的展开状态。
const evidence = document.querySelector("details[data-evidence]");
let evidenceWasOpen = null;

window.addEventListener("beforeprint", () => {
  if (!evidence || evidenceWasOpen !== null) return;
  evidenceWasOpen = evidence.open;
  evidence.open = true;
});

window.addEventListener("afterprint", () => {
  if (!evidence || evidenceWasOpen === null) return;
  evidence.open = evidenceWasOpen;
  evidenceWasOpen = null;
});
