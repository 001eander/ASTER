// 跳转链接在部分浏览器里只滚动不移动焦点，这里补一次焦点移动，便于键盘继续操作。
const skip = document.querySelector(".skip");
const main = document.getElementById("main");

if (skip && main) {
  skip.addEventListener("click", () => {
    main.focus({ preventScroll: true });
  });
}
