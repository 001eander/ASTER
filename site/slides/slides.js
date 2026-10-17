/**
 * 幻灯片交互：深链接、键盘翻页、页码与进度同步、16:9 缩放。
 * 路由解析与页码夹取全部来自纯模块 slides-route.js，本文件只做 DOM 接线。
 */

import {
  FIRST_SLIDE,
  LAST_SLIDE,
  clampSlide,
  nextSlide,
  parseSlideHash,
  prevSlide,
  slideHash,
} from "./slides-route.js";

const DESIGN_WIDTH = 1280;

const deck = document.querySelector(".deck");
const slides = Array.from(document.querySelectorAll("[data-slide]"));
const prevButton = document.querySelector('[data-action="prev"]');
const nextButton = document.querySelector('[data-action="next"]');
const currentOutput = document.querySelector("[data-current]");
const progressBar = document.querySelector("[data-progress]");
const progressTrack = document.querySelector('[role="progressbar"]');
const slideLinks = Array.from(document.querySelectorAll("[data-slide-link]"));

let current = FIRST_SLIDE;

function render() {
  for (const slide of slides) {
    const page = Number(slide.getAttribute("data-slide"));
    slide.classList.toggle("is-current", page === current);
  }

  if (currentOutput) currentOutput.textContent = String(current);
  if (prevButton) prevButton.disabled = current === FIRST_SLIDE;
  if (nextButton) nextButton.disabled = current === LAST_SLIDE;

  if (progressBar) progressBar.style.width = `${(current / LAST_SLIDE) * 100}%`;
  if (progressTrack) progressTrack.setAttribute("aria-valuenow", String(current));

  for (const link of slideLinks) {
    if (Number(link.getAttribute("data-slide-link")) === current) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
}

function writeHash() {
  const target = slideHash(current);
  if (window.location.hash === target) return;

  try {
    window.history.replaceState(null, "", target);
  } catch (error) {
    // file:// 下部分浏览器对 replaceState 抛 SecurityError，降级为直接改 hash。
    window.location.hash = target;
  }
}

function show(page, { syncHash = true } = {}) {
  const target = clampSlide(page);
  if (target === current) {
    if (syncHash) writeHash();
    return;
  }

  current = target;
  render();
  if (syncHash) writeHash();
}

function fit() {
  if (!deck) return;
  const width = deck.getBoundingClientRect().width;
  if (!width) return;
  document.documentElement.style.setProperty("--slide-scale", String(width / DESIGN_WIDTH));
}

prevButton?.addEventListener("click", () => show(prevSlide(current)));
nextButton?.addEventListener("click", () => show(nextSlide(current)));

window.addEventListener("hashchange", () => {
  const target = parseSlideHash(window.location.hash);
  if (target !== current) show(target, { syncHash: false });
});

window.addEventListener("popstate", () => {
  const target = parseSlideHash(window.location.hash);
  if (target !== current) show(target, { syncHash: false });
});

document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;

  const target = event.target;
  if (target instanceof HTMLElement) {
    if (target.isContentEditable) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
  }

  switch (event.key) {
    case "ArrowRight":
    case "PageDown":
      show(nextSlide(current));
      break;
    case "ArrowLeft":
    case "PageUp":
      show(prevSlide(current));
      break;
    case "Home":
      show(FIRST_SLIDE);
      break;
    case "End":
      show(LAST_SLIDE);
      break;
    default:
      return;
  }

  event.preventDefault();
});

current = parseSlideHash(window.location.hash);
render();

if (deck) {
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(fit).observe(deck);
  } else {
    window.addEventListener("resize", fit, { passive: true });
  }
  fit();
}

writeHash();
