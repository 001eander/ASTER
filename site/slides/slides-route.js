/**
 * 幻灯片路由的纯逻辑：解析 hash、夹取页码、前后翻页。
 *
 * 不依赖浏览器环境，浏览器直接 import，vitest 在 node 里跑同一份实现。
 * 页码范围固定在 1..11，任何越界输入都会被夹取。
 */

export const FIRST_SLIDE = 1;
export const LAST_SLIDE = 11;
export const SLIDE_COUNT = LAST_SLIDE - FIRST_SLIDE + 1;

/**
 * 把任意输入夹到 1..11。
 * 非数字回落到第 1 页，小数截断，越界夹到首末页。
 */
export function clampSlide(n) {
  const value = Number(n);
  if (!Number.isFinite(value)) return FIRST_SLIDE;
  const truncated = Math.trunc(value);
  return Math.min(Math.max(truncated, FIRST_SLIDE), LAST_SLIDE);
}

/**
 * 解析 location.hash 形式的字符串。
 * `#3` 得 3，`#3?x` 也只取开头的数字；空串、`#`、非数字与负数回落到第 1 页。
 */
export function parseSlideHash(hash) {
  if (typeof hash !== "string") return FIRST_SLIDE;
  const body = hash.trim().replace(/^#/, "");
  const match = /^-?\d+/.exec(body);
  if (!match) return FIRST_SLIDE;
  return clampSlide(Number.parseInt(match[0], 10));
}

/** 上一页，首页保持首页。 */
export function prevSlide(n) {
  return clampSlide(clampSlide(n) - 1);
}

/** 下一页，末页保持末页。 */
export function nextSlide(n) {
  return clampSlide(clampSlide(n) + 1);
}

/** 页码对应的 hash，输出前先夹取。 */
export function slideHash(n) {
  return `#${clampSlide(n)}`;
}
