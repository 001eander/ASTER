import { describe, expect, it } from "vitest";
import {
  FIRST_SLIDE,
  LAST_SLIDE,
  SLIDE_COUNT,
  clampSlide,
  nextSlide,
  parseSlideHash,
  prevSlide,
  slideHash,
} from "../site/slides/slides-route.js";

describe("slides-route 页码常量", () => {
  it("deck 共 11 页，页码范围 1..11", () => {
    expect(FIRST_SLIDE).toBe(1);
    expect(LAST_SLIDE).toBe(11);
    expect(SLIDE_COUNT).toBe(11);
  });
});

describe("parseSlideHash", () => {
  it("把 #3 解析为第 3 页", () => {
    expect(parseSlideHash("#3")).toBe(3);
  });

  it("容忍尾部杂质，取开头的数字部分", () => {
    expect(parseSlideHash("#3?x")).toBe(3);
  });

  it("无 hash 回落到第 1 页", () => {
    expect(parseSlideHash("")).toBe(1);
    expect(parseSlideHash("#")).toBe(1);
    expect(parseSlideHash(null)).toBe(1);
  });

  it("非数字回落到第 1 页", () => {
    expect(parseSlideHash("#abc")).toBe(1);
  });

  it("把 #0 与负数夹到第 1 页", () => {
    expect(parseSlideHash("#0")).toBe(1);
    expect(parseSlideHash("#-4")).toBe(1);
  });

  it("把超出末页的 #99 夹到第 11 页", () => {
    expect(parseSlideHash("#99")).toBe(11);
  });

  it("末页 #11 保持第 11 页", () => {
    expect(parseSlideHash("#11")).toBe(11);
  });
});

describe("clampSlide", () => {
  it("保持范围内页码不变", () => {
    expect(clampSlide(1)).toBe(1);
    expect(clampSlide(7)).toBe(7);
    expect(clampSlide(11)).toBe(11);
  });

  it("把越界页码夹到首末页", () => {
    expect(clampSlide(0)).toBe(1);
    expect(clampSlide(-5)).toBe(1);
    expect(clampSlide(12)).toBe(11);
    expect(clampSlide(99)).toBe(11);
  });
});

describe("prevSlide / nextSlide", () => {
  it("首页的上一页仍是首页", () => {
    expect(prevSlide(1)).toBe(1);
  });

  it("末页的下一页仍是末页", () => {
    expect(nextSlide(11)).toBe(11);
  });

  it("中间页正常前后移动", () => {
    expect(prevSlide(2)).toBe(1);
    expect(prevSlide(11)).toBe(10);
    expect(nextSlide(1)).toBe(2);
    expect(nextSlide(10)).toBe(11);
  });

  it("越界输入先夹取再移动", () => {
    expect(prevSlide(99)).toBe(10);
    expect(nextSlide(0)).toBe(2);
  });
});

describe("slideHash", () => {
  it("输出带井号的页码", () => {
    expect(slideHash(3)).toBe("#3");
  });

  it("输出前先夹取到合法范围", () => {
    expect(slideHash(0)).toBe("#1");
    expect(slideHash(12)).toBe("#11");
    expect(slideHash(99)).toBe("#11");
  });

  it("首末页输出稳定", () => {
    expect(slideHash(1)).toBe("#1");
    expect(slideHash(11)).toBe("#11");
  });
});
