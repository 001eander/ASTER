import { describe, expect, it } from "vitest";
import { escapeRawControlCharsInStrings, parseLooseJson } from "./loose-json.js";

describe("parseLooseJson", () => {
  it("parses well-formed json unchanged", () => {
    expect(parseLooseJson('{"a":1}')).toEqual({ a: 1 });
  });

  it("repairs raw newlines inside a string", () => {
    const broken = '{\n  "context": "第一行\n第二行",\n  "stop": false\n}';
    expect(() => JSON.parse(broken)).toThrow();
    expect(parseLooseJson(broken)).toEqual({ context: "第一行\n第二行", stop: false });
  });

  it("repairs raw tabs and other control characters inside a string", () => {
    const broken = '{"context":"a\tb\u0001c"}';
    expect(parseLooseJson(broken)).toEqual({ context: "a\tb\u0001c" });
  });

  it("keeps already-escaped sequences intact", () => {
    const text = '{"context":"a\\nb\\"c"}';
    expect(parseLooseJson(text)).toEqual({ context: 'a\nb"c' });
  });

  it("does not escape control characters outside strings", () => {
    expect(escapeRawControlCharsInStrings('{\n"a":1\n}')).toBe('{\n"a":1\n}');
  });

  it("still throws for json that cannot be repaired", () => {
    expect(() => parseLooseJson('{"a": ')).toThrow();
  });
});
