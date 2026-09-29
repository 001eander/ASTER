import { describe, expect, it } from "vitest";
import { parseRoleModel } from "./role-model.js";

describe("parseRoleModel", () => {
  it("reads provider, id, and thinking level", () => {
    expect(parseRoleModel("deepseek/deepseek-v4-pro:max")).toEqual({
      provider: "deepseek",
      id: "deepseek-v4-pro",
      thinkingLevel: "max",
    });
    expect(parseRoleModel("deepseek/deepseek-v4-flash-vision-exp:high")).toEqual({
      provider: "deepseek",
      id: "deepseek-v4-flash-vision-exp",
      thinkingLevel: "high",
    });
  });
});
