import { describe, expect, it } from "vitest";
import { withTimeout } from "./timeout.js";

describe("withTimeout", () => {
  it("returns the value when the work finishes in time", async () => {
    await expect(withTimeout(Promise.resolve(7), 200, "fast")).resolves.toBe(7);
  });

  it("rejects when the work does not finish in time", async () => {
    const hang = new Promise<number>(() => undefined);
    await expect(withTimeout(hang, 20, "proposal write")).rejects.toThrow(
      "proposal write timed out after 20ms",
    );
  });

  it("does not impose a deadline when the timeout is omitted", async () => {
    await expect(withTimeout(Promise.resolve(3), undefined, "proposal write")).resolves.toBe(3);
  });
});
