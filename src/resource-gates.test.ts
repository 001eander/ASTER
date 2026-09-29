import { describe, expect, it } from "vitest";
import { ResourceGates } from "./resource-gates.js";

describe("ResourceGates", () => {
  it("caps how many proposals can write at once, then frees a slot when one finishes", () => {
    const gates = new ResourceGates({ maxProposals: 2, maxSandboxes: 1 });
    expect(gates.tryStartProposal()).toBe(true);
    expect(gates.tryStartProposal()).toBe(true);
    expect(gates.tryStartProposal()).toBe(false);
    expect(gates.proposalsInFlight()).toBe(2);

    gates.finishProposal();
    expect(gates.proposalsInFlight()).toBe(1);
    expect(gates.tryStartProposal()).toBe(true);
    expect(gates.proposalsInFlight()).toBe(2);
  });

  it("counts sandbox slots separately from proposal slots", () => {
    const gates = new ResourceGates({ maxProposals: 1, maxSandboxes: 2 });
    expect(gates.tryStartProposal()).toBe(true);
    expect(gates.tryStartSandbox()).toBe(true);
    expect(gates.tryStartSandbox()).toBe(true);
    expect(gates.tryStartSandbox()).toBe(false);
    expect(gates.proposalsInFlight()).toBe(1);
    expect(gates.sandboxesInFlight()).toBe(2);
  });

  it("still lets Context start when every proposal slot is taken", () => {
    const gates = new ResourceGates({ maxProposals: 2, maxSandboxes: 2 });
    expect(gates.tryStartProposal()).toBe(true);
    expect(gates.tryStartProposal()).toBe(true);
    expect(gates.tryStartProposal()).toBe(false);

    expect(gates.tryStartContext()).toBe(true);
    expect(gates.contextInFlight()).toBe(true);
    expect(gates.tryStartContext()).toBe(false);

    gates.finishContext();
    expect(gates.contextInFlight()).toBe(false);
    expect(gates.tryStartContext()).toBe(true);
  });

  it("lets a rewrite wait for a proposal slot the same way a sandbox waits", async () => {
    const gates = new ResourceGates({ maxProposals: 1, maxSandboxes: 1 });
    expect(gates.tryStartProposal()).toBe(true);

    let acquired = false;
    const waiting = gates.acquireProposal().then(() => {
      acquired = true;
    });
    await Promise.resolve();
    expect(acquired).toBe(false);

    gates.finishProposal();
    await waiting;
    expect(acquired).toBe(true);
    expect(gates.proposalsInFlight()).toBe(1);

    gates.finishProposal();
    expect(gates.proposalsInFlight()).toBe(0);
  });
});
