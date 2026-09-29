export class ResourceGates {
  readonly maxProposals: number;
  readonly maxSandboxes: number;
  private proposals = 0;
  private sandboxes = 0;
  private context = false;
  private sandboxWaiters: Array<() => void> = [];
  private proposalWaiters: Array<() => void> = [];

  constructor(opts: { maxProposals: number; maxSandboxes: number }) {
    if (opts.maxProposals < 1 || opts.maxSandboxes < 1) {
      throw new Error("need at least one proposal slot and one sandbox slot");
    }
    this.maxProposals = opts.maxProposals;
    this.maxSandboxes = opts.maxSandboxes;
  }

  tryStartProposal(): boolean {
    if (this.proposals >= this.maxProposals) return false;
    this.proposals += 1;
    return true;
  }

  async acquireProposal(): Promise<void> {
    if (this.tryStartProposal()) return;
    await new Promise<void>((resolve) => {
      this.proposalWaiters.push(resolve);
    });
  }

  finishProposal(): void {
    if (this.proposals === 0) throw new Error("no proposal slot to finish");
    const next = this.proposalWaiters.shift();
    if (next) {
      next();
      return;
    }
    this.proposals -= 1;
  }

  tryStartSandbox(): boolean {
    if (this.sandboxes >= this.maxSandboxes) return false;
    this.sandboxes += 1;
    return true;
  }

  async acquireSandbox(): Promise<void> {
    if (this.tryStartSandbox()) return;
    await new Promise<void>((resolve) => {
      this.sandboxWaiters.push(resolve);
    });
  }

  finishSandbox(): void {
    if (this.sandboxes === 0) throw new Error("no sandbox slot to finish");
    const next = this.sandboxWaiters.shift();
    if (next) {
      next();
      return;
    }
    this.sandboxes -= 1;
  }

  tryStartContext(): boolean {
    if (this.context) return false;
    this.context = true;
    return true;
  }

  finishContext(): void {
    if (!this.context) throw new Error("context is not running");
    this.context = false;
  }

  proposalsInFlight(): number {
    return this.proposals;
  }

  sandboxesInFlight(): number {
    return this.sandboxes;
  }

  contextInFlight(): boolean {
    return this.context;
  }
}
