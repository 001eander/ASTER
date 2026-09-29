import type { Score } from "./experience-bank.js";

export const DEFAULT_PROPOSAL_REWRITES = 2;

export function needsProposalRewrite(outcome: {
  writeError?: string;
  ok?: boolean;
  score: Score | null;
}): boolean {
  if (outcome.writeError) return true;
  if (outcome.ok !== true) return true;
  return outcome.score === null;
}

export function parseProposalRewrites(text: string | undefined): number {
  if (text === undefined || text === "") return DEFAULT_PROPOSAL_REWRITES;
  const n = Number(text);
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`invalid --rewrites ${text}`);
  }
  return n;
}
