export const THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;
export type ThinkingLevelName = (typeof THINKING_LEVELS)[number];

export type RoleModel = {
  provider: string;
  id: string;
  thinkingLevel: ThinkingLevelName;
};

export const DEFAULT_CONTEXT_MODEL = "deepseek/deepseek-v4-pro:max";
export const DEFAULT_PROPOSAL_MODEL = "deepseek/deepseek-v4-flash-vision-exp:high";

export function parseRoleModel(text: string): RoleModel {
  const lastColon = text.lastIndexOf(":");
  const lastSlash = text.lastIndexOf("/");
  if (lastSlash <= 0 || lastColon <= lastSlash) {
    throw new Error(`模型写法应为 provider/id:thinking，例如 ${DEFAULT_CONTEXT_MODEL}`);
  }
  const thinkingLevel = text.slice(lastColon + 1);
  if (!THINKING_LEVELS.includes(thinkingLevel as ThinkingLevelName)) {
    throw new Error(`未知思考深度 ${thinkingLevel}，可用：${THINKING_LEVELS.join(", ")}`);
  }
  return {
    provider: text.slice(0, lastSlash),
    id: text.slice(lastSlash + 1, lastColon),
    thinkingLevel: thinkingLevel as ThinkingLevelName,
  };
}
