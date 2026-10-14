# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

## Category roles

Every triaged issue carries exactly one category label:

- `bug`: something is broken
- `enhancement`: new feature or improvement

## Module labels

In addition to category and state, an issue may carry one or more module labels indicating which part of the system it touches:

- `data`: 数据层
- `factor`: 因子相关
- `backtest`: 回测与账户
- `portfolio`: 组合优化
- `automl`: 建模与训练
- `agent`: Agent 内循环
- `infra`: 基础设施与跑批

Module labels are informational only; they do not participate in the triage state machine.
