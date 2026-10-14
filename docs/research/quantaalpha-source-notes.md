# QuantaAlpha 源码精读笔记

对应 issue：#129（wayfinder 决策地图 #128 下的 research ticket）

- 源码：`git@github.com:QuantaAlpha/QuantaAlpha.git`，本次读到的 commit 为 `b7ceb27b1001261d7a95b209a963664ae1f8ab23`（main，提交日期 2026-06-29）。
- 论文：arXiv:2602.07085v3《QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining》（2026-05-18）。
- 读法：论文只用于补充解释，每条实现性论断以源码为准。凡论文与源码不一致处，专列一节（见第 9 节）。引用格式为 `路径:行号`，路径相对仓库根。

---

## 0. 全局链路（先看谁调用谁）

入口 `run.sh` -> `quantaalpha mine --direction ... --config_path configs/experiment.yaml` -> `quantaalpha/pipeline/factor_mining.py:main()`（514 行起）。`main()` 读配置后分三支：

- `evolution.enabled=true`：走 `run_evolution_loop()`（`quantaalpha/pipeline/factor_mining.py:318`）。
- planning 打开、无 evolution：按方向逐个跑 `AlphaAgentLoop`，可用 `multiprocessing` 并行（`factor_mining.py:615-645`）。
- 给了 `path`：`AlphaAgentLoop.load()` 续跑（`factor_mining.py:647`）。

进化模式下每一"轮任务"的执行单元是 `_run_evolution_task()`（`factor_mining.py:115`），它构造一个**全新的** `AlphaAgentLoop` 跑 `step_n = steps_per_loop = 5` 步，再把结果打包成 `StrategyTrajectory` 交给 `EvolutionController`。

五个步骤由 `LoopMeta` 从类方法自动收集（`quantaalpha/utils/workflow.py:24-65`，跳过下划线开头的方法），顺序为：

```
factor_propose -> factor_construct -> factor_calculate -> factor_backtest -> feedback
quantaalpha/pipeline/loop.py:141,153,162,172,186
```

对应的组件（由 `AlphaAgentFactorBasePropSetting` 的字符串类路径动态导入，`quantaalpha/pipeline/settings.py:48-58`）：

| 步骤 | 组件类 | 文件 |
|---|---|---|
| propose | `AlphaAgentHypothesisGen` | `quantaalpha/factors/proposal.py:202` |
| construct | `AlphaAgentHypothesis2FactorExpression` | `quantaalpha/factors/proposal.py:338` |
| calculate | `QlibFactorParser`（= `FactorParser`） | `quantaalpha/factors/coder/__init__.py:26` |
| backtest | `QlibFactorRunner` | `quantaalpha/factors/runner.py:37` |
| feedback | `AlphaAgentQlibFactorHypothesisExperiment2Feedback` | `quantaalpha/factors/feedback.py:215` |

---

## 1. 轨迹（trajectory）的数据结构

### 1.1 `StrategyTrajectory`

定义在 `quantaalpha/pipeline/evolution/trajectory.py:28-145`，是一个 `@dataclass`，字段如下（与源码注释一致）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `trajectory_id` | str | 12 位 md5 短 hash |
| `direction_id` | int | 所属规划方向编号（交叉轮用 crossover group 序号顶替） |
| `round_idx` | int | 轮序号，0=original，1+ 依次为 mutation/crossover |
| `phase` | `RoundPhase` | `original` / `mutation` / `crossover`，枚举见 `trajectory.py:21-25` |
| `hypothesis` | str | `str(hypothesis_obj)`，即 `AlphaAgentHypothesis.__str__` 的整段文本 |
| `hypothesis_details` | dict | 从 hypothesis 对象上白名单抓的字段，见下 |
| `factors` | list[dict] | 每个元素含 `name` / `expression` / `description` / 可选 `code` |
| `backtest_result` | Any | 原始回测结果（`pd.DataFrame`/`Series`） |
| `backtest_metrics` | dict[str, float\|None] | 归一化后的 7 个指标 |
| `feedback` | str | `str(feedback_obj)` |
| `feedback_details` | dict | `observations` / `hypothesis_evaluation` / `new_hypothesis` / `reason` / `decision` |
| `parent_ids` | list[str] | 变异/交叉的父轨迹 id |
| `created_at` | str | `datetime.now().isoformat()` |
| `extra_info` | dict | 预留，实际未被填充 |

`trajectory_id` 生成规则（`trajectory.py:83-88`）：

```python
content = f"{direction_id}_{round_idx}_{phase.value}_{ts}"
return hashlib.md5(content.encode()).hexdigest()[:12]
```

字段填充在 `EvolutionController.create_trajectory_from_loop_result()`（`controller.py:710-793`）：hypothesis 详情取属性 `hypothesis/reason/concise_reason/concise_observation/concise_justification/concise_knowledge`；factor 的 `code` 直接取 `workspace.code_dict["factor.py"]`（`controller.py:755-759`）；feedback 取 `observations/hypothesis_evaluation/new_hypothesis/reason/decision`。

指标归一化在 `_extract_metrics()`（`controller.py:795-859`），固定 7 键 `IC / ICIR / RankIC / RankICIR / annualized_return / information_ratio / max_drawdown`。它按候选名列表在 `result.index` 里查，候选名来自 Qlib 的回测输出，例如 `1day.excess_return_with_cost.annualized_return`（`controller.py:818-835`）。**这是硬编码的 Qlib 指标命名，见第 7 节。**

### 1.2 轨迹的持久化格式

`TrajectoryPool._save()`（`trajectory.py:341-358`）写单个 JSON：

```json
{
  "trajectories": { "<traj_id>": { ...to_dict()... } },
  "by_direction": { "0": ["<traj_id>", ...] },
  "by_phase": { "original": [...], "mutation": [...], "crossover": [...] },
  "saved_at": "2026-...T..."
}
```

`to_dict()`（`trajectory.py:131-137`）刻意把 `backtest_result` 置为 `None`（原始 DataFrame 不可 JSON 序列化），因此**轨迹池只保留了汇总指标，没保留逐日因子值**。因子值落在沙盒 workspace 的 `result.h5`，由 `FactorLibraryManager` 另建一份"因子库"JSON 并同步到 md5 命名的 pkl 缓存（`quantaalpha/factors/library.py:56-176`）。

落盘位置：`Path(logger.log_trace_path) / "trajectory_pool.json"`（`factor_mining.py:375`）。控制器状态另存 `evolution_state.json`（`factor_mining.py:498`，格式见 `controller.py:877-903`）。

### 1.3 第二套持久化：因子库

`AlphaAgentLoop.feedback()` 末尾无条件把本轮因子写入 `data/factorlib/all_factors_library_{suffix}.json`（文件名可由环境变量 `FACTOR_LIBRARY_SUFFIX` 定制），代码在 `quantaalpha/pipeline/loop.py:194-249`。条目结构见 `library.py:119-140`：

```
factor_id (md5(name+expr)[:16])
factor_name / factor_expression / factor_implementation_code
factor_description / factor_formulation
cache_location { workspace_suffix, workspace_path, factor_dir, result_h5_path }
metadata { experiment_id, round_number, evolution_phase, trajectory_id,
           parent_trajectory_ids, hypothesis, initial_direction,
           planning_direction, created_at }
backtest_results { 逐指标 }
feedback { observations, hypothesis_evaluation, decision, reason,
           new_hypothesis }
```

`metadata.evolution_phase` 与 `parent_trajectory_ids` 就是血统链，与 ASTER 架构文档里的 `lineage` 字段意图一致，但它是扁平字符串，没有 generation 计数。

---

## 2. 变异（mutation）

### 2.1 源码实际做法：整轨迹假设级重写

`MutationOperator`（`quantaalpha/pipeline/evolution/mutation.py:25-267`）只做一件事：把父轨迹的 hypothesis/factors/metrics/feedback 塞进一个 prompt，让 LLM 产出一个**与父轨迹正交的新假设**，返回四个字段 `new_hypothesis / exploration_direction / orthogonality_reason / expected_characteristics`（JSON 模式，解析见 `mutation.py:161-190`）。

它**没有**失败步骤定位，没有子轨迹切分，没有前缀冻结。全仓库检索 `localiz|suboptimal|failing step` 无命中。也就是说，论文 4.2.2 节写的"self-reflection 诊断出导致低终端奖励的决策节点 k，冻结前缀 s0..sk、只重写 ak 并续跑后续步骤"（式 6）在开源代码里没有对应实现。代码里最接近"自我反思"的东西是 backtest 之后的 feedback 步骤：`feedback.py:198-202` 解析 LLM 返回的 `Observations / Feedback for Hypothesis / New Hypothesis / Reasoning / Replace Best Result`，其中 `New Hypothesis` 是下一轮假设生成的输入。"定位"这件事被压缩成了一句自然语言建议。

### 2.2 判据与 prompt

system prompt 在 `quantaalpha/pipeline/prompts/evolution_prompts.yaml:10-21`，明确把 orthogonal 定义为四条：市场假设不同、数据维度/特征类型不同、投资逻辑不同、生成因子与父轨迹低相关。user prompt（同文件 23-64）要求输出 JSON 四字段。

`generate_mutation_prompt_suffix()`（`mutation.py:219-267`）把上面的结果拼成一段 Markdown 追加到"方向文本"后面，模板见 `evolution_prompts.yaml:75-95`：

```
## Mutation Round Guidance
### Parent Strategy Summary
{parent.to_summary_text()}
### Mutation Direction Suggestions
- New Hypothesis Direction: ...
- Exploration Dimension: ...
- Orthogonality Reasoning: ...
### Important Notes
1. 新假设必须与父轨迹正交 ...
```

这条 suffix 就是变异轮进入下游生成的唯一通道，注入点在 `AlphaAgentLoop.__init__`（`loop.py:106-111`）：

```python
effective_direction = potential_direction
if strategy_suffix:
    effective_direction = (potential_direction or "") + "\n" + strategy_suffix
```

### 2.3 "重写后如何续跑"

每个进化轮任务都是一次全新 `AlphaAgentLoop` + `step_n=5`（`factor_mining.py:167-183`），所以不存在"从某一步续跑"；变异轮的产物是一条完整的新轨迹。轨迹之间的关联只靠 `parent_ids` 与 `phase`。

断点续跑是另一套机制：`LoopBase.dump()` 每一步后 pickle 整个 loop 对象到 `__session__/{loop}/{step}_{name}`（`utils/workflow.py:140,147-151`），`AlphaAgentLoop.load()` 再 pickle 回来。这是 RD-Agent 血统的会话快照，与进化无关。

失败兜底：LLM 调用抛异常时用关键词匹配的硬编码模板（`mutation.py:192-217`，模板列表在 `evolution_prompts.yaml:98-104`，如父假设含 `momentum` 就返回"探索均值回复"）。

### 2.4 变异对象如何选出

`controller._prepare_mutation_targets()`（`controller.py:464-508`）：第一轮变异（`round_idx==1`）取 phase 为 `ORIGINAL` 的全部轨迹；之后取最近一次 crossover 轮的全部轨迹。按 `direction_id` 排序，逐个成为变异父本。注意 `TrajectoryPool.select_parents_for_mutation()`（`trajectory.py:231-249`，"同方向取最新"）**从未被调用**，是死代码。

---

## 3. 交叉（crossover）

### 3.1 源码实际做法：多父假设融合

`CrossoverOperator.generate_crossover()`（`quantaalpha/pipeline/evolution/crossover.py:126-190`）把 2 个以上父轨迹摘要拼进 prompt，让 LLM 输出 `hybrid_hypothesis / fusion_logic / innovation_points / expected_benefits`（JSON 解析见 `crossover.py:192-219`）。少于 2 个父本时直接返回第一个父本的假设文本（`crossover.py:146-148`）。

同样地，论文 4.2.2 节描述的"识别跨父轨迹一致贡献高累计奖励的片段（假设模板、构造模式、修复动作）并拼接成连贯子轨迹"（式 7）在代码里没有实现；代码层面重组的是自然语言假设，没有片段切分与拼接的数据结构。

### 3.2 亲本选择

选择逻辑有两套，实际生效的是后一套：

- `TrajectoryPool.select_parents_for_crossover()`（`trajectory.py:251-339`）：支持 `weighted/best/random` 三种策略，**死代码**。
- 生效路径：`controller._prepare_crossover_groups()`（`controller.py:510-539`）-> `CrossoverOperator.select_crossover_pairs()`（`crossover.py:303-373`）。

候选池的取法在 `_get_crossover_candidates()`（`controller.py:541-636`）：

- 首次交叉（尚无 crossover 轨迹）：`original` 全部 + 最近一轮 `mutation` 全部。
- 之后的交叉：最近一轮 `mutation` + 最近一轮 `crossover`。

即候选被限制在"最近两轮"，不会从历史深处取老轨迹。

选择流程（`crossover.py:303-373`）：

1. `_select_candidates_by_strategy()`（`crossover.py:375-442`）先按 `get_primary_metric()`（RankIC）降序，按策略取 `crossover_n * crossover_size` 个候选：
   - `best`：取 top-N（默认策略，`configs/experiment.yaml:89`）。
   - `random`：`random.sample`。
   - `weighted`：性能加权无放回采样，权重 `normalized + 0.1` 再归一（`crossover.py:444-508`）。
   - `weighted_inverse`：反向加权，`1 - normalized + 0.1`，注释写明"鼓励探索低性能轨迹"，标注参考 EvoControl。
   - `top_percent_plus_random`：top 30% 保底 + 其余随机补足。
2. 对候选做 `itertools.combinations(candidates, crossover_size)`。
3. `prefer_diverse=True` 时给组合打分（`crossover.py:351-367`）：

```python
directions = len(set(t.direction_id for t in combo))
phases     = len(set(t.phase       for t in combo))
avg_metric = sum(t.get_primary_metric() or 0 for t in combo) / len(combo)
score      = directions * 2 + phases + avg_metric
```

取分数最高的 `crossover_n` 组。即"跨方向"权重最高（x2），"跨 phase"次之，性能只作次要项。

切片参数默认值：`crossover_size=2`、`crossover_n=3`（`controller.py:43-44`）；`configs/experiment.yaml:68-70` 给的是 `2 / 2`；`docs/experiment_hyperparameters.md:89-90` 记录的实验口径是 `crossover_size=2`、`crossover_n=10`。三者不一致，见第 9 节。

### 3.3 重组产物如何变成后续示范

与变异完全同构：`generate_crossover_prompt_suffix()`（`crossover.py:242-301`）把 hybrid 结果拼成 `## Crossover Round Guidance` 段落（模板 `evolution_prompts.yaml:180-201`），作为 `strategy_suffix` 进入下一轮的 `effective_direction`。父轨迹摘要格式见 `_format_parent_summary()`（`crossover.py:75-124`）：phase 名、direction_id、假设截断 300 字、最多 3 条因子表达式各截 80 字、指标保留 4 位小数、feedback 截 200 字。

### 3.4 轮次编排

`EvolutionController` 的状态机（`controller.py:134-291`，并行版本 167-291，串行版本 134-165）：

```
round 0: original（每条规划方向一个任务）
round 1: mutation（父本 = round 0 的 original 轨迹）
round 2: crossover（父本 = round 0 original + round 1 mutation）
round 3: mutation（父本 = round 2 的 crossover 轨迹）
round 4: crossover（父本 = 最近 mutation + 最近 crossover）
... 直到 _current_round >= max_rounds
```

`mutation_enabled` / `crossover_enabled` 可单独关掉，退化成 mutation-only 或 crossover-only 循环（`controller.py:199-217, 259-289`）。分支数在首次交叉后由 `num_directions` 变成 `crossover_n`（注释 `controller.py:85-87`）。

---

## 4. 生成门控（一致性 / 复杂度 / 冗余）

门控实现集中在两处：`quantaalpha/factors/regulator/` 与 `quantaalpha/factors/proposal.py` 的重试循环。

### 4.1 一致性校验：LLM 评审，三步链式

`FactorConsistencyChecker.check_consistency()`（`quantaalpha/factors/regulator/consistency_checker.py:59-140`）把 hypothesis、factor_name、description、formulation(LaTeX)、expression、variables 一起交给 LLM，让它按三条链逐段判断：

1. hypothesis -> description
2. description -> formulation
3. formulation -> expression

prompt 在 `quantaalpha/factors/regulator/consistency_prompts.yaml:6-76`。**关键判据是 prompt 里的宽容条款**（`consistency_prompts.yaml:20-24`）：窗口大小差异（10 天 vs 15 天）与归一化差异可接受，只看核心逻辑与经济含义是否保持。

输出 JSON 字段：`is_consistent`、`severity`（`none/minor/major/critical`）、三段分析、`overall_feedback`、`corrected_expression`、`corrected_description`。

`check_and_correct()`（`consistency_checker.py:142-193`）最多重试 `max_correction_attempts` 次（默认 3，`configs/experiment.yaml:119`）。每轮优先采纳 `corrected_expression`，其次 `corrected_description`；`strict_mode` 打开时任何不一致直接放弃纠正。

放行判据在 `should_proceed_to_backtest()`（`consistency_checker.py:222-236`）：

```python
if result.is_consistent:      return True
if self.strict_mode:          return False
if severity in ["none","minor"]: return True
return False                   # major / critical 拦下
```

`FactorQualityGate.evaluate()`（`consistency_checker.py:385-454`）把三者串起来，若一致性检查给出了 `corrected_expression`，后续复杂度与冗余都用**纠正后**的表达式（`consistency_checker.py:422`）。一致性检查本身抛异常时按"通过"处理并记 note（`consistency_checker.py:131-140`），属于 fail-open。

### 4.2 复杂度约束：AST 统计，五个条件

`FactorRegulator.is_expression_acceptable()`（`quantaalpha/factors/regulator/factor_regulator.py:115-170`）返回五个条件的与：

| 条件 | 判据 | 默认阈值 | 出处 |
|---|---|---|---|
| cond1 | 与因子库的最大公共子树节点数 <= `duplication_threshold` | 8（`coder/config.py:33`） | `factor_regulator.py:131` |
| cond2 | `-log(1 - free_args_ratio) < 0.693`  等价于  自由参数占比 < 0.5 | 0.693（硬编码） | `factor_regulator.py:154-157` |
| cond3 | `-log(1 - unique_vars_ratio) < 0.693`  等价于  不同变量占比 < 0.5 | 0.693（硬编码） | `factor_regulator.py:159-160` |
| cond4 | `symbol_length <= symbol_length_threshold` | 300（`coder/config.py:37`） | `factor_regulator.py:162-163` |
| cond5 | `num_base_features <= base_features_threshold` | 6（`coder/config.py:41`） | `factor_regulator.py:165-167` |

统计量定义：

- `calculate_symbol_length` = `len(expr.strip())`（`factors/coder/factor_ast.py:482-493`）。
- `count_free_args` = AST 中 `NumberNode` 个数，即数值常量（`factor_ast.py:387-422`）。
- `count_unique_vars` = 名字以 `$` 开头的去重变量（`factor_ast.py:426-465`）。
- `count_all_nodes` = 全部节点数（`factor_ast.py:468-479, 540-560`）。
- `count_base_features` = 去重的 `$` 前缀原始字段（`factor_ast.py:496-537`）。

另有一套独立的 `ComplexityChecker`（`consistency_checker.py:239-305`），阈值默认 `symbol_length 250 / base_features 6 / free_args_ratio 0.5`。它只在 `FactorQualityGate` 里被调用。

### 4.3 冗余约束：AST 最大公共子树

`match_alphazoo()`（`factor_ast.py:370-383`）遍历因子库 CSV 的每一行表达式，对每个候选算最大公共子树并记录最大值：

- `find_largest_common_subtree()`（`factor_ast.py:278-360`）：枚举两棵树的所有子树，按 `get_subtree_size` 从大到小找第一个同构匹配。`are_subtrees_equal()` 对交换律算子（`+ * == != & && | ||`）会尝试左右互换（`factor_ast.py:310-330`）。
- 比较是**节点数**口径，不是字符或数值等价；`are_nodes_equal` 对 `ConditionalNode` 直接返回 True（`factor_ast.py:265-266`），意味着条件表达式之间的子树匹配偏宽松。
- 因子库来源 `FACTOR_COSTEER_SETTINGS.factor_zoo_path`，默认 `None`（`coder/config.py:28-31`）。为 `None` 时不加载 CSV（`factor_regulator.py:32-36`），`match_alphazoo` 在空 DataFrame 上循环 0 次，`duplicated_subtree_size` 恒为 0，**冗余检查自动通过**。

`RedundancyChecker.check()`（`consistency_checker.py:334-361`）先 `is_parsable()`，再比 `duplicated_subtree_size > duplication_threshold`（默认 5，`consistency_checker.py:314`）。

### 4.4 门控在生成流程里怎么被调用

核心循环在 `AlphaAgentHypothesis2FactorExpression._convert_with_history_limit()`（`quantaalpha/factors/proposal.py:409-567`）。它有一个 `while True` 的重生成循环（438-561）：

1. LLM 出 JSON，逐因子取 `expression / description / formulation / variables`。
2. `factor_regulator.is_parsable(expr)` 失败 -> `break` 重出（`proposal.py:461-463`）。
3. `factor_regulator.evaluate(expr)` 出统计字典（`proposal.py:465-467`）。
4. 若 `consistency_enabled` 且 `quality_gate` 非空 -> 跑一致性（`proposal.py:470-499`）；若一致性给了纠正表达式，替换后**重新 parse + 重新 evaluate**（`proposal.py:482-494`）。
5. `is_expression_acceptable(eval_dict)` 为假 -> 渲染 `expression_duplication` 模板追加到 user prompt 并 `break` 重出（`proposal.py:502-553`）。
6. 全部因子通过 -> `factor_regulator.add_factor()` 把新因子写进内存因子库（`proposal.py:564`），供本 loop 内后续候选查重。

`expression_duplication` 反馈模板在 `quantaalpha/factors/prompts/prompts.yaml:447-481`，分五段把失败原因回灌：重复子树、自由参数占比 >= 50%、不同变量占比 >= 50%、符号长度超限、基础特征超限。符号长度那一段措辞最重（含 `OVERFITTING ALERT`、要求"从头重新设计，目标 50-150 字符"）。

生成侧 prompt 里的硬约束写在 `factors/prompts/prompts.yaml:343-363`：符号长度 <= 250 字符（自称 HARD LIMIT）、基础特征 <= 6、目标长度 50-150 字符、避免深嵌套与复杂条件表达式。**注意 prompt 里的 250 与代码里的 300（`coder/config.py:37`）不一致。**

### 4.5 开关与阈值的实际来源（容易踩坑）

`QualityGate` 三开关来自 `configs/experiment.yaml:105-119` 的 `quality_gate` 段，经 `loop.py:95-100` 读取。默认 `consistency_enabled: false`、`complexity_enabled: true`、`redundancy_enabled: true`。但 `loop.py:114-116` 只把 `consistency_enabled` 传给了 `AlphaAgentHypothesis2FactorExpression`，复杂度与冗余开关没传。`proposal.py:352-366` 的 `quality_gate` 属性里写死 `complexity_enabled=True, redundancy_enabled=True`，并且**只有 `consistency_enabled=True` 才会构造**：

```python
if self._quality_gate is None and self.consistency_enabled:
    self._quality_gate = FactorQualityGate(
        consistency_enabled=self.consistency_enabled,
        complexity_enabled=True,
        redundancy_enabled=True,
    )
```

所以默认配置下的实际门控是"可解析 + `is_expression_acceptable` 五条件 + 超限回灌重生成"，一致性 LLM 评审、`ComplexityChecker`、`RedundancyChecker` 三类都不生效。这与论文图 3 的"三项门控都要开"的消融口径有出入。

另外 `configs/experiment.yaml:126-141` 的 `factor.complexity.*`（`symbol_length_threshold: 200`、`base_features_threshold: 5`、`duplication.threshold: 5`）以及 `factors_per_hypothesis: 1` **没有任何后端代码读取**（只有 `frontend-v2/backend/app.py:243` 往配置里写）。真实阈值来自 `FACTOR_COSTEER_SETTINGS`，可用环境变量 `FACTOR_CoSTEER_SYMBOL_LENGTH_THRESHOLD` 之类覆盖（`coder/config.py:8`，pydantic-settings 前缀 `FACTOR_CoSTEER_`）。

### 4.6 输出层相关性过滤：代码里被关掉了

论文 4.1.2 节末尾提到"在 2021 验证面板做输出级相关过滤，超过阈值就丢掉 RankIC 低的那一个"。对应实现是 `QlibFactorRunner.deduplicate_new_factors()`（`quantaalpha/factors/runner.py:58-73`），它算 SOTA 因子列与新因子列逐日 IC 的均值，去掉 `IC_max < 0.99` 之外的列。该分支的调用点被 `if False:` 注释掉（`factors/runner.py:138`）：

```python
if False: # SOTA_factor is not None and not SOTA_factor.empty:
    new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
```

即发布版不跑输出级相关过滤。

---

## 5. 多样化规划初始化

实现只有一个小文件 `quantaalpha/pipeline/planning.py`（117 行）。

`generate_parallel_directions()`（`planning.py:69-106`）流程：

1. 读 `quantaalpha/pipeline/prompts/planning_prompts.yaml`，format 出 system / user，并把 `output_format` 段拼到 user prompt 尾部。
2. `use_llm=True` 时调 LLM（`json_mode=False`，因为要求输出 ```json 代码块）。
3. `_parse_directions()`（`planning.py:38-48`）抽 JSON，要求 `directions` 是字符串数组且**长度 >= n**，不足则视为失败。
4. 失败最多重试 `max_attempts` 次（默认 5），每次往 system prompt 追加一句 `Strictly output valid JSON. No extra text.`（`planning.py:101`）。
5. 仍失败且 `allow_fallback=True` 时用 `_fallback_directions()`（`planning.py:51-66`）：把初始方向与 8 条硬编码模式模板循环拼接，模式包括短动量+量确认、滚动方差波动率状态、流动性/换手调整、截面 rank+行业中性、日内反转与隔夜漂移分解、基本面代理对齐、日历效应、下行波动率。

"如何保证互补"完全落在 prompt 上（`planning_prompts.yaml:2-21`）：system 要求"每条方向具体、可检验、与初始方向一致但在方法上正交"；user 要求"数据信号、时间尺度或推理框架必须不同"，并强制输出恰好 n 条字符串的 JSON。

配置项含义（`configs/experiment.yaml:10-22`，读取处 `factor_mining.py:336-370`）：

| 配置 | 含义 | 仓库默认 | 论文/文档口径 |
|---|---|---|---|
| `planning.enabled` | 是否展开多方向 | true | true |
| `planning.num_directions` | 并行方向数 | 2 | 10（论文附录 B；`docs/experiment_hyperparameters.md:57`） |
| `planning.max_attempts` | JSON 解析重试次数 | 5 | 5 |
| `planning.use_llm` | false 则只走模板兜底 | true | true |
| `planning.allow_fallback` | LLM 全失败时是否兜底 | true | true |
| `planning.prompt_file` | prompt 文件名 | planning_prompts.yaml | 同 |

论文 4.2.1 节说初始种子池取自 Alpha158(20) 子集并按低相关分组，这个分组动作在挖掘路径里找不到；`backtest/factor_loader.py:510-525` 里的 Alpha158(20) 只用于独立回测的基线因子组合。挖掘侧的方向生成就是上面这一次 LLM 调用。

`num_directions` 同时决定轨迹池的初始分支数与 `EvolutionConfig.num_directions`（`factor_mining.py:381`），并在原始轮生成 `direction_id = 0..n-1` 的任务（`controller.py:184-193`）。当 `planning.enabled` 但没给 `initial_direction` 时方向填 `None`（`factor_mining.py:366-367`），此时假设生成走"无方向、鼓励显著发散"的分支（`factors/proposal.py:220-221`）。

---

## 6. 轨迹池与示范注入

### 6.1 轨迹池的维护：无容量、无淘汰

`TrajectoryPool`（`trajectory.py:148-409`）内部就是三个字典加一个 JSON 落盘：

- `_trajectories: dict[str, StrategyTrajectory]`（按 id）
- `_by_direction: dict[int, list[str]]`
- `_by_phase: dict[RoundPhase, list[str]]`

`add()`（`trajectory.py:178-205`）每次写入都立刻 `_save()` 全量重写 JSON。**没有容量上限，没有淘汰/剪枝，没有按分数保留 top-k**。池子随轮数线性增长；`max_rounds` 通过限制轮数间接兜住规模。

生命周期开关：

- `fresh_start`（默认 true）为 true 时即使 `trajectory_pool.json` 存在也不加载，只记一条日志（`trajectory.py:175-176`）；`experiment.yaml:95` 也是 `true`，所以默认行为等价于"每次实验从空池开始"。
- `cleanup_on_finish`（默认 false）为 true 时实验结束删除池文件（`factor_mining.py:509-511`）。

读取接口：`get()` / `get_by_direction()` / `get_by_phase()` / `get_all()` / `get_latest_round_idx()` / `get_statistics()`（`trajectory.py:207-229, 384-392`）。`get_best_trajectories()` 在控制器上（`controller.py:865-875`），先筛 `is_successful()`（RankIC 非空且 > 0，`trajectory.py:94-97`）再按 RankIC 取 top-n，**只在收尾打日志时用**（`factor_mining.py:500-506`）。

### 6.2 示范怎么进上下文：两条通道，都不带原文

通道 A：**轮内 trace**。每个 `AlphaAgentLoop` 持有 `Trace`（`core/proposal.py:90-103`），`hist: list[(Hypothesis, Experiment, Feedback)]`，在 feedback 步骤 append（`loop.py:190`）。渲染时只取最近 `history_limit` 条：

```python
DEFAULT_HISTORY_LIMIT = 6   # factors/proposal.py:19
MIN_HISTORY_LIMIT = 1       # factors/proposal.py:20
limited_trace.hist = trace.hist[-history_limit:]
```
（`render_hypothesis_and_feedback()`，`factors/proposal.py:23-34`）

模板 `hypothesis_and_feedback` 在 `factors/prompts/prompts.yaml:8-17`，本意是 `trace.hist[-10:]`（硬编码 10，实际被上面切到 6），逐条输出：假设全文、代码（`experiment.sub_workspace_list[0].code_dict["model.py"]`，对因子场景这个键通常不存在，会渲染成空）、观察、评估、新假设、理由、decision。

通道 B：**跨轮 strategy_suffix**。变异/交叉轮把父轨迹压成 `to_summary_text()` 或 `_format_parent_summary()` 的短文本，拼到方向文本后面。截断额度（`trajectory.py:99-129` 与 `crossover.py:75-124`）：

| 位置 | 截断 |
|---|---|
| mutation: hypothesis | 500 字符 |
| mutation: factors | 最多 5 条，expression 各 100 字符 |
| mutation: feedback | 300 字符 |
| crossover: hypothesis | 300 字符 |
| crossover: factors | 最多 3 条，expression 各 80 字符 |
| crossover: feedback | 200 字符 |

注意跨轮通道只传假设/表达式/指标/反馈的摘要，**不传父轨迹代码，也不传逐日因子值**。父轨迹的 `backtest_result` 在 `to_dict()` 里已被清空，无法参与后续生成。

### 6.3 上下文预算控制

三处 token 预算逻辑，都是"超限就删最旧的一条"：

- 假设生成与因子生成：`is_input_length_error()` 识别 `input length / context length / max_tokens / too long` 等关键词，命中后把 `history_limit` 减 1 重试，最低到 1（`factors/proposal.py:37-50, 246-309, 392-407`）。
- 代码生成的 CoSTEER 循环：先 `build_messages_and_calculate_token` 估 token，超 `LLM_SETTINGS.chat_token_limit` 就依次删历史失败案例、删相似成功案例、删相似错误案例，最多调 10 次（`factors/coder/evolving_strategy.py:50-67, 121-174, 275-329`）。

### 6.4 轨迹池是写多读少的

`trajectory_pool.json` 全仓库只有 `factor_mining.py:375` 一处作为变量出现；`_load()` 只在 `fresh_start=False` 时被调用（`trajectory.py:173-174`）。也就是说它主要是运行记录/可分析产物，并非跨实验复用的知识库。真正"跨轮复用"的信息流是第 6.2 节的通道 B。

另外 `evolution_prompts.yaml:252-268` 里的 `trajectory_summary.template` 没有任何代码引用，是死配置；`orthogonality_check`（同文件 215-244，四维 1-10 分打分：数据维度/逻辑/时间尺度/市场状态）也没有调用点。交叉里的"互补"是靠 `score = directions*2 + phases + avg_metric` 的启发式，没有走 LLM 正交性打分。

---

## 7. 对 Qlib 的依赖与移植到 polars 栈的替换清单

### 7.1 依赖分三类

第一类：**Qlib 运行时**（数据、表达式引擎、数据集、模型、回测）

- `run.sh` 强制校验 `QLIB_DATA_DIR` 下存在 `calendars/ features/ instruments/`，并软链到 `~/.qlib/qlib_data/cn_data`。
- `quantaalpha/backtest/runner.py:38-51` `qlib.init(provider_uri, region)`；`:453-476` `from qlib.data import D` 取因子；`:333` 导入 `qlib.data.dataset.processor` 的 `Fillna/ProcessInf/CSRankNorm/DropnaLabel`；`:481-486` 导入 `LGBModel`、`qlib.workflow.R`、`SignalRecord/SigAnaRecord`、`qlib.backtest.backtest`、`qlib.contrib.evaluate.risk_analysis`；`:585` 调 `qlib_backtest`。
- `quantaalpha/backtest/factor_calculator.py:351-383` `QlibDataProvider` 也做 `qlib.init` + `D.features`。
- `quantaalpha/backtest/custom_factor_calculator.py:552-605` 同样从 `qlib.data.D` 拉行情。
- 因子实现沙盒的代码模板直接读 `daily_pv.h5`（`factors/coder/template.jinjia2`），该 HDF5 由 Qlib 数据离线导出（README 里说明可下载，也说明可从 Qlib 数据生成）。
- 回测配置 `configs/backtest.yaml` 里 label 写成 Qlib 表达式 `Ref($close, -2) / Ref($close, -1) - 1`（`:55`），处理器全是 Qlib class（`:57-80`），策略是 `qlib.contrib.strategy.TopkDropoutStrategy`（`:116-117`）。
- 因子实现的 Qlib 配置模板：`quantaalpha/factors/factor_template/conf_baseline.yaml`、`conf_combined_factors.yaml`，由 `workspace.py:17-27` 覆盖 rdagent 默认模板后灌进 workspace。
- 输入字段命名是 Qlib 风格 `$open/$close/$high/$low/$volume/$return`（`factors/prompts/prompts.yaml:421-426`）。

第二类：**RD-Agent 框架**（`rdagent==0.8.0`，`requirements.txt:57`）

- `quantaalpha/factors/experiment.py:9-20` 直接继承 `rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario / QlibFactorExperiment`，并用 `rdagent.utils.agent.tpl.T`（模板）与 `rdagent.log.rdagent_logger`。
- `quantaalpha/factors/workspace.py:11-12` 继承 `rdagent.scenarios.qlib.experiment.workspace.QlibFBWorkspace`。
- `quantaalpha/core/conf.py` 里的 `RD_AGENT_SETTINGS`（workspace 路径、pickle 缓存、file lock）是 RD-Agent 的设置对象；`factor_mining.py:206-210, 331-333` 直接改它的字段。
- 会话快照机制 `LoopBase.dump/load`（`utils/workflow.py`）也源自 RD-Agent。
- 更上游的 `quantaalpha/coder/costeer/`（CoSTEER 进化式代码生成、知识库、RAG 策略）整体是 RD-Agent 的子模块移植。

第三类：**与框架无关的纯逻辑**

轨迹/进化（`pipeline/evolution/*`）、规划（`planning.py`）、推理步骤编排（`pipeline/loop.py` 的五步 + `utils/workflow.py` 的 LoopBase）、AST 一致性与冗余工具（`factors/coder/factor_ast.py`）、表达式解析（`expr_parser.py`）、因子库管理（`factors/library.py`）、质量门控（`factors/regulator/*`）。

### 7.2 替换清单

| QuantaAlpha 依赖点 | 具体位置 | ASTER（polars + 自研评估）替换方案 |
|---|---|---|
| Qlib 行情/日历 | `run.sh` 校验、`qlib.init`、`qlib.data.D` | 已有 akshare -> Parquet + `quant/data/source/base.py` 协议，直接对接 |
| Qlib 表达式算子（`$open`、`Ref`、`Mean` 等） | `configs/backtest.yaml:55`、prompts 里的 `$` 前缀字段 | 换成 polars 列名 `open/close/...` 与自研算子库；AST 层可保留 `$` 作为"原始字段"约定 |
| Qlib 数据集处理器 `Fillna/ProcessInf/CSRankNorm/DropnaLabel` | `backtest/runner.py:333` | polars 实现：`fill_null`、`is_finite().not_().then(None)`、按 date 分组的 `rank()/zscore`、标签空值过滤 |
| `qlib.contrib.model.gbdt.LGBModel` | `backtest/runner.py:481` | AutoGluon Tabular（GPU），walk-forward 归 `quant/eval/model.py` |
| `qlib.backtest.backtest` + `TopkDropoutStrategy` + `exchange_kwargs` | `backtest/runner.py:585-600`、`backtest.yaml:114-133` | 自研 `quant/backtest/`（account/broker/fee/engine），A 股规则按架构文档 3.5 节，标签换 open-to-open |
| Qlib recorder / `R` / `SignalRecord` / `risk_analysis` | `backtest/runner.py:483-486` | 自研 `quant/eval/metrics.py` 输出 JSON，写入 runs 目录 |
| `daily_pv.h5` 沙盒数据（HDF5 + pandas） | `factors/coder/template.jinjia2`、`factors/runner.py:109-124` | 与 ASTER 现有 `tasks/factor-mining/TASK.md` 接口对齐：传 `pl.DataFrame` 10 列，输出 `(date, instrument, value)`；数据用只读 Parquet 挂载 |
| `FactorFBWorkspace.execute()` 用 subprocess 跑 `factor.py` + `result.h5` | `factors/coder/factor.py:104-226` | ASTER 已有 Docker 沙盒 + `eval.sh`，可保留"写文件 + 子进程执行"的形态，产物从 h5 换成 parquet 或内存回传 |
| `rdagent` 的 `QlibFactorScenario/Experiment/Workspace` | `factors/experiment.py`、`factors/workspace.py` | 自研 `Scenario/Experiment/FBWorkspace` 极简基类（字段加路径，几十行），或直接抛弃这层抽象，把 `Trajectory` 当唯一实验对象 |
| `RD_AGENT_SETTINGS`（workspace/pickle 缓存/file lock/multi_proc_n） | `core/conf.py`、`factors/runner.py:208-211` | 保留同名配置对象但自研实现；多进程评估可用 ASTER 的沙盒并行 |
| `pandarallel` 的 `groupby().parallel_apply()` | `factors/runner.py:64-70` | polars `group_by().agg()`，向量化后无需并行库 |
| 假设/反馈里硬编码 Qlib 指标名 | `pipeline/evolution/controller.py:813-836`、`factors/feedback.py:42-47, 88-93` | 换成 ASTER 指标名（rank_ic/icir/mono/turnover/max_corr），`_extract_metrics` 的候选名映射表要重写 |
| CoSTEER 代码生成 + 知识库 + RAG | `quantaalpha/coder/costeer/*`、`factors/coder/evolving_strategy.py` | 可整层丢掉：AST -> 模板编译成 `factor.py` 的路径已足够（`FactorParsingStrategy` 首次执行走的就是模板渲染，`evolving_strategy.py:249-254`） |
| `.env` 里的 Qlib 路径约定 | `run.sh`（`QLIB_DATA_DIR` 校验与 `~/.qlib/qlib_data/cn_data` 软链一段） | 换成本仓库的 data/ 与 runs/ 约定 |

### 7.3 可直接借鉴（框架无关）

1. **轨迹 dataclass 与 Phase 枚举**（`trajectory.py:21-145`）。字段设计（假设/表达式/代码/指标/反馈/父 id/轮次/阶段）与 ASTER 架构文档里的血统链 schema 几乎可一比一映射，改造点是补 `generation` 与因子 id，以及把 `backtest_result` 换成 runs 目录指针。
2. **进化轮次状态机**（`controller.py:134-636`）。original -> mutation -> crossover 交替、父本取自"最近两轮"、交叉后分支数从 `num_directions` 变成 `crossover_n`，这套编排与数据源无关，可直接搬。
3. **`strategy_suffix` 注入通道**（`loop.py:106-111` + `mutation.py:219-267` + `crossover.py:242-301`）。把父轨迹压成一段有结构的 Markdown 追加到方向文本，实现成本低，且天然适配 ASTER 的 Context/Proposal 双 agent（可作为 Context 灵感队列的一条灵感来源）。
4. **AST 一致性与复杂度工具**（`factors/coder/factor_ast.py` 全文件）。`pyparsing` 表达式 AST、最大公共子树、各类节点计数只依赖 pyparsing（文件顶部 `import pandas` 实际只被注释掉的 `__main__` 用到），与 Qlib 无关；因子库 CSV 的读取在 `factors/regulator/factor_regulator.py:32-36`，换 polars 的 `read_csv` 即可。若 ASTER 采用"因子中间表示 + 一致性门控"（决策地图上的下一张 ticket），这份实现可以直接移植。
5. **门控的"回灌重生成"模式**（`factors/proposal.py:438-561`）。失败原因渲染成结构化 checklist 追加到 user prompt 再让 LLM 重出，比单纯拒收更能提升一次通过率；`factors/prompts/prompts.yaml:447-481` 的模板可直接改写。
6. **反馈的 JSON 契约**（`factors/feedback.py:198-202`）：`Observations / Feedback for Hypothesis / New Hypothesis / Reasoning / Replace Best Result` 五字段，其中 `decision` 用于判断是否刷新 SOTA（`core/proposal.py:96-103` 的 `get_sota_hypothesis_and_experiment` 反向查找最近一条 decision 为真的轨迹）。ASTER 的 `score.json` 可对齐这一契约。
7. **两层冗余控制思路**：结构层 AST 子树匹配 + 功能层逐日因子值相关性。结构层在本仓库就是 `factor_ast`，功能层在 ASTER 已有 `max_corr` 折扣机制，两者互补。
8. **交叉组合打分**（`crossover.py:351-367`）：`directions*2 + phases + avg_metric`，一个很轻的"跨方向优先"启发式，可作为 ASTER inspiration-queue 方向配额的补充判据。
9. **上下文预算的降级阶梯**（`factors/proposal.py:246-309`、`evolving_strategy.py:121-174`）：按 token 估算逐条丢弃最旧历史，最后保底 `history_limit=1`。

### 7.4 不建议照搬

1. **`TrajectoryPool` 的样例**：无容量、无淘汰、每次 add 全量重写 JSON。规模上去后是 O(n) 写放大，ASTER 里应改成 append-only JSONL + 定期压缩，或直接 SQLite。
2. **轨迹池当知识库**：实际代码里池子是只写不读的（只用来选父本，且 `fresh_start` 默认清空）。示范注入完全靠 `strategy_suffix` 文本，没有任何机制让被验证过的表达式/代码进入后续生成。这与论文"demonstrations provide imitation learning priors"的说法差距最大。
3. **一致性的 fail-open**：检查异常时按通过处理（`consistency_checker.py:131-140`），冗余检查异常也按通过处理（`consistency_checker.py:359-361`）。ASTER 更应 fail-closed，或至少把 fail-open 记进 score.json 的 notes。
4. **阈值就地硬编码且多处不一致**：prompt 说 250、`FactorRegulator` 默认 300、`ComplexityChecker` 默认 250、`configs/experiment.yaml` 写 200，而配置文件那一份根本没被读。ASTER 应把阈值收敛到单一配置源并写测试。
5. **`if False:` 关掉的输出级去重**（`factors/runner.py:138`）：要么实现，要么从文档里删掉，否则文档与行为长期不一致。
6. **死代码与死配置**：`TrajectoryPool.select_parents_for_mutation/select_parents_for_crossover`、`evolution_prompts.yaml` 的 `orthogonality_check` 与 `trajectory_summary`、`configs/experiment.yaml` 的 `factor.*` 全段、`complexity_enabled/redundancy_enabled` 开关。移植时不要把这些当作"存在即合理"的设计信号。

---

## 8. 论文与源码的差异（供"以源码为准"时引用）

| # | 论文（arXiv:2602.07085v3）说法 | 源码实际情况 |
|---|---|---|
| 1 | 4.2.2 节：变异先 self-reflection 定位导致低奖励的决策节点 k，冻结前缀、只重写 ak 并续跑（式 6） | 无实现。`MutationOperator` 只做整轨迹假设重写，`parent_ids` 记录父子关系，无节点定位、无片段切分、无前缀冻结 |
| 2 | 4.2.2 节：交叉识别跨父轨迹的"高累计奖励片段"，拼接成连贯子轨迹（式 7） | 无实现。`generate_crossover()` 只让 LLM 输出一段混合假设文本 |
| 3 | 4.2.2 节：变异/交叉产物作为 demonstrations 提供模仿学习先验 | 无实现。产物只以 `strategy_suffix` 的短文本进入下一轮方向，父轨迹代码与因子值不进上下文 |
| 4 | 4.1.2 节：输出级相关过滤，ABS 相关超阈丢低 RankIC 者 | `deduplicate_new_factors()` 存在，调用点被 `if False:` 关闭（`factors/runner.py:138`） |
| 5 | 4.1.2 节：一致性校验覆盖"假设-描述-表达式"与"表达式-代码"两组对齐 | 代码只做假设->描述->公式->表达式三段（`consistency_prompts.yaml:16-18`），表达式与代码之间没有校验器；代码由确定性 Jinja 模板渲染（`template.jinjia2`），仅在执行报错时让 LLM 重写表达式（`evolving_strategy.py:331-350`） |
| 6 | 附录 B：10 条并行方向、5 次迭代、每假设 3 个因子、符号长度 <= 250、基础特征 <= 6、自由参数比 < 50% | 仓库 `configs/experiment.yaml` 给的是 2 条方向、`max_rounds=3`、`factors_per_hypothesis=1`；符号长度代码默认 300；`factors_per_hypothesis` 无代码读取。论文口径与 `docs/experiment_hyperparameters.md`（10/11/3/250）一致，仓库那份配置是缩减版 |
| 7 | 4.2.1 节：初始种子池来自 Alpha158(20) 子集并按低相关分组 | 挖掘路径无此步骤；Alpha158(20) 只出现在独立回测（`backtest/factor_loader.py:510-525`），planning 就是一次 LLM 调用 |
| 8 | 图 3 消融：一致性、复杂度、冗余三项门控都要开 | 默认配置下三项门控都不生效（`consistency_enabled=false` 且 `quality_gate` 需一致性开关才构造）；实际生效的是 `is_expression_acceptable` 的五个条件 |

---

## 9. 关键文件索引（便于后续 ticket 引用）

| 主题 | 文件 | 关键位置 |
|---|---|---|
| 运行入口 | `quantaalpha/pipeline/factor_mining.py` | `main()` 514；`run_evolution_loop()` 318；`_run_evolution_task()` 115 |
| 五步循环 | `quantaalpha/pipeline/loop.py` | `AlphaAgentLoop` 50；五步 141-192；`_get_trajectory_data()` 251 |
| 步骤收集 | `quantaalpha/utils/workflow.py` | `LoopMeta` 24-65；`LoopBase.run()` 90-145 |
| 轨迹与池 | `quantaalpha/pipeline/evolution/trajectory.py` | dataclass 28-145；池 148-409 |
| 变异 | `quantaalpha/pipeline/evolution/mutation.py` | `generate_mutation()` 76；suffix 219 |
| 交叉 | `quantaalpha/pipeline/evolution/crossover.py` | `generate_crossover()` 126；`select_crossover_pairs()` 303；加权采样 444 |
| 轮次编排 | `quantaalpha/pipeline/evolution/controller.py` | `EvolutionConfig` 24；状态机 134-291；父本准备 464-636 |
| 进化 prompt | `quantaalpha/pipeline/prompts/evolution_prompts.yaml` | mutation 9-104；crossover 112-208；死配置 215-268 |
| 规划 | `quantaalpha/pipeline/planning.py` | 69-106 |
| 规划 prompt | `quantaalpha/pipeline/prompts/planning_prompts.yaml` | 全文 22 行 |
| 假设/表达式生成 | `quantaalpha/factors/proposal.py` | `AlphaAgentHypothesisGen` 202；`AlphaAgentHypothesis2FactorExpression` 338；门控循环 409-567 |
| 生成 prompt | `quantaalpha/factors/prompts/prompts.yaml` | 方向转换 1-5；历史渲染 8-17；输出格式 130-179；复杂度硬约束 343-363；重复反馈 447-481 |
| 反馈 | `quantaalpha/factors/feedback.py` | `AlphaAgentQlibFactorHypothesisExperiment2Feedback` 215-339 |
| 一致性门控 | `quantaalpha/factors/regulator/consistency_checker.py` | 一致性 43-236；复杂度 239-305；冗余 308-361；`FactorQualityGate` 364-454 |
| 一致性 prompt | `quantaalpha/factors/regulator/consistency_prompts.yaml` | 6-76；纠正模板 84-145 |
| 复杂度/冗余规则 | `quantaalpha/factors/regulator/factor_regulator.py` | `evaluate()` 61；`is_expression_acceptable()` 115 |
| AST 工具 | `quantaalpha/factors/coder/factor_ast.py` | 解析 239；子树 278-383；统计计数 387-560 |
| 表达式编译 | `quantaalpha/factors/coder/expr_parser.py` | 344-353；模板 `template.jinjia2` |
| 阈值设置 | `quantaalpha/factors/coder/config.py` | 7-44 |
| 因子库 | `quantaalpha/factors/library.py` | 25-150 |
| 因子沙盒执行 | `quantaalpha/factors/coder/factor.py` | `FactorFBWorkspace.execute()` 104-226 |
| 回测（Qlib 绑定） | `quantaalpha/backtest/runner.py` | `qlib.init` 38-51；数据集 145-333；回测 585 |
| 配置 | `configs/experiment.yaml` | planning 10-22；execution 29-43；evolution 51-98；quality_gate 105-119；factor（未被读取）126-141 |
| 独立回测配置 | `configs/backtest.yaml` | label 55；segments 82-85；model 91-108；backtest 114-133 |
| 超参文档（与仓库配置不一致） | `docs/experiment_hyperparameters.md` | 57、88-90、126、132 |

