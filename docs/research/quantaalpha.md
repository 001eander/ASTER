# QuantaAlpha 调研：项目身份、架构、产物形态与接入面

Issue: 001eander/ASTER #143（决策地图 #142 下的 research ticket）。分支 `research/quantaalpha`，一次性研究分支，不开 PR、不合并。

读到的版本：QuantaAlpha `main` @ `b7ceb27b1001261d7a95b209a963664ae1f8ab23`（提交日 2026-06-29），论文 arXiv:2602.07085v3（2026-05-18）。实现性论断以仓库源码与 GitHub 元数据为准；论文只用于对照。既有笔记 `docs/research/quantaalpha-source-notes.md`（issue #129，分支 `research/quantaalpha-source`）已精读过轨迹数据结构、变异/交叉算子与生成门控的逐行实现，本文不重复那部分，只在涉及产物接缝时引用。

## 结论摘要

1. 项目身份唯一，即 `github.com/QuantaAlpha/QuantaAlpha`。定位是 LLM 驱动的自进化因子挖掘框架，代码基于微软 RD-Agent（MIT）与 Qlib（MIT）改造。仓库 2026-01-28 建，1595 star / 307 fork，未归档，但**最后提交停在 2026-06-29**，此后 29 个 open issue 里**没有一个得到维护者回复**。
2. **产物是「Qlib 式表达式字符串 + 该表达式的 pandas 计算脚本 + 单因子 JSON 库 + Qlib 回测指标」**。表达式 DSL 是它的主产物，Python 代码是模板生成的固定壳子，不携带因子逻辑。
3. **最窄接缝是「表达式 DSL -> polars compute」的转译器**：只从 QuantaAlpha 取候选表达式与血统元数据，评测全部由 ASTER 现有门控重跑。接它的沙盒执行与回测等于把评测权交出去，与地图 #142 「同一回测器下新旧实现可比」的验收标准冲突。
4. **风险等级高，卡点在自报指标的可信度**（跑起来不难，难在它的分数不能直接采信）：仓库无 LICENSE 文件（issue #30、#40 挂了三个月无人回）；因子挖掘侧的 fitness 与评测口径有多处前视与数据划分争议（issue #27、#31、#37，其中 #27 已被第三方复现证实）；论文报告的核心指标在 v1 到 v3 之间被大幅下修。
5. 因此「QuantaAlpha 替换整个因子生产层（hyra-pi TS 编排层 + agent 挖因子内循环）」这个原始表述**需要收窄为「替换候选生成环节」**。它替代的是 proposal 生成与内循环编排，评测裁判必须留在 ASTER 侧。

---

## 1 项目身份与维护状态

### 1.1 元数据

来源：`gh api repos/QuantaAlpha/QuantaAlpha`。

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/QuantaAlpha/QuantaAlpha |
| 建仓 | 2026-01-28 |
| 最后提交 | 2026-06-29（`b7ceb27`），之后再无提交 |
| star / fork | 1595 / 307 |
| open issue | 29 |
| archived | false |
| GitHub 识别到的 license | null（无 LICENSE 文件） |
| topics | code, codeagent, factor-mining, quantaalpha, self-evolving |
| 官网 | https://quantaalpha.com/ |
| 数据托管 | https://huggingface.co/datasets/QuantaAlpha/qlib_csi300 |

`pyproject.toml` 声明 `requires-python = ">=3.10"`，classifier 写 `License :: OSI Approved :: MIT License`，包名 `quantaalpha`，CLI 入口 `quantaalpha = quantaalpha.cli:app`。注意 `[project].authors` 写的是 `Ziyi Tang <tangzy27@mail2.sysu.edu.cn>`，与论文作者列表不一致，估计是上游内部仓库的作者信息残留。

### 1.2 同名候选甄别

在 `search/repositories?q=QuantaAlpha` 的结果里，除主候选外全部是同名派生或复现，没有发现第二个独立的量化 alpha 挖掘项目：

| 仓库 | 说明 |
|---|---|
| `QuantaAlpha/QuantaAlpha` | 主候选，本文对象 |
| `QuantaAlpha/QuantaAlpha-claw` | 同团队下一代 harness，README 标注 Coming Soon，未发布 |
| `QuantaAlpha/QuantaAlpha.github.io` | 团队主页源码 |
| `dawwl2026-lang/QuantEvolve` | 自述为「QuantaAlpha paper reproduction」 |
| `lucaluca-04/LLM-Alpha-Mining-QuantaAlpha` | fork，自述修了测试集数据泄露问题 |
| 其余 `*/QuantaAlpha*`、`*/quantaalpha` | 个人 fork 与备份，star 为 0 |

判断依据：主候选与论文（arXiv:2602.07085）、官网（quantaalpha.com）、HuggingFace 数据集互相印证，且 topic 带 `factor-mining`、`self-evolving`，与「量化 alpha 挖掘语境」完全吻合。因此调研对象唯一，不存在选错项目的风险。

### 1.3 License

**仓库根目录没有 LICENSE 文件**，GitHub API 返回 `license: null`。README 有 MIT badge，`pyproject.toml` 有 MIT classifier，`quantaalpha/backtest/README.md` 文末写「MIT License」。

第三方 issue #30（2026-07-29，「What license is this project under? README shows an MIT badge but the repo has no LICENSE file」）明确指出：badge 不构成授权，真正生效的是 LICENSE 文件文本，因此下游再分发与修改的条款不明。issue #40（2026-10-02，「Add a LICENSE file (your pyproject classifier declares MIT)」）重复了同一请求。**两条都零回复、仍 open。**

依赖侧 license 是清楚的：RD-Agent 与 Qlib 均为 MIT。

结论：本机自用不对外分发时风险低；一旦要开源本仓、把修改后的 QuantaAlpha 代码纳入版本库或商用，法律状态需要先解决。

### 1.4 维护状态

issue 时间线（`gh api repos/QuantaAlpha/QuantaAlpha/issues?state=all`）：

| issue | 日期 | 状态 | 回复数 | 主题 |
|---|---|---|---|---|
| #27 | 2026-05-23 | open | 1 | 数据泄露（train/valid/test 区间过滤失效） |
| #28 | 2026-06-10 | open | 0 | macOS 下数据链接失败，因子全部跑不起来 |
| #30 | 2026-07-29 | open | 0 | license 不明 |
| #31 | 2026-08-02 | open | 0 | 开盘价回测的涨跌停判定存在前视 |
| #37 | 2026-09-26 | open | 0 | RFC：研究数据角色、评测溯源与时间序列验证 |
| #39 | 2026-09-26 | open | 0 | 拒绝负 lag（前视算子） |
| #40 | 2026-10-02 | open | 0 | 补 LICENSE 文件 |

2026-05-23 之后的所有 issue 都没有维护者回复，代码提交停在 2026-06-29。项目处于「社区还在用（fork 307）、维护已停摆」的状态，未归档但事实上无响应。这对「用它替换生产层」是明确的减分项：出问题只能自己修。

---

## 2 架构与运行方式

### 2.1 链路

```
run.sh "<研究方向>"
  -> quantaalpha mine --direction ... --config_path configs/experiment.yaml
  -> quantaalpha/pipeline/factor_mining.py: main()
  -> 每个「轮任务」构造一个 AlphaAgentLoop，跑 5 步
     factor_propose -> factor_construct -> factor_calculate -> factor_backtest -> feedback
     (quantaalpha/pipeline/loop.py:141,153,162,172,186)
  -> feedback 步把本轮因子写入 data/factorlib/all_factors_library*.json
     (quantaalpha/pipeline/loop.py:224-247)
```

进化模式（`configs/experiment.yaml` 默认 `evolution.enabled: true`）下，轮次按 `original -> mutation -> crossover -> ...` 排，`max_rounds: 3`，由 `EvolutionController` 组装 `StrategyTrajectory` 并落 `trajectory_pool.json`。轨迹字段、变异/交叉的实现细节、生成门控在源码里的真实行为（例如论文写的「定位失败步骤、冻结前缀、只重写该步」在代码里没有对应实现）见既有笔记 `docs/research/quantaalpha-source-notes.md` 第 1、2 节，本文不复述。

`docs/user_guide.md` 给出的资源量级（`num_directions x max_rounds x factors_per_hypothesis`）：2x3x3 约 10 万 token、30 到 60 分钟；5x10x5 约 200 万 token、8 到 16 小时。

### 2.2 输入数据形态

QuantaAlpha 需要两类数据（`README_EN.md` 第 3 节）：

| 数据 | 用途 | 来源 |
|---|---|---|
| Qlib 行情目录 `cn_data`（含 calendars/features/instruments） | Qlib 初始化与回测 | HuggingFace `cn_data.zip`，493 MB，A 股 2016-2025 |
| `daily_pv.h5` | 因子挖掘时因子实现的输入 | HuggingFace `daily_pv.h5` 398 MB；debug 子集 1.41 MB |

`daily_pv.h5` 的构造在 `quantaalpha/factors/data_template/generate.py`：

```python
fields = ["$open", "$close", "$high", "$low", "$volume"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2015-01-01":]
data["$return"] = data.groupby(level=0)["$close"].pct_change().fillna(0)
data.to_hdf("./daily_pv.h5", key="data")
```

即：pandas DataFrame，两级索引 `(instrument, datetime)`，列只有 `$open/$close/$high/$low/$volume/$return` 六列，不提供 `vwap/amount/adjfactor/turn/pettm/pbmrq`（后几项在源码里被注释掉）。debug 子集只取前 100 只证券，正式挖掘走全量。

与 ASTER 的差距：ASTER 因子输入是 polars 长表 10 列 `date, instrument, open, high, low, close, vwap, volume, amount, adjfactor`，价格未复权、复权靠 `adjfactor`（`quant/factor_api/spec.py`）。字段映射不是一一对应，且**复权口径未在任何文档中声明**（见第 5 节）。

### 2.3 运行要求

安装与配置（`README_EN.md` Quick Start）：conda + Python 3.10，`pip install -e .` + `requirements.txt`，复制 `configs/.env.example` 为 `.env` 并填：

```
QLIB_DATA_DIR / DATA_RESULTS_DIR
OPENAI_API_KEY / OPENAI_BASE_URL / CHAT_MODEL / REASONING_MODEL
```

`requirements.txt` 的关键依赖：`rdagent==0.8.0`（钉死）、`pyqlib`（**未钉版本**）、`pandas>=1.5,<3.0`、`numpy>=1.24,<2.0`、`tables`（PyTables，HDF5）、`docker`（可选远程回测）、`azure-ai-formrecognizer`、`langchain-community`、`pymupdf`、`scikit-learn`、`pandarallel`、`tiktoken`。

要点：

- **必须联网调 LLM**，走 OpenAI 兼容接口，支持切换 DashScope / OpenAI 等，文档示例用 `deepseek-v3`，论文与 README 的旗舰结果是 GPT-5.2。
- **挖掘步骤内嵌 Qlib**：`factor_backtest` 步用 `QlibFactorRunner`，`feedback` 步从 Qlib 回测结果里抽指标。没有 Qlib 就只剩表达式生成，没有 fitness。
- 因子实现是**子进程执行**：把模板生成的 `factor.py` 写进 workspace，`subprocess.check_output(f"{python_bin} {code_path}", shell=True)`，超时默认 1200 秒（`quantaalpha/factors/coder/config.py`）。
- 可选 Web UI（`frontend-v2/`，React + FastAPI）与 Docker 回测；`use_local: true` 时全部本地跑。
- Windows 支持被主动修过（2026-02-11 两条提交「fix win」「fix: improve Windows platform compatibility」），数据链接在 Windows 用 `os.link` 硬链接（`quantaalpha/core/experiment.py`）。同类的 macOS 路径 bug 是 issue #28。详见第 5 节。

---

## 3 产物形态

### 3.1 因子表达式（主产物）

一个因子由四元组描述（`quantaalpha/factors/coder/factor.py` 的 `FactorTask`）：`factor_name`、`factor_description`、`factor_formulation`、`factor_expression`。

`factor_expression` 是 Qlib 风格的**字符串表达式**，变量以 `$` 前缀，函数为大写 DSL 调用。真实样例（README 与 `quantaalpha/backtest/factor_loader.py` 中的 Alpha158(20) 定义）：

```
($close-$open)/$open
$volume/Mean($volume, 20)
($high-$low)/Ref($close, 1)
RANK(TS_MEAN($return, 20) / (TS_STD($return, 20) + 1e-8))
```

DSL 的解析器是 `quantaalpha/factors/coder/expr_parser.py`（pyparsing）：算术四则、比较（编译成 `GT/LT/GE/LE/EQ/NE`）、逻辑 `&&/||`（编译成 `AND/OR`）、三元 `? :`（编译成 `WHERE`）、函数调用。算子集在 `function_lib.py`（约 60 个），分六类：

| 类别 | 算子 |
|---|---|
| 截面 | `RANK` `ZSCORE` `MEAN` `STD` `SKEW` `KURT` `MAX` `MIN` `MEDIAN` `SCALE` |
| 时序 | `DELTA` `DELAY` `TS_MEAN` `TS_STD` `TS_VAR` `TS_MAX` `TS_MIN` `TS_SUM` `TS_RANK` `TS_ZSCORE` `TS_CORR` `TS_COVARIANCE` `TS_QUANTILE` `TS_MAD` `TS_PCTCHANGE` `TS_ARGMAX` `TS_ARGMIN` `TS_SKEW` `TS_KURT` |
| 均线 | `SMA` `EMA` `WMA` `DECAYLINEAR` |
| 数学 | `LOG` `EXP` `SQRT` `POW` `INV` `ABS` `SIGN` `FLOOR` `PROD` `FLOOR` |
| 条件与逻辑 | `COUNT` `SUMIF` `FILTER` `WHERE` `AND` `OR` |
| 回归与技术指标 | `REGBETA` `REGRESI` `SEQUENCE` `RSI` `MACD` `BB_MIDDLE` `BB_UPPER` `BB_LOWER` |

语义细节需要注意：`RANK(cls)` / `MEAN(cls)` 等无窗口参数的算子是**截面**算子（按 datetime 分组），`TS_*` 才是时序窗口算子；算术运算在 pandas 上做索引对齐。前视方面，`DELAY` 显式拒绝负 p，但 `DELTA` 与 `TS_PCTCHANGE` **接受负 p 且不报错**（issue #39，仍 open），意味着表达式层面允许写出取未来值的因子。

### 3.2 生成的实现代码

`quantaalpha/factors/coder/template.jinjia2` 生成固定的 `factor.py`：

```python
def calculate_factor(expr: str, name: str):
    df = pd.read_hdf('./daily_pv.h5', key='data')
    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)
    for col in df.columns:
        expr = expr.replace(col[1:], f"df['{col}']")
    df[name] = eval(expr)
    result = df[name].astype(np.float64)
    result.to_hdf('result.h5', key='data')
```

也就是说 `factor_implementation_code` 里存的「实现」是这段模板加上 `expr = "..."` 的填值，因子逻辑仍然只在表达式串里。`eval` 直接作用于表达式，没有沙盒。

### 3.3 因子库 JSON（交付物）

挖掘完成后落 `data/factorlib/all_factors_library.json`（可用环境变量 `FACTOR_LIBRARY_SUFFIX` 区分实验）。结构见 `quantaalpha/factors/library.py`：

```json
{
  "metadata": {"created_at": "...", "last_updated": "...", "total_factors": 0, "version": "1.0"},
  "factors": {
    "<md5(factor_name + factor_expression)[:16]>": {
      "factor_id": "...",
      "factor_name": "...",
      "factor_expression": "RANK(TS_MEAN($return, 20) / (TS_STD($return, 20) + 1e-8))",
      "factor_implementation_code": "File: factor.py\n\n<模板全文>",
      "factor_description": "...",
      "factor_formulation": "...",
      "cache_location": {"workspace_suffix": "...", "workspace_path": "...", "factor_dir": "...", "result_h5_path": "..."},
      "metadata": {
        "experiment_id": "...", "round_number": 0, "evolution_phase": "original",
        "trajectory_id": "...", "parent_trajectory_ids": [],
        "hypothesis": "...", "initial_direction": "...", "planning_direction": "...",
        "created_at": "..."
      },
      "backtest_results": {"<Qlib 指标名>": 0.0},
      "feedback": {"observations": "...", "hypothesis_evaluation": "...", "decision": true, "reason": "...", "new_hypothesis": "..."}
    }
  }
}
```

三个要点：

- **血统元数据齐全**：`evolution_phase` + `trajectory_id` + `parent_trajectory_ids` 直接对应 ASTER registry 的 `lineage.op/parents/generation` 意图（`factor_library/registry.json`）。这是 QuantaAlpha 相对现有 hyra-pi 内循环的真实增益。
- **指标是 Qlib 命名的字典**，不是 ASTER 的指标键。挖掘侧 feedback 只挑四项给 LLM 看：`IC`、`1day.excess_return_without_cost.max_drawdown`、`.information_ratio`、`.annualized_return`（`quantaalpha/factors/feedback.py: process_results`）。这些是「LGB 模型 + TopkDropout 组合」的结果，混了模型层与组合层，与 ASTER 的纯因子级 `rank_ic_mean` 不同层。
- **逐日因子值不在 JSON 里**，在 workspace 的 `result.h5`，另有一份 md5 命名的 pkl 缓存（`data/results/factor_cache/<md5(expression)>.pkl`）。`FactorLibraryManager.check_cache_status()` / `warm_cache_from_json()` 可以据此判断缓存是否还在。

### 3.4 质量分级字段

独立回测工具（`quantaalpha/backtest/`）的文档给出带质量分级的图书馆格式：

```json
"metadata": {"classification_type": "quality", "category": "high_quality", "total_factors": 60},
"factors": {"...": {"quality": "high_quality", "backtest_metrics": {"IC": 0.0627, "ICIR": 0.639}}}
```

`configs/backtest.yaml` 有 `quality_filter: null  # high_quality | medium_quality | low_quality`，`factor_loader.py` 会按 `factor_info["quality"]` 过滤。

但**仓库代码里没有写 `quality` 字段的路径**（GitHub code search 命中的只有 `backtest/README.md` 与 `configs/backtest.yaml` 两处文档），而挖掘产出的 `library.py` 条目也不含该字段。分级看起来是仓库外的再加工产物（示例路径形如 `/path/to/factor_data/quality/high_quality_1.json`）。接入时不能指望拿到官方分级，需要自己按 ASTER 门控重算。

### 3.5 生成门控（与 ASTER 门的对照素材）

`configs/experiment.yaml` 的 `quality_gate` 三项（`consistency_enabled` / `complexity_enabled` / `redundancy_enabled`）在 `quantaalpha/pipeline/loop.py:95-97` 被消费。复杂度与冗余的实现在 `quantaalpha/factors/regulator/factor_regulator.py`，判据是表达式 AST 层面的五条：

| 条件 | 含义 | 默认阈值 |
|---|---|---|
| cond1 | 与 factor zoo 的最大重复子树大小 | `duplication_threshold = 8` |
| cond2 | 自由参数占比 `-log(1-ratio) < 0.693`，即 ratio < 0.5 | 0.5 |
| cond3 | 唯一变量占比同上 | 0.5 |
| cond4 | 符号长度 SL | `symbol_length_threshold = 300` |
| cond5 | 使用的基础特征数 ER | `base_features_threshold = 6` |

另有一致性门 `regulator/consistency_checker.py`（LLM 校验假设-描述-表达式-代码对齐，可开关）。

一个实现细节：`configs/experiment.yaml` 里 `factor.complexity` 写了 `symbol_length_threshold: 200` / `base_features_threshold: 5` / `free_args_ratio_threshold: 0.5`，但 `factor_mining.py` 只读取 `quality_gate` 段并透传（`quality_gate_cfg`），**没有读取 `factor.complexity`**，实际生效阈值来自 `FactorCoSTEERSettings` 的默认值 300/6/8 或环境变量 `FACTOR_CoSTEER_*`。该 YAML 段目前是死配置。

---

## 4 接入面

### 4.1 两侧对照

来源：ASTER 侧 `quant/factor_api/spec.py`、`quant/factor_api/complexity.py`、`quant/eval/factor.py`、`docs/architecture.md`、`docs/research/eval-diagnosis.md`。

| 维度 | ASTER | QuantaAlpha |
|---|---|---|
| 因子表示 | 每因子一个 `.py`，实现 `compute(pl.DataFrame) -> pl.DataFrame` | 一条 DSL 表达式字符串 + 模板生成的 `factor.py` |
| 输入 | polars 长表 10 列（未复权 + adjfactor） | pandas 两级索引 `(instrument, datetime)`，`$` 前缀 6 列 |
| 输出 | `(date, instrument, value)` 长表 | `result.h5`，pandas Series（key `data`） |
| 标签 | open-to-open，T+1 开盘 -> T+2 开盘 | Qlib `Ref($close, -2)/Ref($close, -1) - 1`，T+1 收盘 -> T+2 收盘 |
| 内循环 | TS 编排层（hyra-pi）proposal -> 沙盒 `eval.sh` -> score | Python 五步循环，`QlibFactorRunner` 出 fitness |
| 因子级指标 | `rank_ic_mean` / `icir` / `mono` / `turnover_mean` / `max_corr` | Qlib 指标名（IC 来自 LGB 合成信号），无单因子 IC 口径 |
| 前视防护 | 截断重算（按日截断重算对比，`quant/factor_api/truncation.py`） | 无截断检测；解释器允许负 lag |
| 复杂度门 | radon 圈复杂度 15 / AST 节点 500 / 字段数 10 | 表达式符号长度 / 基础特征数 / 自由参数占比 / AST 子树重复 |
| 冗余门 | 行为相关 `max_corr <= 0.7`，并按阈值连续折扣 score | AST 子树结构匹配（对 CSV factor zoo），与数据无关 |
| 入库 | `factor_library/registry.json` + `lineage` | `all_factors_library.json` + trajectory 血统 |

### 4.2 三种接缝候选

**候选 A（推荐）：文件接缝 + 表达式转译器。**

QuantaAlpha 只负责挖掘，产出 `all_factors_library.json`；ASTER 侧写一个 DSL 到 polars 的转译器，把 `factor_expression` 编译成 `factor_library/<factor_id>.py`（实现 `compute`），之后完全走现有管线：复杂度 -> schema -> 截断重算 -> 指标 -> 门控。血统从 `metadata.evolution_phase/parent_trajectory_ids` 映射进 registry 的 `lineage`。

优点：评测权完全留在 ASTER，满足地图 #142 的验收标准「同一回测器下新旧实现可比」；转译后是普通因子，三层回测与生产跑批无需特殊分支。
成本：需要写并测试一个算子级转译器（约 60 个算子），且必须为 `DELTA`/`TS_PCTCHANGE` 之类的负 lag 输入加拒绝规则。

**候选 B：值接缝（沙盒取值）。**

把 QuantaAlpha 的 workspace 当取值器：喂 `daily_pv.h5`、读 `result.h5`、转成 `(date, instrument, value)` 长表。

优点：不用重写算子语义，理论上数值与上游一致。
缺点：ASTER 的截断重算门要求「按日截断输入重算」，而 QuantaAlpha 的实现是「全量读 h5 + eval 写文件」，不满足 ASTER 三条禁令（禁文件 IO / 禁随机性 / 禁用 cutoff 之后数据），在不改上游代码的前提下做不了。只适合当对拍兜底，不适合当主接缝。

**候选 C：只存表达式，不落 `.py`。**

把表达式收进因子 spec，评估时由 ASTER 侧解释器求值。这本质是候选 A 的变体（省掉代码生成），是否可行取决于 #144「新因子 spec」是否允许表达式形态，以及 #132「因子中间表示」的去向。

### 4.3 与现有评估管线的兼容与冲突

兼容（可直接复用，不需要改语义）：

- **门控意图同构**：QuantaAlpha 的 Rank IC 门槛、冗余抑制、复杂度上限、假设-表达式-代码一致性，与 ASTER 现有四道门一一对应，ASTER 是更严的实现（多了截断重算与连续折扣）。转译后的因子可以无缝进 `quant/eval/factor.py`。
- **血统字段直接映射**：`evolution_phase` -> `lineage.op`，`parent_trajectory_ids` -> `lineage.parents`，`trajectory_id` -> `run_id`。ASTER registry 现有的扁平 `lineage` 足够装下，缺 `generation` 计数（可由 `round_number` 补）。
- **冗余门互补**：QuantaAlpha 的结构匹配能拦「换皮的表达式」，ASTER 的行为相关能拦「表达式不同但信号相同」。两者不互斥，可并存。

冲突（必须在 #144 或 #132 定掉）：

- **复杂度门的度量对象不同**。ASTER 的 `measure_complexity` 作用于 Python 源码（radon + AST），对 DSL 字符串无效。若走候选 A，转译产物的复杂度是转译器的复杂度，度量失真；若走候选 C，需要新增表达式级复杂度度量（恰好可以照搬 QuantaAlpha 的 SL / ER / 自由参数比）。这与 #132「因子中间表示与假设-表达式-代码一致性门控」直接相关。
- **数据口径少列且复权未知**。`daily_pv.h5` 没有 `vwap/amount/adjfactor`，价格是否复权未声明。转译器必须明确「ASTER 未复权价 -> QuantaAlpha 语义」的映射，否则跨除权日的 rolling 与收益率会偏。
- **标签口径不同**。QuantaAlpha 的 fitness 基于 `$close` 的 T+1 到 T+2 收盘收益，ASTER 统一到 T+1 开盘到 T+2 开盘。用 QuantaAlpha 的 IC 挑因子、再用 ASTER 的标签评估，两套口径不同源，不能直接比较。结论是 QuantaAlpha 的 `backtest_results` 只可当参考，不可当准入判据。
- **指标层级混淆**。QuantaAlpha 的 `IC` 来自 LGB 合成模型的 `SigAnaRecord`，属于模型级信号 IC；ASTER 的 `rank_ic_mean` 是单因子级。同名不同义，接进 registry 时必须改名（例如 `source_ic_qlib`）。
- **前视防护缺位**。上游解释器允许负 lag（#39），挖掘侧数据生成没有 end-time 边界（#37）。这两条决定了 ASTER 侧必须重跑截断重算，也决定了不能采信上游分数。`docs/research/eval-diagnosis.md` 已指出 ASTER 相对 QuantaAlpha 缺样本外分层与逐年 IC 报告，这部分能力要自己补，不能指望上游。
- **运行环境边界**。ASTER 禁 pandas，QuantaAlpha 全栈 pandas + Qlib + RD-Agent。进程/文件边界天然隔离（与地图 #142 已定的「接缝形态：进程/文件边界」一致），代价是要维护一份独立的 conda 环境与 493 MB Qlib 数据。

### 4.4 依赖关系

- 本报告的解空间给 #144（新因子 spec）提供了一个具体输入：因子表示是否允许 DSL 表达式形态，决定了接缝是候选 A 还是候选 C。
- 与 #149（hyra-pi TS 编排层的去留）相关：被替换掉的是 TS 内循环编排与 proposal 生成，ASTER 的 Python 执行层保持不动。
- 与 #131（轨迹与进化是否作为挖掘主干）相关：QuantaAlpha 的进化算子实现与论文描述有出入，见 `docs/research/quantaalpha-source-notes.md` 第 2 节。

---

## 5 风险与未知

### 5.1 已知硬风险

1. **没有 LICENSE 文件**（#30、#40 零回复）。自用可以，纳入本仓版本库或对外分发前必须先解决。
2. **维护停摆**：最后提交 2026-06-29，此后的 issue 全部零回复。所有 bug 只能自己修。
3. **上游自报指标可被显著质疑**：
   - #27（2026-05-23，第三方）指出 `quantaalpha/backtest/runner.py` 的 `isinstance(selector, tuple)` 拒绝了 YAML 里的 list，导致 `dataset.prepare("train")` 与 `prepare("valid")` 都返回全样本（含 test 段），训练集与早停验证集完全相同。报告者的复算结论是：真实 test IC 约 0.034，而跑出来的年化 46.4% / IR 3.87 属虚高。该 issue 仍 open。
   - #31（2026-08-02，第三方）指出 `deal_price: open` 配 `limit_threshold: 0.095` 时，Qlib 用当日完整涨幅 `$change` 判定涨跌停，属于用收盘后信息判定开盘可交易性；换用 point-in-time 涨跌停价后组合收益变化明显。该 issue 仍 open。
   - #37（2026-09-26，第三方 RFC）列出更系统的问题：挖掘侧数据生成不设 end-time 边界而初步评分止于 2021、因子值缓存只按表达式做键、评测报告是裸指标字典、负 lag 算子可通过解析。报告者用合成数据做了有界行为检验，明确指出这些不是端到端复现。
4. **论文数字在版本间被大幅下修**：arXiv v3 摘要与 README 是 IC 0.0472 / ARR 4.68% / MDD 11.80%（GPT-5.2，CSI 300，2022-2025）；而东方证券 2026-04-07 的研究报告（发布在项目官网 quantaalpha.com）描述 2 月版本时写的是「IC 达 0.1501，年化超额收益 27.75%，MDD 7.98%」。README 有一次提交是「Update README results and figures to match arXiv v2」。同一份工作的核心指标在 v1 与 v3 之间差约 3 倍，这本身说明评测口径脆弱。
5. **第三方复现结论保守**：东方证券用 `qwen3.5-plus` 跑 13 个挖掘方向、约 100 万 token，得 21 个有效因子，结论是「具备 alpha 雏形，IC/RankIC 在有效区间但 ICIR 偏低，扣费后年化超额有限、回撤较大」。同一份报告还指出上游把测试集 RankIC 当候选评价标准驱动搜索，测试集与最终评价区间高度重叠。

综合 3、4、5：**QuantaAlpha 的价值在候选生成与血统管理，不在评测**。它自带的分数不能进 ASTER 的准入判据。

### 5.2 需要试用才能确认

1. **Windows 能否跑通五步内循环**（不依赖 WSL/Docker）。已看到两条 Windows 兼容提交，数据链接用 `os.link` 硬链接（要求同卷）；但 `factor.py` 执行路径里 `env['PYTHONPATH'] = pythonpath + ':' + env['PYTHONPATH']` 用的是 POSIX 分隔符，且 `subprocess.check_output(f"{python_bin} {path}", shell=True)` 拼裸字符串，含空格路径下可能出问题；`python_bin` 默认 `"python"`。需要实跑验证。
2. **从 ASTER 行情面板构造 `daily_pv.h5` 的可行性**：字段映射、复权口径（上游未声明是否复权；未复权价在除权日会污染 `pct_change` 与比值标签）、是否必须复刻 100 只 debug 子集的划分。
3. **是否必须恢复 Qlib**：挖掘的 fitness 走 `QlibFactorRunner`。ASTER 架构已定「Qlib 完全移除」，接入意味着要么为挖掘单独装 Qlib（进程边界内自洽），要么替换 runner 但失去上游 fitness。这是接入方案里最大的工程量变量。
4. **表达式转译器的算子覆盖与数值一致性**：`function_lib.py` 里若干算子定义了非标准语义（`RANK` 是时序百分位 rank，而 `RANK(cls)` 又是截面 rank；`PERCENTILE` 与 `TS_QUANTILE` 参数顺序会自适应交换；`TS_CORR` 对 ndarray 走另一条分支）。转译必须逐算子对拍数值，`1e-12` 之类的常数与 `min_periods=1` 的预热行为都要对齐。
5. **负 lag 的实际出现频率**：需要扫一遍真实产出的 `all_factors_library.json`，确认 `DELTA(x, -n)` / `TS_PCTCHANGE(x, -n)` 是否出现，以及转译器拒绝后是否显著减少候选量。
6. **`quality` 分级字段的产出路径**：仓库内找不到生成代码，需确认是外部工具还是前端后端补写。
7. **LLM 成本与时延**：`.env` 的 `CHAT_MODEL`/`REASONING_MODEL` 决定成本；上游文档给的量级是单个实验 10 万到 200 万 token、半小时到 16 小时。需要一次小规模实测（单方向、`max_rounds=1`、debug 子集）拿真实数字。
8. **QuantaAlpha-claw 的动向**：同团队下一代 harness 宣称 Lead/Reviewer/Miner 蜂群、Skill 化算子、血统追踪与更严的 Walk-Forward，会改变产物形态。接入方案应把「表达式 + 血统 JSON」当稳定面，别绑死当前目录结构。

---

## 附：来源清单

上游仓库（读到 `b7ceb27`，2026-06-29）：

- 仓库主页与 README：https://github.com/QuantaAlpha/QuantaAlpha
- 用户指南（项目结构、配置、时间段、输出物）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/docs/user_guide.md
- `pyproject.toml`（license classifier、Python 版本、CLI 入口）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/pyproject.toml
- `requirements.txt`（rdagent/pyqlib/pandas 等依赖）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/requirements.txt
- `configs/experiment.yaml`（进化、生成门控、factor.complexity）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/configs/experiment.yaml
- `configs/backtest.yaml`（标签、分段、TopkDropout、费率）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/configs/backtest.yaml
- `quantaalpha/factors/library.py`（因子库 JSON schema）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/library.py
- `quantaalpha/factors/coder/template.jinjia2`（生成的 factor.py 模板）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/coder/template.jinjia2
- `quantaalpha/factors/coder/expr_parser.py`（DSL 语法）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/coder/expr_parser.py
- `quantaalpha/factors/coder/function_lib.py`（算子集）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/coder/function_lib.py
- `quantaalpha/factors/coder/config.py`（CoSTEER 阈值默认值）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/coder/config.py
- `quantaalpha/factors/coder/factor.py`（FactorTask、子进程执行）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/coder/factor.py
- `quantaalpha/factors/regulator/factor_regulator.py`（复杂度与冗余判据）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/regulator/factor_regulator.py
- `quantaalpha/factors/feedback.py`（进入 LLM 的四项指标）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/feedback.py
- `quantaalpha/factors/data_template/generate.py`（daily_pv.h5 构造）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/data_template/generate.py
- `quantaalpha/factors/factor_template/conf_baseline.yaml`（初步回测配置）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/factor_template/conf_baseline.yaml
- `quantaalpha/backtest/README.md`（自定义因子 JSON 与 quality 分级示例）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/backtest/README.md
- `quantaalpha/backtest/factor_loader.py`（Alpha158 表达式与 quality_filter）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/backtest/factor_loader.py
- `quantaalpha/core/experiment.py`（workspace 数据链接，Windows 用 os.link）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/core/experiment.py
- `quantaalpha/pipeline/loop.py`（五步与因子库落盘）：https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/pipeline/loop.py

上游 issue（均为 2026-10-16 时的 open 状态）：

- #27 数据泄露：https://github.com/QuantaAlpha/QuantaAlpha/issues/27
- #28 macOS 数据链接：https://github.com/QuantaAlpha/QuantaAlpha/issues/28
- #30 license 不明：https://github.com/QuantaAlpha/QuantaAlpha/issues/30
- #31 开盘价回测前视：https://github.com/QuantaAlpha/QuantaAlpha/issues/31
- #37 数据角色与评测溯源 RFC：https://github.com/QuantaAlpha/QuantaAlpha/issues/37
- #39 负 lag 算子：https://github.com/QuantaAlpha/QuantaAlpha/issues/39
- #40 补 LICENSE：https://github.com/QuantaAlpha/QuantaAlpha/issues/40

论文、数据与第三方材料：

- arXiv:2602.07085《QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining》（v1 2026-02-06，v3 2026-05-18）：https://arxiv.org/abs/2602.07085
- 官方数据仓：https://huggingface.co/datasets/QuantaAlpha/qlib_csi300
- 官网产品页：https://quantaalpha.com/product.html
- 东方证券研究报告《QuantaAlpha：用大模型做量化因子挖掘》（2026-04-07，官网托管 PDF）：https://quantaalpha.com/documents/orient-20260407-quantaalpha-factor-mining.pdf
- QuantaAlpha-claw（下一代 harness，未发布）：https://github.com/QuantaAlpha/QuantaAlpha-claw
- RD-Agent 与 Qlib 均为 MIT：https://github.com/microsoft/RD-Agent ，https://github.com/microsoft/qlib

ASTER 侧对照：

- `quant/factor_api/spec.py`（因子契约与输入列）
- `quant/factor_api/complexity.py`（圈复杂度 15 / AST 500 / 字段 10）
- `quant/eval/factor.py`（门控阈值与共线性折扣）
- `factor_library/registry.json`（现有 registry 与 lineage 字段）
- `docs/architecture.md`（数据层、标签口径、因子接口）
- `docs/research/quantaalpha-source-notes.md`（issue #129，轨迹与进化算子的源码精读）
- `docs/research/eval-diagnosis.md`（issue #133，现有评估体系失效模式）
