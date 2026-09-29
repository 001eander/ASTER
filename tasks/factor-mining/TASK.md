# 因子挖掘：A 股日频选股因子

## 任务

在当前方案目录写出 `factor.py`，实现一个 A 股日频选股因子。文件是单文件、可执行代码，入口函数签名固定：

```python
def compute(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

因子值越大表示越看多。评分由沙盒里的 `eval.sh` 完成，你只负责写出合法的因子实现。

## 接口契约

### 输入

`compute` 收到一张长表面板，恰好 10 列，已按 `(instrument, date)` 排序：

| 列 | 类型 | 含义与单位 |
|---|---|---|
| `date` | Date | 交易日 |
| `instrument` | String | 证券代码，形如 `600000.SH` |
| `open` `high` `low` `close` | Float64 | 未复权价，单位元 |
| `vwap` | Float64 | 当日成交均价，单位元 |
| `volume` | Float64 | 成交量，单位股 |
| `amount` | Float64 | 成交额，单位元 |
| `adjfactor` | Float64 | 后复权因子，`复权价 = 未复权价 × adjfactor` |

输入价格未复权，除权日的价格跳空会污染收益比较。凡是涉及价格比较与收益计算，用 `close * adjfactor` 这类后复权口径，不要直接比未复权价。

### 输出

返回恰好 `(date, instrument, value)` 三列的长表：

- `value` 为 Float64。
- 窗口不足等缺失情形用 null，不要填 0。
- 不做截面标准化，不留中间列。
- 行序不作要求。

## 三条禁令

1. 禁止网络与文件 IO。只允许基于传入的 DataFrame 计算，不读磁盘、不联网、不 import 数据源。
2. 禁止随机性。同一输入必须得到同一输出，不引入任何随机数。
3. 禁止使用当日之后的数据。`shift` 只能传正数，统计量只能用滚动窗口（`rolling_*`、`shift`、按 `over("instrument")` 的累积操作）。用全序列均值、排名、分位数回填历史会检出前视，直接零分。

## 写作要求

- 只依赖 polars（`import polars as pl`）。沙盒里没有 pandas。
- 全程向量化，禁止逐行循环。
- 窗口等参数提升为模块顶部常量；用模块级 `REQUIRED_COLUMNS: tuple[str, ...]` 列出依赖列。
- 模块 docstring 写清：经济假设、使用字段、窗口、复权口径。
- 可声明模块级 `WARMUP: int` 告知预热期（默认 60 个交易日），截断重算检测会避开这段起步期。
- 文件里是可执行代码，没有 TODO、省略号占位、假函数。
- 写完执行 `python3 -m py_compile factor.py` 做语法检查。

## 评分机制

`eval.sh` 调用 `quant.eval.factor` 管线，按顺序执行：

1. **复杂度**：圈复杂度 ≤ 15，AST 节点 ≤ 500。
2. **加载**：文件须能被加载为可用的 `compute`。
3. **schema 校验**：输入面板与因子输出的列、dtype、键唯一性。
4. **计算**：调用 `compute`。
5. **截断重算**：抽样检测日，把数据截断到该日重算，与全量结果对比。检出前视即硬失败。
6. **指标与行为查重**：RankIC 均值、ICIR、分层单调性、换手，以及与库内已入库因子（`status == "pool"`）的逐日截面相关均值 `max_corr`。

跑完全流程则写出 `/work/score.json`：`score = quality × (1 - max_corr)`（可负，越高越好）。`quality` 为折扣前的 `rank_ic_mean`，`max_corr` 取绝对值最大者（库内无因子可查时为 None，折扣记 1.0），共线性越强分数越低；`higher_is_better` 为 true，`notes` 与 `details.metrics` 把 `quality` 与 `corr_discount` 拆开写。硬失败（崩溃、schema 不合规、前视、复杂度超标）由退出码非 0 表达，不写 `score.json`。

## 过关线

Context 判断 `stop` 的依据是 `score.json` 里 `details.gate_passed == true`。这等价于同时满足：

- `rank_ic_mean ≥ 0.02`
- `icir ≥ 0.2`
- 分层单调性 `mono > 0`
- 与库内 pool 因子的行为相关性 `max_corr ≤ 0.7`（`|max_corr| > 0.7` 即判冗余拒绝；库内无因子可查时该项跳过）
- 截断重算与复杂度全部通过

五项缺一不可。注意共线性折扣已先行压低 `score`，`max_corr > 0.7` 的拒绝只是兜底，不存在「分数高却被拒」的隐藏规则。

## 数据

沙盒内 `/data` 为只读挂载的全市场日线，起始 2021-09-29。评估用全样本，不做样本外切分。

## 提示

- 因子值越大越看多，先想清楚信号方向。
- RankIC 稳定为负的因子把符号取反即可提升分数。
- 同一机制反复接近 0 说明信号没有信息，换机制比调参更有效。
