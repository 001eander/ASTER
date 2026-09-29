# factor_library — 因子库

每个因子一个 `.py` 文件，文件名即因子名。因子实现统一接口：

```python
def compute(data: pl.DataFrame) -> pl.DataFrame:
    """输入 DAILY_BARS 前 10 列，输出 (date, instrument, value)。"""
```

约定（AGENTS.md 量化纪律第 3 条）：

- 禁止网络 / 文件 IO，禁止随机性（同输入两次调用结果必须一致）。
- 只能使用每行 `date` 当天及之前的同票数据；rolling / shift 窗口天然满足。
- 价格类因子用 `close × adjfactor` 等复权价；量额类因子不复权（同日内比值因子
  的复权因子自动约去）。
- 输出 `(date, instrument, value)` 三列，缺失用 null，不做截面标准化。

本期为 Alpha158 风格手工基线因子集（M1 基线 + 给 Agent 的示范）。

## 注册表 registry.json

`registry.json` 是因子集与元数据的唯一真源，把此前的「目录即真源」换成
「registry 即真源」：因子库的成员、经济假设、评估指标、方向标签与血统链都记在
这份文件里，不再靠扫描目录隐式决定。

`quant/daily/pipeline.discover_factors` 在注册表存在时，只加载 `status == "pool"`
且 `code_path` 指向的 `.py` 存在的条目；graveyard 因子留在库里但不参与建模。
注册表缺失才回退为扫描目录（兼容旧跑批）。读写与校验见 `quant/factor_lib/`。

顶层 `{"version": 1, "factors": [...]}`，条目字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `factor_id` | str | 因子唯一标识，种子因子取文件名 stem |
| `hypothesis` | str | 经济假设一句话 |
| `code_path` | str | 相对仓库根的 `.py` 路径 |
| `metrics` | dict | `rank_ic` / `icir` / `max_corr` 等，未评估填 `null` |
| `direction` | dict | `signal_source` / `time_scale` / `mechanism`，均为非空字符串 |
| `lineage` | dict | `op` / `parents` / `run_id`（可空）/ `generation` |
| `status` | str | `pool`（参与建模）或 `graveyard`（淘汰，留库不建模） |

`direction` 为依据因子类别做的 best-effort 结构化标签：`signal_source` 取
`price` / `volume` / `price_volume`；`time_scale` 按最长窗口推断，`short` ≤ 10、
`medium` ≤ 30、`long` > 30；`mechanism` 记经济机制，目前用到 `momentum` /
`reversal` / `volatility` / `ma_bias` / `volume_ratio` / `price_volume_corr` /
`liquidity` / `range` / `vwap_bias`。

`lineage.op` 取 `seed` / `mutation` / `crossover`。种子因子为
`{"op": "seed", "parents": [], "run_id": null, "generation": 0}`，自动挖掘产出的
因子填写父因子 id、产出 run 与代数。

## 定期整库

注册表只增不减，同一信号的换皮变体会稀释建模数据集。整库把「池内留谁」写成可复算的
规则，代码在 `quant/factor_lib/prune.py`，按需手动跑，不挂在 daily pipeline 上：

```bash
# 先干跑，只打印计划不写盘
uv run python scripts/prune_factor_library.py --dry-run

# 只看某段窗口的相关性，确认后写回
uv run python scripts/prune_factor_library.py --start 2023-01-01 --end 2024-12-31
```

参数：`--data-dir`（默认 `data`）、`--factor-library-dir`（默认 `factor_library`）、
`--start` / `--end`（相关性窗口，默认全样本）、`--dry-run`。被降级的条目改写成
`status: graveyard`，留在库里保留血统，不再参与建模。

两条规则，先聚簇后容量：

1. **聚簇降级**：pool 因子两两算逐日截面相关的时间序列均值，`|corr| > 0.7`
   （复用入库查重阈值 `MAX_CORR_REJECT`，负相关同样入簇）的因子连边，连通分量为一个
   簇；每簇只留 `metrics.rank_ic` 最高的一个。`rank_ic` 为 null 视为最弱，并列依次比
   `icir`、`factor_id` 字典序，计划确定可复现。
2. **容量上限**：pool 因子数不超过库内总条目数（pool 加 graveyard）的一半，向下取整；
   聚簇后仍超限时，按同一质量排序继续降级最弱者。例：13 个种子因子、无 graveyard 时
   上限为 6，第一轮会降 7 个。

`rank_ic` 当前多为 null，容量规则会把「未评估」也按最弱处理，所以整库前应先用
`scripts/eval_baseline_factors.py` / `quant.eval.factor` 把 `metrics` 补上。

退出码：0 表示计划算出（含无需整库与干跑），1 表示读不到行情，2 表示 registry 缺失或
非法。规则幂等，整库后重跑计划为空。库内总条目数不足 2 时上限为 0（`floor(0.5)`），
脚本会额外提示，此时写盘会清空 pool。

## 因子清单

| 因子 | 类别 | 经济假设 | 窗口 | 使用字段 |
|---|---|---|---|---|
| `mom_5` | 动量 | 过去 5 日复权收盘涨幅短期延续 | 5 | close, adjfactor |
| `mom_20` | 动量 | 过去 20 日复权收盘涨幅中期延续 | 20 | close, adjfactor |
| `reversal_5` | 反转 | 5 日开盘涨幅过度反应后回落（开盘口径，区别于 mom） | 5 | open, adjfactor |
| `vol_ratio_5` | 量价 | 短期均量 / 长期均量放大代表关注度抬升 | 5 / 60 | volume |
| `turnover_proxy` | 量价 | 成交额相对自身近期均值放大代表换手活跃度上升 | 20 | amount |
| `price_volume_corr_10` | 量价 | 复权收盘与成交量的 10 日时序相关，度量量价配合 | 10 | close, adjfactor, volume |
| `volatility_20` | 波动率 | 20 日日收益标准差，高波动对应低风险调整收益 | 20 | close, adjfactor |
| `max_drawdown_20` | 波动率 | 相对 20 日高点的回撤，越深越看空 | 20 | close, adjfactor |
| `ma_bias_10` | 均线乖离 | 收盘对 10 日均线偏离，过大则回归 | 10 | close, adjfactor |
| `ma_bias_20` | 均线乖离 | 收盘对 20 日均线偏离，过大则回归 | 20 | close, adjfactor |
| `range_pct` | 其他 | 日内相对振幅的 10 日均值，度量震荡强度 | 10 | high, low, close |
| `vwap_bias` | 其他 | 收盘对当日 VWAP 的偏离，度量日内资金强弱 | 无 | close, vwap |
| `vol_ratio_5_20_neg` | 量能 | 5/20 日均量比取负，缩量做多（M2 内循环冒烟首个入库因子） | 5 / 20 | volume |

评估脚本见 `scripts/eval_baseline_factors.py`，测试见 `tests/test_baseline_factors.py`；
注册表与整库规则测试见 `tests/test_factor_lib.py` / `tests/test_factor_lib_prune.py`。
