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

评估脚本见 `scripts/eval_baseline_factors.py`，测试见 `tests/test_baseline_factors.py`。
