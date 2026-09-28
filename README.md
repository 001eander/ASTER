# ASTER

A 股日频量化系统。架构与决策见 `AGENTS.md`。

## 数据抓取

首次全量抓取（默认 2015-01-01 至今）：

```bash
ASTER_NO_PROXY=1 uv run python scripts/fetch_data.py
```

调试时只抓少量证券：

```bash
ASTER_NO_PROXY=1 uv run python scripts/fetch_data.py --instruments 600519,300750 --start 2026-09-01
```

产物落在 `--data-dir`（默认 `data/`）：

```
data/
  calendar.parquet           # 交易日历
  instruments.parquet        # 证券信息（含退市股）
  corporate_actions.parquet  # 分红送转
  bars/YYYY.parquet          # 每自然年一个日线文件
  _manifest.json             # 抓取账本，记录每只票的进度
```

抓取支持断点续传：进程被杀后重跑同一条命令，只补未完成的缺口。

### 代理

`requests` 会读 Windows 注册表里的系统代理配置。本机系统代理可能损坏，设置
`ASTER_NO_PROXY=1` 时脚本会在导入网络库之前把 `NO_PROXY` 置为 `*`，绕过系统
代理直连。
