"""滚动重训：模型版本注册表 + 到期判定 + 按需重训（issue #34）。

生产形态
--------
每日跑批不再固定加载一个静态模型，而是面向一个滚动注册表工作：注册表下按
训练截止日（``train_end``）分版本存放 predictor，信号日 T 跑批时先判定
最近版本是否到期（距 ``train_end`` 已满 ``retrain_every_days`` 个开市日），
到期则用截止 T 的最新数据重训一版，否则复用最近版本。重训周期与训练窗口
全部配置化（:class:`RollingConfig`）。

无前视
------
信号日 T 收盘后跑批时，T 日行情已可得，但 T 日样本的标签（T+1 开盘 →
T+1+horizon 开盘）尚未发生；训练集构造沿用 ``build_dataset`` 后丢弃
``label`` 为空的尾部行，因此实际用到的最后一个训练日 ``train_end`` 自然
落在 T 之前约 ``1 + horizon`` 个交易日，训练标签全部已兑现。

目录约定
--------
::

    runs/automl/rolling/
    ├── v_2026-09-25/              # AutoGluon predictor 目录（训练截止日命名）
    ├── v_2026-09-25.meta.json     # 同版本元信息（同级、后缀 .meta.json）
    └── ...
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from quant.automl.dataset import (
    DATE_COL,
    LABEL_COL,
    FactorCompute,
    build_dataset,
)
from quant.automl.trainer import (
    DEFAULT_PRESETS,
    DEFAULT_TIME_LIMIT,
    BaselineTrainer,
)
from quant.data.cache import load_bars, load_calendar
from quant.labels.open_to_open import DEFAULT_HORIZON
from quant.universe.members import filter_bars_to_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 重训周期（开市日数）：距最近版本 train_end 满该天数即重训。
DEFAULT_RETRAIN_EVERY_DAYS: int = 20

#: 训练窗口的开市日数。
DEFAULT_TRAIN_WINDOW_DAYS: int = 500

#: 因子 warmup 余量（开市日）：多加载这段历史供 rolling / shift 预热，
#: 构造数据集后裁掉，保证窗口首日的因子值不因预热不足而失真。
DEFAULT_WARMUP_DAYS: int = 60

#: 默认注册表目录（``runs/`` 不入 git）。
DEFAULT_REGISTRY_DIR: str = "runs/automl/rolling"

#: 版本目录前缀与元信息文件后缀。
VERSION_PREFIX: str = "v_"
META_SUFFIX: str = ".meta.json"

#: 训练集行数上限的默认值（0 = 不截断）。
DEFAULT_MAX_ROWS: int = 0

#: ``--max-rows`` 截断时的抽样随机种子，固定以保证可复现。
SAMPLING_SEED: int = 20260929


@dataclass(frozen=True)
class RollingConfig:
    """滚动重训配置。

    Attributes
    ----------
    retrain_every_days:
        重训周期（开市日）。
    train_window_days:
        每次重训使用的训练窗口（开市日）。
    warmup_days:
        训练窗口之前多加载的预热开市日数（构造数据集后裁掉）。
    horizon:
        标签持有期，透传 :func:`build_dataset`。
    presets / time_limit / use_gpu:
        AutoGluon 训练配方，透传 :class:`BaselineTrainer`。
    max_rows:
        训练集行数上限（固定种子抽样），0 表示不截断；配合 16GB 内存约束
        使用（issue #63）。
    """

    retrain_every_days: int = DEFAULT_RETRAIN_EVERY_DAYS
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS
    warmup_days: int = DEFAULT_WARMUP_DAYS
    horizon: int = DEFAULT_HORIZON
    presets: str = DEFAULT_PRESETS
    time_limit: float = DEFAULT_TIME_LIMIT
    use_gpu: bool | None = None
    max_rows: int = DEFAULT_MAX_ROWS

    def __post_init__(self) -> None:
        for name in ("retrain_every_days", "train_window_days", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是 >= 1 的整数，收到 {value!r}")
        if isinstance(self.warmup_days, bool) or not isinstance(self.warmup_days, int) \
                or self.warmup_days < 0:
            raise ValueError(f"warmup_days 必须是 >= 0 的整数，收到 {self.warmup_days!r}")
        if isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int) \
                or self.max_rows < 0:
            raise ValueError(f"max_rows 必须是 >= 0 的整数，收到 {self.max_rows!r}")


# ---------------------------------------------------------------------------
# 训练器协议与工厂
# ---------------------------------------------------------------------------


class RollingTrainer(Protocol):
    """滚动重训需要的训练器能力：训练 + 预测 + 落盘。"""

    def train(
        self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None
    ) -> Any:
        ...

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        ...

    def save(self) -> str:
        ...


#: 训练工厂：入参为版本目录，返回全新训练器。
TrainerFactory = Callable[[Path], RollingTrainer]

#: 加载器：入参为版本目录，返回可直接预测的训练器。
TrainerLoader = Callable[[Path], Any]


def default_trainer_factory(
    config: RollingConfig, feature_columns: list[str]
) -> TrainerFactory:
    """按配置构造 :class:`BaselineTrainer` 工厂。"""

    def factory(path: Path) -> BaselineTrainer:
        return BaselineTrainer(
            feature_columns=feature_columns,
            presets=config.presets,
            time_limit=config.time_limit,
            path=path,
            use_gpu=config.use_gpu,
        )

    return factory


# ---------------------------------------------------------------------------
# 版本注册表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMeta:
    """一个模型版本的元信息。"""

    train_start: date
    train_end: date
    n_rows: int
    feature_columns: list[str]
    presets: str
    time_limit: float
    horizon: int
    universe: str | None
    trained_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_rows": self.n_rows,
            "feature_columns": list(self.feature_columns),
            "presets": self.presets,
            "time_limit": self.time_limit,
            "horizon": self.horizon,
            "universe": self.universe,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelMeta:
        return cls(
            train_start=date.fromisoformat(str(data["train_start"])),
            train_end=date.fromisoformat(str(data["train_end"])),
            n_rows=int(data["n_rows"]),
            feature_columns=[str(c) for c in data["feature_columns"]],
            presets=str(data["presets"]),
            time_limit=float(data["time_limit"]),
            horizon=int(data["horizon"]),
            universe=None if data.get("universe") is None else str(data["universe"]),
            trained_at=str(data.get("trained_at", "")),
        )


def version_dir(registry_dir: str | Path, train_end: date) -> Path:
    """版本目录路径：``<registry>/v_<train_end>``。"""
    return Path(registry_dir) / f"{VERSION_PREFIX}{train_end.isoformat()}"


def meta_path(registry_dir: str | Path, train_end: date) -> Path:
    """元信息路径：版本目录同级、带 ``.meta.json`` 后缀。"""
    return Path(registry_dir) / f"{VERSION_PREFIX}{train_end.isoformat()}{META_SUFFIX}"


def list_versions(registry_dir: str | Path) -> list[date]:
    """列出注册表下全部版本的 train_end，升序。"""
    root = Path(registry_dir)
    if not root.is_dir():
        return []
    versions: list[date] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(VERSION_PREFIX):
            continue
        try:
            versions.append(date.fromisoformat(child.name[len(VERSION_PREFIX):]))
        except ValueError:
            continue
    return sorted(versions)


def latest_version(registry_dir: str | Path) -> date | None:
    """最近版本的 train_end；无版本返回 None。"""
    versions = list_versions(registry_dir)
    return versions[-1] if versions else None


def load_meta(registry_dir: str | Path, train_end: date) -> ModelMeta:
    """读取指定版本的元信息。"""
    path = meta_path(registry_dir, train_end)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"模型元信息 JSON 顶层应为对象：{path}")
    return ModelMeta.from_dict(raw)


def write_meta(registry_dir: str | Path, meta: ModelMeta) -> Path:
    """落盘元信息。"""
    path = meta_path(registry_dir, meta.train_end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# 到期判定（纯函数）
# ---------------------------------------------------------------------------


def retrain_due(
    open_days: list[date],
    latest_train_end: date | None,
    signal_day: date,
    retrain_every_days: int,
) -> bool:
    """是否到期重训。

    无历史版本时到期；否则统计 ``(latest_train_end, signal_day]`` 区间内的
    开市日数，达到 ``retrain_every_days`` 即到期。``signal_day`` 不晚于
    ``latest_train_end`` 时不到期（防御回退调用）。
    """
    if retrain_every_days < 1:
        raise ValueError(f"retrain_every_days 必须 >= 1，收到 {retrain_every_days}")
    if latest_train_end is None:
        return True
    if signal_day <= latest_train_end:
        return False
    elapsed = sum(1 for day in open_days if latest_train_end < day <= signal_day)
    return elapsed >= retrain_every_days


# ---------------------------------------------------------------------------
# 训练集构造
# ---------------------------------------------------------------------------


def build_rolling_trainset(
    data_dir: str | Path,
    signal_day: date,
    factors: Mapping[str, FactorCompute],
    config: RollingConfig,
    *,
    universe: str | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """构造一次重训的训练集，返回 ``(训练宽表, 特征列)``。

    加载 ``signal_day`` 之前 ``train_window_days + warmup_days`` 个开市日的
    行情，构造数据集后裁掉预热段，再丢弃 ``label`` 空（尾部未兑现）与全部
    特征空的行；``max_rows > 0`` 时固定种子抽样截断。
    """
    calendar = load_calendar(data_dir, end=signal_day)
    open_days = calendar.filter(pl.col("is_open"))[DATE_COL].to_list()
    if not open_days:
        raise ValueError(f"截至 {signal_day} 没有开市日：{data_dir}")
    n_load = min(len(open_days), config.train_window_days + config.warmup_days)
    load_start = open_days[-n_load]
    keep_start = open_days[max(0, len(open_days) - config.train_window_days)]

    bars = load_bars(data_dir, start=load_start, end=signal_day)
    if bars.height == 0:
        raise ValueError(f"行情窗口 [{load_start}, {signal_day}] 内没有数据")
    if universe is not None:
        bars = filter_bars_to_universe(bars, universe, data_dir=data_dir)
        if bars.height == 0:
            raise ValueError(f"股票池 {universe!r} 在窗口内没有行情")

    dataset = build_dataset(bars, factors, horizon=config.horizon)
    feature_columns = [
        col for col in dataset.columns
        if col not in (DATE_COL, "instrument", LABEL_COL, "delay_days")
    ]
    if not feature_columns:
        raise ValueError("数据集没有任何特征列")

    any_feature = pl.any_horizontal(*[pl.col(c).is_not_null() for c in feature_columns])
    kept = dataset.filter(
        (pl.col(DATE_COL) >= keep_start)
        & pl.col(LABEL_COL).is_not_null()
        & any_feature
    )
    if config.max_rows > 0 and kept.height > config.max_rows:
        kept = kept.sample(n=config.max_rows, seed=SAMPLING_SEED)
    kept = kept.sort("instrument", DATE_COL)
    if kept.height == 0:
        raise ValueError(
            f"训练窗口 [{keep_start}, {signal_day}] 内没有可训练行"
        )
    return kept, feature_columns


# ---------------------------------------------------------------------------
# 解析入口：到期重训 / 复用
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolveResult:
    """一次模型解析的产出。

    ``action`` 为 ``"retrained"``（本次新训）或 ``"reused"``（复用最近版本）。
    """

    trainer: Any
    action: str
    train_end: date
    meta: ModelMeta | None


def resolve_model(
    data_dir: str | Path,
    signal_day: date,
    factors: Mapping[str, FactorCompute],
    config: RollingConfig | None = None,
    *,
    registry_dir: str | Path = DEFAULT_REGISTRY_DIR,
    universe: str | None = None,
    trainer_factory: TrainerFactory | None = None,
    trainer_loader: TrainerLoader | None = None,
    force_retrain: bool = False,
    no_train: bool = False,
) -> ResolveResult:
    """解析信号日可用的模型：到期重训，否则复用最近版本。

    ``trainer_factory`` / ``trainer_loader`` 为依赖注入点（测试传假实现，
    跳过真实 AutoGluon）；缺省分别是 :class:`BaselineTrainer` 的构造与
    :meth:`BaselineTrainer.load`。

    ``no_train=True``（dry-run）时不允许重训：到期但注册表存在旧版本则
    复用旧版本，无版本可复用时抛 :class:`ValueError`。
    """
    cfg = config if config is not None else RollingConfig()
    registry_dir = Path(registry_dir)
    calendar = load_calendar(data_dir, end=signal_day)
    open_days = calendar.filter(pl.col("is_open"))[DATE_COL].to_list()

    latest = latest_version(registry_dir)
    due = force_retrain or retrain_due(
        open_days, latest, signal_day, cfg.retrain_every_days
    )
    if latest is not None and (not due or no_train):
        loader = (
            trainer_loader
            if trainer_loader is not None
            else lambda path: BaselineTrainer.load(path)
        )
        trainer = loader(version_dir(registry_dir, latest))
        logger.info("复用模型版本 %s（%s）", latest,
                    "dry-run 不重训" if due else "未到期")
        return ResolveResult(
            trainer=trainer,
            action="reused",
            train_end=latest,
            meta=None,
        )
    if no_train:
        raise ValueError(
            f"no_train 模式下无法重训，且注册表中没有可复用的版本：{registry_dir}"
        )

    train_df, feature_columns = build_rolling_trainset(
        data_dir, signal_day, factors, cfg, universe=universe
    )
    train_start = train_df[DATE_COL].min()
    train_end = train_df[DATE_COL].max()
    factory = (
        trainer_factory
        if trainer_factory is not None
        else default_trainer_factory(cfg, feature_columns)
    )
    path = version_dir(registry_dir, train_end)
    if path.exists():
        shutil.rmtree(path)
    trainer = factory(path)
    trainer.train(train_df)
    trainer.save()

    meta = ModelMeta(
        train_start=train_start,
        train_end=train_end,
        n_rows=train_df.height,
        feature_columns=feature_columns,
        presets=cfg.presets,
        time_limit=cfg.time_limit,
        horizon=cfg.horizon,
        universe=universe,
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )
    write_meta(registry_dir, meta)
    logger.info(
        "滚动重训完成：%s（%s ~ %s，%d 行）", path.name, train_start, train_end,
        train_df.height,
    )
    return ResolveResult(
        trainer=trainer, action="retrained", train_end=train_end, meta=meta
    )


__all__ = [
    "DEFAULT_REGISTRY_DIR",
    "DEFAULT_RETRAIN_EVERY_DAYS",
    "DEFAULT_TRAIN_WINDOW_DAYS",
    "DEFAULT_WARMUP_DAYS",
    "META_SUFFIX",
    "ModelMeta",
    "ResolveResult",
    "RollingConfig",
    "RollingTrainer",
    "TrainerFactory",
    "TrainerLoader",
    "VERSION_PREFIX",
    "build_rolling_trainset",
    "default_trainer_factory",
    "latest_version",
    "list_versions",
    "load_meta",
    "meta_path",
    "resolve_model",
    "retrain_due",
    "version_dir",
    "write_meta",
]
