"""
config.py
error_estimate_exp の設定（config/<cfg_name>/config.yaml）を読む。

閾値は time_aware_exp の ThresholdConfig をそのまま使う。
dataset / iou_threshold / integrate_way / thresholds は time_aware_exp の
build_integrator と BaseRunner が参照する属性名に合わせている。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import ERROR_ESTIMATE_EXP_ROOT
from src.time_aware_exp.config.config import ThresholdConfig

CONFIG_ROOT = ERROR_ESTIMATE_EXP_ROOT / "config"
# キャリブレーション結果は config ごとに置かず、ここに名前を付けて共有する
CALIBRATION_ROOT = ERROR_ESTIMATE_EXP_ROOT / "calibration"


def config_path(cfg_name: str) -> Path:
    """config/<cfg_name>/config.yaml。結果もこのディレクトリに保存される。"""
    return CONFIG_ROOT / cfg_name / "config.yaml"


def calibration_path(name: str) -> Path:
    """calibration/<name>.json。複数の config から同じ name を指定して使い回す。"""
    return CALIBRATION_ROOT / f"{name}.json"


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int
    min_hits: int
    iou_threshold: float


@dataclass(frozen=True)
class SelectionConfig:
    base_model: str              # 常に動く基本モデル
    candidates: tuple[str, ...]  # 第 2・第 3 モデルの候補

    @property
    def models(self) -> tuple[str, ...]:
        """基本モデルと候補（重複なし）。キャリブレーション対象になる。"""
        return tuple(dict.fromkeys((self.base_model, *self.candidates)))


@dataclass(frozen=True)
class CalibrationConfig:
    maps: tuple[str, ...]
    low_confidence_threshold: float
    name: str                    # calibration/<name>.json（共有）

    @property
    def path(self) -> Path:
        return calibration_path(self.name)


@dataclass(frozen=True)
class CostConfig:
    maps: tuple[str, ...]        # 実行時間を読む系列（全モデルを同じ系列で計測していること）


@dataclass(frozen=True)
class ExpConfig:
    dataset: str
    test_map: str
    iou_threshold: float
    integrate_way: str
    max_version: int
    thresholds: ThresholdConfig
    tracker: TrackerConfig
    selection: SelectionConfig
    calibration: CalibrationConfig
    cost: CostConfig


def load_config(path: Path) -> ExpConfig:
    return parse_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_config(raw: dict) -> ExpConfig:
    """読み込み済みの dict から ExpConfig を作る（比較手法の config が上書きした dict にも使う）。"""
    sel, cal = raw["selection"], raw["calibration"]

    cfg = ExpConfig(
        dataset=raw["dataset"],
        test_map=str(raw["test_map"]),
        iou_threshold=raw["iou_threshold"],
        integrate_way=raw["integrate_way"],
        max_version=raw["max_version"],
        thresholds=ThresholdConfig(**raw["thresholds"]),
        tracker=TrackerConfig(**raw["tracker"]),
        selection=SelectionConfig(
            base_model=sel["base_model"],
            candidates=tuple(sel["candidates"]),
        ),
        calibration=CalibrationConfig(
            maps=tuple(str(m) for m in cal["maps"]),
            low_confidence_threshold=float(cal["low_confidence_threshold"]),
            name=str(cal["name"]),
        ),
        cost=CostConfig(maps=tuple(str(m) for m in raw["cost"]["maps"])),
    )
    if cfg.test_map in cfg.calibration.maps:
        raise ValueError(f"test_map {cfg.test_map} が calibration.maps に含まれています")
    return cfg
