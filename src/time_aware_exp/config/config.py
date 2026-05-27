"""
config.py
設定の定義と読み込みのみを担う。
"""
from __future__ import annotations
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class ThresholdConfig:
    theta_high: float   # Jaccard係数がこれ以上でTwo→Single
    theta_low:  float   # Jaccard係数がこれ未満でTwo→Three
    theta_track: float  # トラッキング不確実性がこれ以上でSingle→Two
    # tau_p: float        # conf_baseの採用閾値


@dataclass
class SortConfig:
    iou_threshold: float   # マッチングのIoU閾値
    max_age:       int     # 検出なしで生存させるフレーム数
    min_hits:      int     # 確立済みトラックとみなす最低連続マッチ数


@dataclass
class AdrodConfig:
    dataset:       str
    integrate_way: str
    model_1:       str
    model_2:       str
    model_3:       str
    iou_threshold: int
    thresholds:    ThresholdConfig
    sort:          SortConfig


def load_config(path: str | Path) -> AdrodConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    print(d)
    return AdrodConfig(
        dataset=d["dataset"],
        integrate_way=d["integrate_way"],
        model_1=d["model_1"],
        model_2=d["model_2"],
        model_3=d["model_3"],
        iou_threshold=d["iou_threshold"],
        thresholds=ThresholdConfig(**d["thresholds"]),
        sort=SortConfig(**d["sort"]),
    )