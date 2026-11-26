"""
config.py
比較手法の設定（comparison_exp/config/<method>_<map>/config.yaml）を読む。
error_estimate_exp と同じく 1 マップ 1 ディレクトリで、結果もそのディレクトリに保存する。

共通の設定（test_map・閾値・統合方法・候補など）は、同じマップの提案手法の config をそのまま使う。
    base_config: conf_base_0020   # error_estimate_exp/config/<name>/config.yaml を読む
    thresholds:                   # 任意: 上書きしたいキーだけ書く（dict は再帰的にマージ）
      theta_low: 0.2
    method:                       # 比較手法の指定（下の MethodConfig）
      name: fixed
      fixed_models: [rtdetr_torch, yolov11n]

config は make_configs.py でマップごとに作る。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.error_estimate_exp.config import ExpConfig, config_path as ee_config_path, parse_config

COMPARISON_EXP_ROOT = PROJECT_ROOT / "src" / "comparison_exp"
CONFIG_ROOT = COMPARISON_EXP_ROOT / "config"

# 状態遷移は提案手法のまま、追加モデルの選び方だけ替える手法
SELECTION_METHODS = ("proposed", "fixed", "random", "oracle")
# 先行研究の構成を移した手法
ARCHITECTURE_METHODS = ("recovery_block", "confidence_exit")
METHODS = SELECTION_METHODS + ARCHITECTURE_METHODS


def config_path(cfg_name: str) -> Path:
    """config/<cfg_name>/config.yaml。結果もこのディレクトリに保存される。"""
    return CONFIG_ROOT / cfg_name / "config.yaml"


def all_config_names() -> list[str]:
    return sorted(p.parent.name for p in CONFIG_ROOT.glob("*/config.yaml"))


@dataclass(frozen=True)
class MethodConfig:
    name: str
    fixed_models: tuple[str, ...] = ()       # fixed: 第 2・第 3 モデルの順
    seeds: tuple[int, ...] = (0,)            # random: シードごとに 1 回実行して平均
    primary: tuple[str, ...] = ()            # recovery_block: 常に動かす 2 モデル
    backup: str | None = None                # recovery_block: 不一致のときに置き換えるモデル
    threshold: float | None = None           # recovery_block: 一致度の閾値（省略時 theta_low）
    order: tuple[str, ...] = ()              # confidence_exit: 追加する順（先頭から、最大 3 モデル）
    confidence_threshold: float | None = None  # confidence_exit: 平均確信度の閾値（省略時 tau_p）

    def models(self) -> tuple[str, ...]:
        """この手法が動かしうるモデル（selection.models に含まれている必要がある）。"""
        listed = (*self.fixed_models, *self.primary, *self.order)
        if self.backup:
            listed += (self.backup,)
        return tuple(dict.fromkeys(listed))


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> tuple[ExpConfig, MethodConfig]:
    raw = read_raw(path)
    base_name = raw.pop("base_config")
    method_raw = raw.pop("method")

    base_raw = read_raw(ee_config_path(base_name))
    cfg = parse_config(_merge(base_raw, raw))
    method = _parse_method(method_raw)
    _validate(cfg, method)
    return cfg, method


def base_config_name(path: Path) -> str:
    return str(read_raw(path)["base_config"])


def _parse_method(raw: dict) -> MethodConfig:
    tuples = ("fixed_models", "seeds", "primary", "order")
    return MethodConfig(**{k: tuple(v) if k in tuples else v for k, v in raw.items()})


def _validate(cfg: ExpConfig, m: MethodConfig) -> None:
    if m.name not in METHODS:
        raise ValueError(f"method.name は {METHODS} のいずれかです: {m.name}")

    required = {
        "fixed": ("fixed_models",),
        "recovery_block": ("primary", "backup"),
        "confidence_exit": ("order",),
    }.get(m.name, ())
    for key in required:
        if not getattr(m, key):
            raise ValueError(f"method.name={m.name} には method.{key} が必要です")
    if m.name == "recovery_block" and len(m.primary) != 2:
        raise ValueError(f"recovery_block の primary は 2 モデルです: {m.primary}")
    if m.name == "confidence_exit" and not 1 <= len(m.order) <= cfg.max_version:
        raise ValueError(f"confidence_exit の order は 1〜{cfg.max_version} モデルです: {m.order}")

    # モデルはすべて事前に読み込むので selection に含まれている必要がある
    missing = sorted(set(m.models()) - set(cfg.selection.models))
    if missing:
        raise ValueError(f"selection.base_model / candidates に含まれないモデルがあります: {missing}")
