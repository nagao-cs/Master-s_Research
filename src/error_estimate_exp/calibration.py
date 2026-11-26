"""
calibration.py
事前計測の結果（エラー推定プロファイルと解消率）を保持し、JSON で保存・読み込みする。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.error_estimate.estimation.frame_type import FrameErrorProfile

from .selection import Resolution

Pair = tuple[str, str]


@dataclass
class Calibration:
    dataset: str
    fit_maps: list[str]
    low_confidence_threshold: float
    one_model: dict[str, FrameErrorProfile]   # 対象モデル → 単一モデル時のプロファイル
    two_model: dict[Pair, FrameErrorProfile]  # (対象, 相手) → 2 モデル時のプロファイル
    resolution: dict[Pair, Resolution]        # (対象, 候補) → 解消率

    @property
    def models(self) -> list[str]:
        return sorted(self.one_model)

    def save(self, path: Path) -> None:
        payload = {
            "dataset": self.dataset,
            "fit_maps": list(self.fit_maps),
            "low_confidence_threshold": self.low_confidence_threshold,
            "one_model": {m: asdict(p) for m, p in self.one_model.items()},
            "two_model": _nest({k: asdict(p) for k, p in self.two_model.items()}),
            "resolution": _nest({k: asdict(r) for k, r in self.resolution.items()}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dataset=payload["dataset"],
            fit_maps=list(payload["fit_maps"]),
            low_confidence_threshold=float(payload["low_confidence_threshold"]),
            one_model={m: FrameErrorProfile(**p) for m, p in payload["one_model"].items()},
            two_model={k: FrameErrorProfile(**p) for k, p in _flatten(payload["two_model"]).items()},
            resolution={k: Resolution(**r) for k, r in _flatten(payload["resolution"]).items()},
        )


def _nest(table: dict[Pair, dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for (a, b), v in table.items():
        out.setdefault(a, {})[b] = v
    return out


def _flatten(nested: dict[str, dict[str, dict]]) -> dict[Pair, dict]:
    return {(a, b): v for a, inner in nested.items() for b, v in inner.items()}
