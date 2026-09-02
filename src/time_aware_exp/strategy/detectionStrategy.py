"""
detectionStrategy.py
検出結果の取得方法を Strategy パターンで定義する。

Context は DetectionStrategy に依存し、
キャッシュ・モデル推論の切り替えは外部から注入するだけでよい。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable

from src.boundingBox.boundingBox import DetectionBoundingBox


class DetectionStrategy(ABC):
    """
    検出結果取得の抽象 Strategy。
    frame_ref はキャッシュなら int（フレームインデックス）、
    モデル推論なら画像データなど。
    """

    @abstractmethod
    def detect(self, model: str, frame_ref: Any) -> list[DetectionBoundingBox]:
        ...


class CacheDetectionStrategy(DetectionStrategy):
    """
    事前計算済みキャッシュから検出結果を返す Strategy。
    frame_ref は int（フレームインデックス）。
    """

    def __init__(self, cache: dict[str, list[list[DetectionBoundingBox]]]) -> None:
        self._cache = cache

    def detect(self, model: str, frame_ref: int) -> list[DetectionBoundingBox]:
        return self._cache[model][frame_ref]


class ModelDetectionStrategy(DetectionStrategy):
    """
    実モデルで推論する Strategy。
    frame_ref は画像データ（ndarray など）。
    infer_fns: {model_name: Callable[[image], list[DetectionBoundingBox]]}
    """

    def __init__(self, infer_fns: dict[str, Callable]) -> None:
        self._infer_fns = infer_fns

    def detect(self, model: str, frame_ref: Any) -> list[DetectionBoundingBox]:
        return self._infer_fns[model](frame_ref)