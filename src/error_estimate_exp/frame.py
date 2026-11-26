"""
frame.py
1 フレーム分の検出結果とトラッカー出力を、必要になった時点で一度だけ計算して保持する。

State は frame.det(model) / frame.track(model) を呼ぶだけでよく、
キャッシュかモデル推論かを意識しない。同じフレームで同じモデルを二度実行しないので、
実行したモデルの一覧（executed）がそのまま計算コストの根拠になる。
"""
from __future__ import annotations

from typing import Callable


class Frame:
    def __init__(
        self,
        detect: Callable[[str], list],
        track: Callable[[str, list], list],
    ) -> None:
        self._detect = detect
        self._track = track
        self._dets: dict[str, list] = {}
        self._tracks: dict[str, list] = {}

    def det(self, model: str) -> list:
        if model not in self._dets:
            self._dets[model] = self._detect(model)
        return self._dets[model]

    def track(self, model: str) -> list:
        if model not in self._tracks:
            self._tracks[model] = self._track(model, self.det(model))
        return self._tracks[model]

    @property
    def detections(self) -> dict[str, list]:
        """実行した各モデルの検出結果（統合前）。"""
        return dict(self._dets)

    @property
    def executed(self) -> tuple[str, ...]:
        """このフレームで検出を実行したモデル（実行順）。"""
        return tuple(self._dets)
