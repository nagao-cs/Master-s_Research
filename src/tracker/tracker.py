"""
tracker.py
SORT を使ったトラッカー。
SOLO→PAIR の遷移判定に使う不確実性信号と、
追跡済みバウンディングボックスのリストを提供する。
クラス別に Sort インスタンスを管理する。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from src.boundingBox.boundingBox import DetectionBoundingBox
from .sort_core import Sort


MAX_AGE=1
MIN_HITS=1
IOU_THRESHOLD=0.5

@dataclass
class TrackingResult:
    lost_count:    int                       # 前フレームにいたが消えた確立済みトラック数
    new_count:     int                       # 今フレームで新たに確立されたトラック数
    total_tracked: int                       # 現在の確立済みトラック総数
    tracked_boxes: list[DetectionBoundingBox] = field(default_factory=list)  # 追跡済みbboxリスト


class SortTracker:
    """
    クラスごとに Sort インスタンスを持つトラッカー。
    update() を毎フレーム呼び出し、TrackingResult を得る。

    TrackingResult.tracked_boxes には確立済みトラックの現在位置が
    DetectionBoundingBox として格納される。
    座標系は入力の DetectionBoundingBox と同じ（center形式・正規化済み）。
    confidenceScore は SORT が保持しないため 1.0 で埋める。
    """

    def __init__(self):
        self._sorters: dict[str, Sort] = {}
        self._prev_track_ids: set[int] = set()

    def update(self, detections: list[DetectionBoundingBox]) -> list[DetectionBoundingBox]:
        # ラベルごとに検出を分割
        by_label: dict[str, list[DetectionBoundingBox]] = {}
        for det in detections:
            by_label.setdefault(str(det.classId), []).append(det)

        current_ids: set[int] = set()
        tracked_boxes: list[DetectionBoundingBox] = []

        # 検出があるクラスを更新
        for label, dets in by_label.items():
            sorter = self._get_sorter(label)
            dets_np = np.array([
                [
                    d.xCenter - d.width  / 2,
                    d.yCenter - d.height / 2,
                    d.xCenter + d.width  / 2,
                    d.yCenter + d.height / 2,
                    d.confidenceScore,
                ]
                for d in dets
            ])
            tracks = sorter.update(dets_np)

            for row in tracks:
                x1, y1, x2, y2, track_id, confidence = row
                current_ids.add(int(track_id))
                tracked_boxes.append(
                    DetectionBoundingBox(
                        xCenter=(x1 + x2) / 2,
                        yCenter=(y1 + y2) / 2,
                        width=x2 - x1,
                        height=y2 - y1,
                        classId=int(label),
                        confidenceScore=float(confidence),
                    )
                )

        # 検出がないクラスも更新（トラック削除のため必須）
        for label, sorter in self._sorters.items():
            if label not in by_label:
                sorter.update(np.empty((0, 5)))

        lost_count = len(self._prev_track_ids - current_ids)
        new_count  = len(current_ids - self._prev_track_ids)
        self._prev_track_ids = current_ids

        return tracked_boxes

    def reset(self) -> None:
        self._sorters = {}
        self._prev_track_ids = set()

    def _get_sorter(self, label: str) -> Sort:
        if label not in self._sorters:
            self._sorters[label] = Sort(
                max_age=MAX_AGE,
                min_hits=MIN_HITS,
                iou_threshold=IOU_THRESHOLD
            )
        return self._sorters[label]