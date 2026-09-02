"""
tracker.py
SORT を使ったトラッカー。
SOLO→PAIR の遷移判定に使う不確実性信号と、
追跡済みバウンディングボックスのリストを提供する。
クラス別に Sort インスタンスを管理する。

TrackingResult は2系統のバウンディングボックスを持つ。
    tracked_boxes   : 現在の検出でマッチしたトラックの補正後の状態
                       （従来の tracker_result と同じ意味・同じ条件）
    predicted_boxes : 確立済み全トラックの、現在の検出で補正する前の
                       線形予測状態（マッチの有無を問わず出力）
                       モデルが見逃した物体をトラッカーが独自に
                       予測できているかを見るための差分評価に使う。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from src.boundingBox.boundingBox import DetectionBoundingBox
from .sort_core import Sort, TrackedObject


MAX_AGE = 2
MIN_HITS = 1
IOU_THRESHOLD = 0.5


@dataclass
class TrackingResult:
    lost_count:      int                       # 前フレームにいたが消えた確立済みトラック数
    new_count:       int                       # 今フレームで新たに確立されたトラック数
    total_tracked:   int                       # 現在の確立済みトラック総数
    tracked_boxes:   list[DetectionBoundingBox] = field(default_factory=list)  # 補正後（従来互換）
    predicted_boxes: list[DetectionBoundingBox] = field(default_factory=list)  # 補正前（新規）


class SortTracker:
    """
    クラスごとに Sort インスタンスを持つトラッカー。
    update() を毎フレーム呼び出し、TrackingResult を得る。

    座標系は入力の DetectionBoundingBox と同じ（center形式・正規化済み）。
    confidenceScore は SORT が保持する値をそのまま使う。
    """

    def __init__(self):
        self._sorters: dict[str, Sort] = {}
        self._prev_track_ids: set[int] = set()

    def update(self, detections: list[DetectionBoundingBox]) -> TrackingResult:
        # ラベルごとに検出を分割
        by_label: dict[str, list[DetectionBoundingBox]] = {}
        for det in detections:
            by_label.setdefault(str(det.classId), []).append(det)

        current_ids:     set[int] = set()
        tracked_boxes:   list[DetectionBoundingBox] = []
        predicted_boxes: list[DetectionBoundingBox] = []

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
            current_ids.update(t.track_id for t in tracks)

            # tracked_boxes は従来通りマッチしたものだけ
            tracked_boxes.extend(
                self._to_boxes([t for t in tracks if t.matched], label, use_predicted=False)
            )
            # predicted_boxes は確立済みなら全て（未マッチの予測含む）
            predicted_boxes.extend(self._to_boxes(tracks, label, use_predicted=True))

        # 検出がないクラスも更新（トラック削除・予測維持のため必須）
        for label, sorter in self._sorters.items():
            if label not in by_label:
                tracks = sorter.update(np.empty((0, 5)))
                current_ids.update(t.track_id for t in tracks)
                # このフレームでは検出が無いのでマッチはあり得ない
                predicted_boxes.extend(self._to_boxes(tracks, label, use_predicted=True))

        lost_count    = len(self._prev_track_ids - current_ids)
        new_count     = len(current_ids - self._prev_track_ids)
        total_tracked = len(current_ids)
        self._prev_track_ids = current_ids

        return TrackingResult(
            lost_count=lost_count,
            new_count=new_count,
            total_tracked=total_tracked,
            tracked_boxes=tracked_boxes,
            predicted_boxes=predicted_boxes,
        )

    def reset(self) -> None:
        self._sorters = {}
        self._prev_track_ids = set()

    def _get_sorter(self, label: str) -> Sort:
        if label not in self._sorters:
            self._sorters[label] = Sort(
                max_age=MAX_AGE,
                min_hits=MIN_HITS,
                iou_threshold=IOU_THRESHOLD,
            )
        return self._sorters[label]

    @staticmethod
    def _to_boxes(
        tracks: list[TrackedObject],
        label: str,
        use_predicted: bool,
    ) -> list[DetectionBoundingBox]:
        boxes = []
        for t in tracks:
            x1, y1, x2, y2 = t.predicted_bbox if use_predicted else t.updated_bbox
            boxes.append(
                DetectionBoundingBox(
                    xCenter=(x1 + x2) / 2,
                    yCenter=(y1 + y2) / 2,
                    width=x2 - x1,
                    height=y2 - y1,
                    classId=int(label),
                    confidenceScore=float(t.confidence),
                )
            )
        return boxes