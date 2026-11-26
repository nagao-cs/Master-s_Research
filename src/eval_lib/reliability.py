"""
reliability.py
N-version 物体検出の信頼性指標 Cov_OD / Cer_OD（ISSRE 論文の式 (4)-(10)）。

フレーム x で実行した各バージョン m_i の TP_i, FP_i, FN_i から
    E_i(x)   = FP_i(x) ∪ FN_i(x)
    IoE(x)   = ∩_i E_i(x)       全バージョンに共通する誤り
    UoE(x)   = ∪_i E_i(x)       いずれかのバージョンの誤り
    O(x)     = ∪_i (TP_i ∪ FP_i ∪ FN_i)
    Cov_OD   = 1 - mean_x |IoE(x)| / |O(x)|
    Cer_OD   = 1 - mean_x |UoE(x)| / |O(x)|
（|O(x)| = 0 のフレームは誤り率 0 として平均に含める）

ReliabilityOfNversionObjectDetection/Metrics/CovOD.py, CerOD.py と
processor/analyzer.py の考え方に合わせ、
- FN は GT の物体なので GT のインデックスで同一性を判定する（厳密）
- FP は物体が存在しないので、バージョン間で IoU >= iou_threshold かつ同クラスの
  検出を同じ物体とみなす
- |O(x)| = |TP の積集合| + |FP の和集合| + |FN の和集合|
  （TP の積集合と FN の和集合を合わせると GT 全体になる）
とする。AdROD ではフレームごとに実行モデル数が変わるので、
各フレームで実際に実行したバージョンだけを使う（1 モデルなら IoE = UoE = そのモデルの誤り）。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.boundingBox.boundingBox import BoundingBox, DetectionBoundingBox, GroundTruthBoundingBox


@dataclass
class VersionResult:
    """1 バージョン・1 フレームの分類結果。"""
    tp_gt: set[int]                     # 検出できた GT のインデックス
    fn_gt: set[int]                     # 見逃した GT のインデックス
    fp: list[DetectionBoundingBox]      # GT に対応しない検出


@dataclass
class FrameReliability:
    """1 フレームの集合の大きさ。"""
    intersection_errors: int   # |IoE(x)| = |∩FP| + |∩FN|
    union_errors: int          # |UoE(x)| = |∪FP| + |∪FN|
    total_instances: int       # |O(x)|   = |∩TP| + |∪FP| + |∪FN|


def classify_version(
    gt: list[GroundTruthBoundingBox],
    dets: list[DetectionBoundingBox],
    iou_threshold: float,
) -> VersionResult:
    """IoU の高い順に GT と検出を 1 対 1 で対応づける（DetectionClassifier と同じ規則）。"""
    pairs = [
        (iou, g, d)
        for g, gt_box in enumerate(gt)
        for d, det_box in enumerate(dets)
        if (iou := gt_box.computeIoU(det_box)) >= iou_threshold
    ]
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_gt: set[int] = set()
    used_det: set[int] = set()
    for _, g, d in pairs:
        if g in used_gt or d in used_det:
            continue
        used_gt.add(g)
        used_det.add(d)

    return VersionResult(
        tp_gt=used_gt,
        fn_gt=set(range(len(gt))) - used_gt,
        fp=[box for d, box in enumerate(dets) if d not in used_det],
    )


def _intersection_boxes(box_lists: list[list[BoundingBox]], iou_threshold: float) -> list[BoundingBox]:
    """全リストに IoU >= iou_threshold の相手がいる箱（analyzer._intersection_of_errors と同じ）。"""
    common = list(box_lists[0])
    for others in box_lists[1:]:
        used: set[int] = set()
        kept = []
        for box in common:
            best_iou, best = 0.0, None
            for j, other in enumerate(others):
                if j in used:
                    continue
                iou = box.computeIoU(other)
                if iou > best_iou:
                    best_iou, best = iou, j
            if best is not None and best_iou >= iou_threshold:
                kept.append(box)
                used.add(best)
        common = kept
    return common


def _union_boxes(box_lists: list[list[BoundingBox]], iou_threshold: float) -> list[BoundingBox]:
    """重複（IoU >= iou_threshold）を除いた和集合（analyzer._union_of_errors と同じ）。"""
    union: list[BoundingBox] = []
    for boxes in box_lists:
        for box in boxes:
            if all(box.computeIoU(u) < iou_threshold for u in union):
                union.append(box)
    return union


def analyze_frame(
    gt: list[GroundTruthBoundingBox],
    version_dets: list[list[DetectionBoundingBox]],
    iou_threshold: float,
) -> FrameReliability:
    """1 フレームの IoE / UoE / O の大きさを求める。version_dets は 1 つ以上。"""
    if not version_dets:
        raise ValueError("version_dets が空です")
    results = [classify_version(gt, dets, iou_threshold) for dets in version_dets]

    fn_inter = set.intersection(*(r.fn_gt for r in results))
    fn_union = set.union(*(r.fn_gt for r in results))
    tp_inter = set.intersection(*(r.tp_gt for r in results))
    fp_lists = [r.fp for r in results]
    fp_inter = _intersection_boxes(fp_lists, iou_threshold)
    fp_union = _union_boxes(fp_lists, iou_threshold)

    return FrameReliability(
        intersection_errors=len(fp_inter) + len(fn_inter),
        union_errors=len(fp_union) + len(fn_union),
        total_instances=len(tp_inter) + len(fp_union) + len(fn_union),
    )


class CovOD:
    """Cov_OD = 1 - (全バージョン共通の誤り / O(x)) のフレーム平均。"""

    def __init__(self) -> None:
        self.val = 0.0
        self.num_frames = 0

    def update(self, frame: FrameReliability) -> None:
        if frame.total_instances > 0:
            self.val += frame.intersection_errors / frame.total_instances
        self.num_frames += 1

    def compute(self) -> float:
        return 1.0 if self.num_frames == 0 else 1 - self.val / self.num_frames


class CerOD:
    """Cer_OD = 1 - (いずれかのバージョンの誤り / O(x)) のフレーム平均。"""

    def __init__(self) -> None:
        self.val = 0.0
        self.num_frames = 0

    def update(self, frame: FrameReliability) -> None:
        if frame.total_instances > 0:
            self.val += frame.union_errors / frame.total_instances
        self.num_frames += 1

    def compute(self) -> float:
        return 1.0 if self.num_frames == 0 else 1 - self.val / self.num_frames
