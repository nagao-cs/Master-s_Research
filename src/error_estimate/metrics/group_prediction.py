from src.boundingBox.boundingBox import DetectionBoundingBox
from src.BBox_Integrator.grouping import grouping_detections
from dataclasses import dataclass, field


@dataclass
class GroupingResult:
    """Dcur（検出）と Dest（トラック予測）をグルーピングした結果"""
    agreed: list[DetectionBoundingBox] = field(default_factory=list)
    det_only: list[DetectionBoundingBox] = field(default_factory=list)
    trk_only: list[DetectionBoundingBox] = field(default_factory=list)
 
 
def group_detection_and_tracking(
    model_detections: list[DetectionBoundingBox],
    tracking_detections: list[DetectionBoundingBox],
) -> GroupingResult:
    """
    integrator.groupingDetections() の detector制約付きグルーピングを使い、
    Dcur（"model"）と Dest（"tracking"）を 一致(agreed) / 検出のみ(det_only) /
    トラックのみ(trk_only) に仕分ける。
 
    NOTE: キーが "model" / "tracking" の2つだけなので、各グループのサイズは
    最大2（一致なら2、片方のみなら1）になる。agreedにはmodel側のboxのみを
    代表として残す（1グループ = 1件としてdet_only/trk_onlyと数えやすくするため）。
    groupingDetections がintegratorの公開メソッドでない場合は、
    呼び出し側をintegratorの実際の参照に合わせて調整すること。
    """
    groups = grouping_detections({"det": model_detections, "trk": tracking_detections}, iou_threshold=0.5)

    model_ids = {id(b) for b in model_detections}
    tracking_ids = {id(b) for b in tracking_detections}
 
    result = GroupingResult()
    for group in groups:
        has_model = any(id(b) in model_ids for b in group)
        has_tracking = any(id(b) in tracking_ids for b in group)
 
        if has_model and has_tracking:
            model_box = next(b for b in group if id(b) in model_ids)
            result.agreed.append(model_box)
        elif has_model:
            result.det_only.extend(group)
        elif has_tracking:
            result.trk_only.extend(group)
 
    return result