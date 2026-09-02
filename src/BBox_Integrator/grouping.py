"""
grouping.py

detector制約付き Union-Find による bounding box のグルーピング。
同一 detector（辞書のキー）のboxは同一グループに最大1つまでという制約を守りながら、
IoUがしきい値以上のboxをまとめる。

Integratorには依存しない standalone な実装。
tracker_error_prediction_experiment.py など、グルーピング結果だけが
必要な場所から直接利用する。
"""
from dataclasses import dataclass, field

from src.boundingBox.boundingBox import DetectionBoundingBox


def grouping_detections(
    detection_model_dict: dict[object, list[DetectionBoundingBox]],
    iou_threshold: float,
) -> list[list[DetectionBoundingBox]]:
    """
    detector 制約付き Union-Find によるグルーピング

    制約:
    ・IoU >= iou_threshold
    ・同一 detector（辞書のキー）は同一グループに最大 1 box
    """
    # --- Step 1: box を ID 化 ---
    id_to_box: dict[int, tuple[object, int]] = {}
    box_to_id: dict[tuple[object, int], int] = {}

    box_id = 0
    for detector, boxes in detection_model_dict.items():
        for idx in range(len(boxes)):
            id_to_box[box_id] = (detector, idx)
            box_to_id[(detector, idx)] = box_id
            box_id += 1

    total_boxes = box_id

    # --- Step 2: Union-Find 初期化 ---
    parent = list(range(total_boxes))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[rx] = ry

    # --- Step 3: 全ペア IoU 計算 ---
    matches: list[tuple[int, int, float]] = []

    detectors = list(detection_model_dict.keys())
    for i, d1 in enumerate(detectors):
        for d2 in detectors[i + 1:]:
            boxes1 = detection_model_dict[d1]
            boxes2 = detection_model_dict[d2]

            for idx1, b1 in enumerate(boxes1):
                for idx2, b2 in enumerate(boxes2):
                    iou = b1.computeIoU(b2)
                    if iou >= iou_threshold:
                        id1 = box_to_id[(d1, idx1)]
                        id2 = box_to_id[(d2, idx2)]
                        matches.append((id1, id2, iou))

    # IoU 降順
    matches.sort(key=lambda x: x[2], reverse=True)

    # --- Step 4: detector 制約付き union ---
    # 各 root が含む detector 集合を管理
    root_detectors: dict[int, set[object]] = {
        i: {id_to_box[i][0]} for i in range(total_boxes)
    }

    for id1, id2, _ in matches:
        r1 = find(id1)
        r2 = find(id2)

        if r1 == r2:
            continue

        # detector が衝突するならスキップ
        if root_detectors[r1] & root_detectors[r2]:
            continue

        # union 実行
        union(r1, r2)
        new_root = find(r1)

        # detector 集合を更新
        root_detectors[new_root] = (
            root_detectors[r1] | root_detectors[r2]
        )

    # --- Step 5: グループ生成 ---
    groups: dict[int, list[DetectionBoundingBox]] = {}

    for box_id in range(total_boxes):
        root = find(box_id)
        if root not in groups:
            groups[root] = []

        detector, idx = id_to_box[box_id]
        groups[root].append(detection_model_dict[detector][idx])

    return list(groups.values())


@dataclass
class GroupingResult:
    """2つの検出集合（例: Dcur と Dest）をグルーピングした結果"""
    agreed: list[DetectionBoundingBox] = field(default_factory=list)
    det_only: list[DetectionBoundingBox] = field(default_factory=list)
    trk_only: list[DetectionBoundingBox] = field(default_factory=list)


def group_detection_and_tracking(
    model_detections: list[DetectionBoundingBox],
    tracking_detections: list[DetectionBoundingBox],
    iou_threshold: float,
) -> GroupingResult:
    """
    grouping_detections() を使い、Dcur（"model"）と Dest（"tracking"）を
    一致(agreed) / 検出のみ(det_only) / トラックのみ(trk_only) に仕分ける。

    NOTE: キーが "model" / "tracking" の2つだけなので、各グループのサイズは
    最大2（一致なら2、片方のみなら1）になる。agreedにはmodel側のboxのみを
    代表として残す（1グループ = 1件としてdet_only/trk_onlyと数えやすくするため）。
    """
    groups = grouping_detections(
        {"model": model_detections, "tracking": tracking_detections},
        iou_threshold=iou_threshold,
    )

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