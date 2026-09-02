from . import BoundingBox
from src.ObjectDetection.models.ObjectDetector import Detector


def groupingBoundingBox(detectionModelDict: dict[object, list[BoundingBox]], iouThreshold: float) -> list[list[BoundingBox]]:
    """
    detector 制約付き Union-Find によるグルーピング

    制約：
    ・IoU >= threshold
    ・同一 detector は同一グループに最大 1 box
    """

    # --- Step 1: box を ID 化 ---
    id_to_box: dict[int, tuple[object, int]] = {}
    box_to_id: dict[tuple[object, int], int] = {}

    box_id = 0
    for detector, boxes in detectionModelDict.items():
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

    detectors = list(detectionModelDict.keys())
    for i, d1 in enumerate(detectors):
        for d2 in detectors[i + 1:]:
            boxes1 = detectionModelDict[d1]
            boxes2 = detectionModelDict[d2]

            for idx1, b1 in enumerate(boxes1):
                for idx2, b2 in enumerate(boxes2):
                    iou = b1.computeIoU(b2)
                    if iou >= iouThreshold:
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
    groups: dict[int, list[BoundingBox]] = {}

    for box_id in range(total_boxes):
        root = find(box_id)
        if root not in groups:
            groups[root] = []

        detector, idx = id_to_box[box_id]
        groups[root].append(detectionModelDict[detector][idx])

    return list(groups.values())
