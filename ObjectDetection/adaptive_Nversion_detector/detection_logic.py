from typing import List, Dict, Any
from math import ceil


def check_switch_to_Nversion(detections: List[Dict[str, Any]], rule: str, threshold: float) -> bool:
    """
    Nバージョンに切り替えるか判定する

    Args:
        detections (List[Dict[str, Any]]): ベースの検出結果
        rule (str): バージョン数決定ルール("n_det" または "min_conf")
        threshold (float): 閾値
    Returns:
        bool: Nバージョンに切り替える場合はTrue、そうでなければFalse
    """

    if rule == "n_det":
        n_detections = len(detections)
        if n_detections >= threshold:
            return True
        else:
            return False
    elif rule == "min_conf":
        if not detections:
            return True
        min_conf = min(det['confidence'] for det in detections)
        if min_conf < threshold:
            return True
        else:
            return False
    else:
        raise ValueError(f"Unknown rule: {rule}")


def integrate_N_detections(detections_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Nバージョンの検出結果を多数決で統合する (レポート 3.2節に基づく)

    Args:
        detections_list: 各バージョンの検出結果のリスト
    Returns:
        統合された検出結果
    """
    integrated_detections = []
    majority_threshold = ceil(len(detections_list) / 2)
    IOU_THRESHOLD = 0.5

    all_detections = []
    for version_id, version_dets in enumerate(detections_list):
        for det in version_dets:
            all_detections.append({**det, 'version_id': version_id})

    is_processed = [False] * len(all_detections)
    groups = []

    for i in range(len(all_detections)):
        if is_processed[i]:
            continue

        det_i = all_detections[i]
        current_group = [det_i]
        is_processed[i] = True
        best_iou = 0.0
        matched_index = -1

        for j in range(i + 1, len(all_detections)):
            det_j = all_detections[j]

            if (det_i['class_id'] != det_j['class_id']) or is_processed[j] or (det_i['version_id'] == det_j['version_id']):
                continue

            iou = _compute_iou(det_i, det_j)
            if iou < IOU_THRESHOLD or iou < best_iou:
                continue

            if iou > best_iou:
                best_iou = iou
                matched_index = j
        if matched_index != -1:
            current_group.append(all_detections[matched_index])
            is_processed[matched_index] = True

        groups.append(current_group)

    for group in groups:
        if len(group) >= majority_threshold:

            avg_det = {
                'x_center': sum(det['x_center'] for det in group) / len(group),
                'y_center': sum(det['y_center'] for det in group) / len(group),
                'width': sum(det['width'] for det in group) / len(group),
                'height': sum(det['height'] for det in group) / len(group),
                'confidence': sum(det['confidence'] for det in group) / len(group),
                'label': group[0]['label'],
                'class_id': group[0]['class_id']
            }
            integrated_detections.append(avg_det)

    return integrated_detections


def _compute_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    2つのバウンディングボックスのIoUを計算する

    Args:
        box1 (Dict[str, Any]): バウンディングボックス1
        box2 (Dict[str, Any]): バウンディングボックス2
    Returns:
        float: IoU値
    """
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2

    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * \
        max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou
