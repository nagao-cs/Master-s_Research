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


# def integrate_N_detections(detections_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Nバージョンの検出結果を多数決で統合する

    Args:
        detections_list (List[List[Dict[str(class_id), Any]]]): 各バージョンの検出結果のリスト
    Returns:
        List[Dict[str, Any]]: 統合された検出結果
    """
    integrated_detections = []
    majority_threshold = ceil(len(detections_list) / 2)
    groups = list()
    for version_id, detections in enumerate(detections_list):
        if version_id == 0:
            for det in detections:
                groups.append([det])
        else:
            matched_flags = [False] * len(groups)
            for det in detections:
                best_iou = 0.0
                best_group = None
                for group in groups:
                    if matched_flags[groups.index(group)]:
                        continue
                    base_det = group[0]
                    if base_det['class_id'] != det['class_id']:
                        continue
                    iou = _compute_iou(base_det, det)
                    if iou >= 0.5 and iou > best_iou:
                        best_iou = iou
                        best_group = group
                if best_group is not None:
                    best_group.append(det)
                    matched_flags[groups.index(best_group)] = True
                else:
                    groups.append([det])
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


def integrate_N_detections(detections_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Nバージョンの検出結果を多数決で統合する (レポート 3.2節に基づく)

    Args:
        detections_list: 各バージョンの検出結果のリスト
    Returns:
        統合された検出結果
    """
    integrated_detections = []
    # N=3の場合、majority_threshold は ceil(3/2) = 2
    majority_threshold = ceil(len(detections_list) / 2)
    IOU_THRESHOLD = 0.5  # レポートの要件

    # すべての検出結果を1つのリストに集め、元のバージョンIDを付与
    all_detections = []
    for version_id, version_dets in enumerate(detections_list):
        for det in version_dets:
            # 元のバージョンIDを追跡し、後の処理で異なるバージョンからの検出であることを確認
            all_detections.append({**det, 'version_id': version_id})

    is_processed = [False] * len(all_detections)
    groups = []

    # 1. すべての検出結果を網羅的にグループ化
    # このロジックは、元のコードの意図 (検出結果を順次グループに振り分ける) を維持します。

    for i in range(len(all_detections)):
        if is_processed[i]:
            continue

        det_i = all_detections[i]
        current_group = [det_i]
        is_processed[i] = True

        for j in range(i + 1, len(all_detections)):
            if is_processed[j]:
                continue

            det_j = all_detections[j]

            # クラスラベルが同一
            if det_i['class_id'] != det_j['class_id']:
                continue

            # 座標の一致度が0.5以上
            iou = _compute_iou(det_i, det_j)
            if iou < IOU_THRESHOLD:
                continue

            # 💡 異なるバージョンからの検出のみをグループに追加する厳密なチェック (より堅牢な多数決のため)
            if det_i['version_id'] != det_j['version_id']:
                # det_j は det_i のグループに属する
                current_group.append(det_j)
                is_processed[j] = True

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
