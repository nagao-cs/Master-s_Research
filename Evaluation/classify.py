import os
import utils
from pprint import pprint


def classify(gt_dir, det_dir) -> list:
    """
    gt_dir: Ground Truth directory
    det_dir: Detection results directory

    return: List of classification results for each frame
    """
    results = list()
    gt_files = [os.path.join(gt_dir, gt_file)
                for gt_file in os.listdir(gt_dir)]
    det_files = [os.path.join(det_dir, det_file)
                 for det_file in os.listdir(det_dir)]

    for gt_file, det_file in zip(gt_files, det_files):
        result = {'TP': dict(), 'FP': dict(), 'FN': dict()}
        gt = get_gt(gt_file)
        det = get_detections(det_file)

        # クラスごとに処理
        for class_id in gt:
            gt_boxes = gt[class_id]
            det_boxes = det.get(class_id, [])

            # 信頼度でソート
            det_boxes.sort(key=lambda x: x[4], reverse=True)  # x[4]はconfidence

            if class_id not in result['TP']:
                result['TP'][class_id] = list()
            if class_id not in result['FN']:
                result['FN'][class_id] = list()
            if class_id not in result['FP']:
                result['FP'][class_id] = list()

            # GTとDetectionのマッチング
            used_gt = set()  # マッチ済みのGTを記録

            for det_box in det_boxes:
                best_iou = 0.0
                best_gt_idx = -1

                # 未使用のGTと最もIoUが高いものを探す
                for i, gt_box in enumerate(gt_boxes):
                    if i in used_gt:
                        continue
                    iou = utils.iou(gt_box, det_box)
                    if iou >= utils.IoU_THRESHOLD and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                # マッチング結果に基づいて分類
                if best_gt_idx >= 0:
                    result['TP'][class_id].append(det_box)
                    used_gt.add(best_gt_idx)
                else:
                    result['FP'][class_id].append(det_box)

            # 未使用のGTをFNとして追加
            for i, gt_box in enumerate(gt_boxes):
                if i not in used_gt:
                    result['FN'][class_id].append(gt_box)

        results.append(result)
        # pprint(f"results: {results}")
    return results


def get_gt(gt_file_path) -> dict:
    """
    gt_file_path: Path to a ground truth file
    """
    gt = dict()
    with open(gt_file_path, 'r') as gt_file:
        lines = gt_file.readlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            class_id = utils.class_Map.get((int(parts[0])), -1)  # -1（無視するクラス）
            if class_id == -1:
                continue
            # class_id = int(parts[0])
            xmin = float(parts[1])
            xmax = float(parts[2])
            ymin = float(parts[3])
            ymax = float(parts[4])
            distance = 0.0  # 仮の値、必要に応じて計算する
            size = (xmax-xmin) * (ymax-ymin)
            if size < utils.SIZE_THRESHOLD:
                continue
            if class_id not in gt:
                gt[class_id] = list()
            # 仮の値、必要に応じて計算する
            gt[class_id].append((xmin, xmax, ymin, ymax, distance))

    return gt


def get_detections(det_file_path) -> dict:
    """
    det_file_path: Path to a detection results file
    """
    detections = dict()
    with open(det_file_path, 'r') as det_file:
        lines = det_file.readlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            class_id = utils.class_Map.get((int(parts[0])), -1)  # -1（無視するクラス）
            if class_id == -1:
                continue
            # class_id = int(parts[0])
            xmin = float(parts[1])
            xmax = float(parts[2])
            ymin = float(parts[3])
            ymax = float(parts[4])
            confidence = float(parts[5])
            size = (xmax-xmin) * (ymax-ymin)
            if size < utils.SIZE_THRESHOLD:
                continue
            if confidence < utils.CONF_THRESHOLD:
                continue
            if class_id not in detections:
                detections[class_id] = list()
            detections[class_id].append((xmin, xmax, ymin, ymax, confidence))

    return detections
