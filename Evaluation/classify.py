import os
import utils


def classify(gt_dir, det_dir):
    """
    gt_dir: Ground Truth directory
    det_dir: Detection results directory
    """
    iou_threshold = 0.5
    results = list()
    gt_files = [os.path.join(gt_dir, gt_file)
                for gt_file in os.listdir(gt_dir)]
    det_files = [os.path.join(det_dir, det_file)
                 for det_file in os.listdir(det_dir)]

    for gt_file, det_file in zip(gt_files, det_files):
        result = {'TP': dict(), 'FP': dict(), 'FN': dict()}
        gt = get_gt(gt_file)
        det = get_detections(det_file)

        # TP, FNを分類する
        for class_id in gt:
            gt_boxes = gt[class_id]
            det_boxes = det.get(class_id, [])
            for gt_box in gt_boxes:
                max_iou = 0.0
                matched_det = None
                for det_box in det_boxes:
                    iou = utils.iou(gt_box, det_box)
                    if iou >= iou_threshold and iou > max_iou:
                        max_iou = iou
                        matched_det = det_box
                if matched_det:
                    if class_id not in result['TP']:
                        result['TP'][class_id] = list()
                    result['TP'][class_id].append(matched_det)
                    det_boxes.remove(matched_det)  # 一度マッチした検出は除外
                else:
                    if class_id not in result['FN']:
                        result['FN'][class_id] = list()
                    result['FN'][class_id].append(gt_box)
        # FPを分類する
        for class_id in det:
            result['FP'][class_id] = det[class_id]
        results.append(result)
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
            xmin = int(float(parts[1]))
            xmax = int(float(parts[2]))
            ymin = int(float(parts[3]))
            ymax = int(float(parts[4]))
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
            xmin = int(float(parts[1]))
            xmax = int(float(parts[2]))
            ymin = int(float(parts[3]))
            ymax = int(float(parts[4]))
            confidence = float(parts[5])
            size = (xmax-xmin) * (ymax-ymin)
            if size < utils.SIZE_THRESHOLD:
                continue
            if class_id not in detections:
                detections[class_id] = list()
            detections[class_id].append((xmin, xmax, ymin, ymax, confidence))

    return detections


if __name__ == '__main__':
    gt_dir = 'C:/CARLA_Latest/WindowsNoEditor/output/label/Town01_Opt/right_1'
    det_dir = 'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/Town01_Opt/labels/yolov8n_results/right_1/'
    results = classify(gt_dir, det_dir)
    precision, recall = 0.0, 0.0

    for result in results:
        TP = sum(len(v) for v in result['TP'].values())
        FP = sum(len(v) for v in result['FP'].values())
        FN = sum(len(v) for v in result['FN'].values())
        precision += TP / (TP + FP) if (TP + FP) > 0 else 0
        recall += TP / (TP + FN) if (TP + FN) > 0 else 0
    num_frame = len(results)
    precision /= num_frame
    recall /= num_frame
    print(f'Precision: {precision}, Recall: {recall}')
