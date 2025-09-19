import os

class_Map = {
    0: 0,  # pedestrian
    1: 2,  # bicycle
    2: 2,  # motorcycle
    3: 2,  # car
    5: 2,  # bus
    7: 2,  # truck
    9: 9,  # traffic light
    # 11: 11, #stop sign
}
SIZE_THRESHOLD = 200  # 物体のサイズの閾値（ピクセル数）


def classify(gt_dir, det_dir):
    """
    gt_dir: Ground Truth directory
    det_dir: Detection results directory
    """
    results = list()
    gt_files = [os.path.join(gt_dir, gt_file)
                for gt_file in os.listdir(gt_dir)]
    det_files = [os.path.join(det_dir, det_file)
                 for det_file in os.listdir(det_dir)]

    for gt_file, det_file in zip(gt_files, det_files):


def get_gt(gt_file_path):
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
            class_id = class_Map.get((int(parts[0])), -1)  # -1（無視するクラス）
            if class_id == -1:
                continue
            xmin = int(float(parts[1]))
            xmax = int(float(parts[2]))
            ymin = int(float(parts[3]))
            ymax = int(float(parts[4]))
            distance = 0.0  # 仮の値、必要に応じて計算する
            size = (xmax-xmin) * (ymax-ymin)
            if size < SIZE_THRESHOLD:
                continue
            if class_id not in gt:
                gt[class_id] = list()
            gt[class_id].append((xmin, xmax, ymin, ymax))

    return gt


def get_detections(det_file_path):
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
            class_id = class_Map.get((int(parts[0])), -1)  # -1（無視するクラス）
            if class_id == -1:
                continue
            xmin = int(float(parts[1]))
            xmax = int(float(parts[2]))
            ymin = int(float(parts[3]))
            ymax = int(float(parts[4]))
            confidence = float(parts[5])
            size = (xmax-xmin) * (ymax-ymin)
            if size < SIZE_THRESHOLD:
                continue
            if class_id not in detections:
                detections[class_id] = list()
            detections[class_id].append((xmin, xmax, ymin, ymax))

    return detections
