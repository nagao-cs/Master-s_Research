from typing import Iterator
from utils import utils
import os


class DetectionResultLoader:
    def __init__(self, gt_dir, det_dirs):
        self.gt_dir = gt_dir
        self.det_dirs = det_dirs
        self.num_version = len(det_dirs)

    def iter_frame(self) -> Iterator[tuple[dict, dict[dict]]]:
        gt_files = [os.path.join(self.gt_dir, f)
                    for f in os.listdir(self.gt_dir)]
        det_files_dict = {version: [os.path.join(det_dir, det_file) for det_file in os.listdir(det_dir)]
                          for version, det_dir in enumerate(self.det_dirs)}

        for frame_idx, gt_file in enumerate(gt_files):
            gt = self._get_gt(gt_file)
            dets = {version: self._get_detections(det_files_dict[version][frame_idx])
                    for version in range(self.num_version)}
            yield frame_idx, gt, dets

    def _classify_frame(self, gt_path, det_paths) -> dict[dict]:
        frame_results = dict()
        # 各バージョンの検出結果を分類
        gt = self._get_gt(gt_path)
        for version, det_path in enumerate(det_paths):
            det = self._get_detections(det_path)
            frame_results[version] = self._classify(gt, det)
        return frame_results

    def _classify(self, gt, det) -> dict:
        det_results = {'TP': dict(), 'FP': dict(), 'FN': dict()}
        # クラスごとに処理
        for class_id in gt.keys():
            gt_boxes = gt[class_id]
            det_boxes = det.get(class_id, [])

            if class_id not in det_results['TP']:
                det_results['TP'][class_id] = list()
            if class_id not in det_results['FN']:
                det_results['FN'][class_id] = list()
            if class_id not in det_results['FP']:
                det_results['FP'][class_id] = list()

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
                    det_results['TP'][class_id].append(det_box)
                    used_gt.add(best_gt_idx)
                else:
                    det_results['FP'][class_id].append(det_box)

            # 未使用のGTをFNとして追加
            for i, gt_box in enumerate(gt_boxes):
                if i not in used_gt:
                    det_results['FN'][class_id].append(gt_box)
        return det_results

    def _get_gt(self, gt_path) -> dict:
        gt = dict()
        with open(gt_path, 'r') as gt_file:
            lines = gt_file.readlines()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split(' ')
                class_id = utils.class_Map.get(
                    (int(parts[0])), -1)  # -1（無視するクラス）
                if class_id == -1:
                    continue
                # class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                distance = 0.0  # 仮の値、必要に応じて計算する
                size = width * height * utils.IM_WIDTH * utils.IM_HEIGHT
                if size < utils.SIZE_THRESHOLD:
                    continue
                if class_id not in gt:
                    gt[class_id] = list()
                # 仮の値、必要に応じて計算する
                gt[class_id].append(
                    (x_center, y_center, width, height, distance))

        return gt

    def _get_detections(self, det_path) -> dict:
        """
        det_file_path: Path to a detection results file
        """
        detections = dict()
        with open(det_path, 'r') as det_file:
            lines = det_file.readlines()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split(' ')
                class_id = utils.class_Map.get(
                    (int(parts[0])), -1)  # -1（無視するクラス）
                if class_id == -1:
                    continue
                # class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5])
                size = width * height * utils.IM_WIDTH * utils.IM_HEIGHT
                if size < utils.SIZE_THRESHOLD:
                    continue
                if confidence < utils.CONF_THRESHOLD:
                    continue
                if class_id not in detections:
                    detections[class_id] = list()
                detections[class_id].append(
                    (x_center, y_center, width, height, confidence))

        return detections
