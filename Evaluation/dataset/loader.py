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
