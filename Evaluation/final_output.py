from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from integrate import DetectionIntegrator
from detection import Detection
import utils


class Output:
    def __init__(self, det_dirs: List[str], iou_threshold: float = 0.5, confidence_threshold: float = 0.3):
        self.det_dirs = det_dirs
        self.num_models = len(det_dirs)
        # frame_id -> detections
        self.detections: Dict[int, List[Detection]] = {}
        self.integrator = DetectionIntegrator(self.num_models,
                                              iou_threshold, confidence_threshold)
        self.integrated_results: Dict[int, List[Detection]] = {}

    def load_detections(self):
        """各モデルの検出結果をロード"""
        for model_id, det_dir in enumerate(self.det_dirs):
            # YOLOフォーマットの検出結果を読み込み
            detections = self._load_model_detections(det_dir)
            for det in detections:
                frame_id = det.frame_id
                if frame_id not in self.detections:
                    self.detections[frame_id] = []
                self.detections[frame_id].append(det)

    def _load_model_detections(self, det_dir: str) -> List[Detection]:
        """指定されたディレクトリからモデルの検出結果を読み込む"""
        model_detections = []
        # ここでYOLOフォーマットのファイルを読み込み、Detectionオブジェクトを作成
        import os
        for filename in os.listdir(det_dir):
            if not filename.endswith('.txt'):
                continue
            frame_id = int(filename.split('.')[0])
            filepath = os.path.join(det_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split(' ')
                    if len(parts) < 6:
                        print("Invalid line:", line)
                        print("in file:", filepath)
                    class_id = utils.class_Map.get(
                        (int(parts[0])), -1)  # -1（無視するクラス）
                    if class_id == -1:
                        continue
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5])

                    bbox = np.array([x_center, y_center, width, height])

                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        frame_id=frame_id,
                        model_id=len(self.det_dirs)  # 仮のモデルID
                    )
                    model_detections.append(detection)
        return model_detections

    def process(self):
        """検出結果のロードと統合を実行"""
        # 検出結果のロード
        self.load_detections()
        # フレームごとに統合
        for frame_id, frame_detections in self.detections.items():
            self.integrated_results[frame_id] = self.integrator.integrate_frame(
                frame_detections)

    def save_results(self, output_dir: str):
        """統合結果をYOLOフォーマットで保存"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        for frame_id, detections in self.integrated_results.items():
            output_path = os.path.join(output_dir, f"{frame_id:06d}.txt")
            with open(output_path, "w") as f:
                for det in detections:
                    # YOLOフォーマット: class_id x_center y_center width height confidence
                    x_center, y_center, width, height = det.bbox
                    f.write(
                        f"{det.class_id} {x_center} {y_center} {width} {height} {det.confidence}\n")


def main():
    import argparse
    argparser = argparse.ArgumentParser(
        description="Evaluate object detection results")
    argparser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode"
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03",
                 "Town04", "Town05", "Town10HD_Opt"],
        help="Map name: Town01, Town02, Town04, Town05, Town10HD",
        required=True
    )
    argparser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr"],
    )
    args = argparser.parse_args()
    debug = args.debug
    map = args.map
    models = args.models
    print(f"map: {map}")
    gt_dir = f'C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front'
    version = len(models)
    print(f"models: {models}")
    det_dirs = [
        f'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{model}/front' for model in models]

    output_dir = f'C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\Integrated_output\{map}_{"_".join(models)}'
    output_processor = Output(det_dirs)
    output_processor.process()
    output_processor.save_results(output_dir)


if __name__ == "__main__":
    main()
