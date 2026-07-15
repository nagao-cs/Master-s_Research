"""
tracker_evaluation.py
tracker とモデル検出の性能を比較・評価
CARLA と KITTI の両方のデータセットに対応

フェーズ分離:
1. 検出 + トラッキング（全フレーム）→ 結果保存
2. 分類 + 可視化（全フレーム）
3. メトリクス計算
"""
import os
import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import time

from src.boundingBox.integrator.integrator import Integrator
from src.boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from src.ObjectDetection.models.ObjectDetector import Detector
from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox
from src.Evaluation.dataset import fileReader
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.metrics import mAP, f1Score
from src.time_aware_exp.config.config import SortConfig
from .tracker import SortTracker, MAX_AGE, MIN_HITS, IOU_THRESHOLD
from .bbox_visualization import draw_detection_boxes, draw_tracker_boxes

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.ObjectDetection.models.factory import build_model
from src.util.dataset import build_dataset, Dataset
from src.config import DATASET_DIR, KITTI_IMAGE_HEIGHT, KITTI_IMAGE_WIDTH

def calc_jaccard(bbox_groups: list[list[DetectionBoundingBox]]) -> float:
        if not bbox_groups:
            return 1.0
        agreed = sum(1 for g in bbox_groups if len(g) == 2)
        return agreed / len(bbox_groups)
    
class TrackerEvaluator:
    """TrackerとSingle Modelの性能を評価"""
    
    def __init__(
        self,
        model_name: str = "yolov8n",
        integrator: str = "affirmative",
        dataset_name: str = "KITTI",
        map_name: str = "Town02",
    ):
        self.model_name = model_name
        self.integrator: Integrator = ConfidenceBaseIntegrator(iouThreshold=IOU_THRESHOLD, confidenceThreshold=0.0)
        self.dataset_name = dataset_name
        self.map_name = map_name
        
        # データセット取得
        self.dataset = build_dataset(dataset_name=self.dataset_name, map_name=self.map_name)
        
        # 出力ディレクトリ作成
        self.output_dir = DATASET_DIR / "tracker"
        self.model_result_dir = self.output_dir / f"{model_name}"
        self.model_image_dir = self.model_result_dir / "images"
        self.model_label_dir = self.model_result_dir / "labels"
        self.tracker_result_dir = self.output_dir / f"{model_name}_tracker"
        self.tracker_image_dir = self.tracker_result_dir / "images"
        self.tracker_label_dir = self.tracker_result_dir / "labels"

        self.output_dir.mkdir(parents=True, exist_ok=True) 
        self.model_result_dir.mkdir(parents=True, exist_ok=True)
        self.model_image_dir.mkdir(parents=True, exist_ok=True)
        self.model_label_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_result_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_image_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_label_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルロード
        self.model = build_model(model_name=model_name, dataset=dataset_name, device="cuda")
        
        self.tracker = SortTracker()
        
        # 分類器初期化
        self.classifier = DetectionClassifier(iouThreshold=IOU_THRESHOLD)
        
        # 結果格納用（フレーム別）
        self.frame_data = {}  # {frame_idx: {image, detections, tracker_result, gt_boxes}}
        self.model_classified_boxes = []
        self.tracker_classified_boxes = []
        self.combined_classified_boxes = []

    def run_evaluation(self):
        """評価を3つのフェーズで実行"""
        print(f"Starting evaluation...")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
        print(f"Total frames: {len(self.dataset)}")
        
        # === フェーズ 1: 検出とトラッキング ===
        print("\n[Phase 1/3] Detection & Tracking...")
        self._phase1_detect_and_track()
        
        # === フェーズ 2: 分類と可視化 ===
        print("\n[Phase 2/3] Classification & Visualization...")
        self._phase2_classify_and_visualize()
        
        # === フェーズ 3: メトリクス計算 ===
        print("\n[Phase 3/3] Computing Metrics...")
        self._phase3_compute_metrics()
        
        print(f"\nEvaluation complete. Results saved to {self.output_dir}")

    def _phase1_detect_and_track(self):
        """
        フェーズ 1: すべてのフレームに対して検出とトラッキングを実行
        """
        for frame_idx, (image_path, label_path) in enumerate(tqdm(iterable=self.dataset)):
            # 検出実行
            model_detections = self.model.predict(image_path)
            
            # トラッキング更新
            tracker_result = self.tracker.update(model_detections)
            
            # 検出とトラッキングを組み合わせた結果
            combined_result, bbox_groups = self.integrator.execute({"model": model_detections, "tracking": tracker_result})
            
            # フレーム結果を保存
            self.frame_data[frame_idx] = {
                'image_path': image_path,
                'model_detections': model_detections,
                'tracker_result': tracker_result,
                "bbox_groups": bbox_groups,
                "combined_result": combined_result,
                'gt_boxes': None
            }

    def _phase2_classify_and_visualize(self):
        """
        フェーズ 2: 全フレーム結果に対して分類と可視化を実行
        """
        for frame_idx in tqdm(sorted(self.frame_data.keys())):
            frame = self.frame_data[frame_idx]
            
            # グラウンドトゥルースパスを生成
            gt_filename = frame['image_path'].stem + ".txt"
            gt_path = self.dataset.label_dir / gt_filename
            
            # グラウンドトゥルース読み込み
            if not gt_path.exists():
                print(f"Warning: GT file not found {gt_path}")
                gt_boxes = []
            else:
                gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(
                    str(gt_path)
                )
            
            frame['gt_boxes'] = gt_boxes
            
            # 分類（TP/FP/FN判定）
            model_classified = self.classifier.classify(gt_boxes, frame['model_detections'])
            tracker_classified = self.classifier.classify(gt_boxes, frame['tracker_result'])
            combined_classified = self.classifier.classify(gt_boxes, frame["combined_result"])
            
            self.model_classified_boxes.extend(model_classified)
            self.tracker_classified_boxes.extend(tracker_classified)
            self.combined_classified_boxes.extend(combined_classified)
            
            base_image = cv2.imread(str(frame["image_path"]))
            if base_image is None:
                raise RuntimeError("failed to read image")
            image_height, image_width = KITTI_IMAGE_HEIGHT, KITTI_IMAGE_WIDTH
            
            # ビジュアライゼーション
            self._save_labels(frame_idx, frame, image_height, image_width)
            self._save_images(frame_idx, frame, base_image)
        
    def _save_labels(self, frame_idx: int, frame: dict, image_height: int, image_width: int):
        """
        フレーム毎のラベルをKITTI形式に近いテキストファイルとして保存する。
        model / tracker をそれぞれ別ディレクトリに保存する。

        DetectionBoundingBox は xCenter, yCenter, width, height（正規化 0-1 center形式）、
        classId, confidenceScore を持つ（ObjectDetector.py / tracker.py 参照）。
        ここでは image_height / image_width を使ってpixel絶対座標の x1 y1 x2 y2 に変換し、
        1行1boxで "class_id confidence x1 y1 x2 y2" の形式で書き出す。
        """
        filename = frame["image_path"].stem + ".txt"

        def write_boxes(boxes: list[DetectionBoundingBox], output_path):
            with open(output_path, "w") as f:
                for box in boxes:
                    f.write(
                        f"{box.classId} {box.xCenter:.3f} {box.yCenter:.3f} {box.width:.3f} {box.height:.3f} {box.confidenceScore:.6f}\n"
                    )

        write_boxes(frame["model_detections"], self.model_label_dir / filename)
        write_boxes(frame["tracker_result"], self.tracker_label_dir / filename)

    def _save_images(self, frame_idx: int, frame: dict, base_image: np.ndarray):

        # モデル検出のみ描画（青）
        model_vis_image = draw_detection_boxes(
            base_image.copy(),
            frame['model_detections'],
            color=(255, 0, 0),
            label_prefix="Model"
        )
        model_output_path = self.model_image_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(model_output_path), model_vis_image)

        # Tracker結果のみ描画（赤）
        tracker_vis_image = draw_tracker_boxes(
            base_image.copy(),
            frame['tracker_result'],
            color=(0, 0, 255),
            label_prefix="Tracker"
        )
        tracker_output_path = self.tracker_image_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(tracker_output_path), tracker_vis_image)

    def _phase3_compute_metrics(self):
        """
        フェーズ 3: メトリクス計算と結果出力
        """
        targetClassIdList = [0, 2, 9, 11]
        
        # モデルのメトリクス
        model_mAP, model_ap_dict = mAP.computeMeanAP(
            self.model_classified_boxes, targetClassIdList
        )
        model_f1, model_prec, model_rec = f1Score.computeF1Score(
            self.model_classified_boxes
        )
        
        # Trackerのメトリクス
        tracker_mAP, tracker_ap_dict = mAP.computeMeanAP(
            self.tracker_classified_boxes, targetClassIdList
        )
        tracker_f1, tracker_prec, tracker_rec = f1Score.computeF1Score(
            self.tracker_classified_boxes
        )
        
        
        jaccard = sum(list(calc_jaccard(self.frame_data[frame_idx]["bbox_groups"]) for frame_idx in self.frame_data.keys())) / len(self.frame_data)

        # CSV保存
        csv_path = self.output_dir / "metrics_summary.csv"
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Method', 'mAP', 'F1', 'Precision', 'Recall', 'Jaccard'])
            writer.writerow([
                self.model_name,
                f"{model_mAP:.4f}", f"{model_f1:.4f}", f"{model_prec:.4f}", f"{model_rec:.4f}",
                ""
            ])
            writer.writerow([
                f"{self.model_name}_tracker",
                f"{tracker_mAP:.4f}", f"{tracker_f1:.4f}", f"{tracker_prec:.4f}", f"{tracker_rec:.4f}",
                f"{jaccard:.4f}"
            ])
        
        print(f"\nMetrics saved to {csv_path}")
        print(f"\n{'='*60}")
        print(f"{'Results':^60}")
        print(f"{'='*60}")
        print(f"{'Method':<20} {'mAP':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
        print(f"{'-'*60}")
        print(f"{'Model':<20} {model_mAP:<12.4f} {model_f1:<12.4f} {model_prec:<12.4f} {model_rec:<12.4f}")
        print(f"{'Tracker':<20} {tracker_mAP:<12.4f} {tracker_f1:<12.4f} {tracker_prec:<12.4f} {tracker_rec:<12.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Tracker Performance on CARLA or KITTI")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model name (default: yolov8n)")
    parser.add_argument("--dataset", type=str, default="CARLA", choices=["CARLA", "KITTI"],
                        help="Dataset type: CARLA or KITTI (default: CARLA)")
    parser.add_argument("--map", type=str, default="Town02", help="CARLA map name (default: Town02)")
    
    args = parser.parse_args()
    
    evaluator = TrackerEvaluator(
        model_name=args.model,
        dataset_name=args.dataset,
        map_name=args.map,
    )
    evaluator.run_evaluation()