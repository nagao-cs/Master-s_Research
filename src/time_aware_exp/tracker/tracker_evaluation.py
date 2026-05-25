"""
tracker_evaluation.py
tracker とモデル検出の性能を比較・評価
"""
import os
import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import time

from src.ObjectDetection.models.ObjectDetector import Detector
from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox
from src.Evaluation.dataset import fileReader
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.metrics import mAP, f1Score
from src.time_aware_exp.config.config import SortConfig
from .tracker import SortTracker
from .bbox_visualization import draw_detection_boxes, draw_tracker_boxes

class TrackerEvaluator:
    """TrackerとSingle Modelの性能を評価"""
    
    def __init__(
        self,
        model_name: str = "yolov8n",
        map_name: str = "Town02",
        ground_truth_dir: str = None,
        input_image_dir: str = None,
        output_dir: str = None,
        iou_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.map_name = map_name
        self.ground_truth_dir = Path(ground_truth_dir)
        self.input_image_dir = Path(input_image_dir)
        self.output_dir = Path(output_dir or "src/time_aware_exp/tracker/result")
        self.iou_threshold = iou_threshold
        
        # 出力ディレクトリ作成
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルロード
        if model_name == "yolov8n":
            from src.ObjectDetection.models.Yolov8n import Yolov8nDetector
            self.model = Yolov8nDetector()
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Tracker初期化
        sort_cfg = SortConfig(
            iou_threshold=self.iou_threshold,
            max_age=1,
            min_hits=1
        )
        self.tracker = SortTracker(sort_cfg)
        
        # 分類器初期化
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)
        
        # メトリクス記録用
        self.frame_metrics = []
        self.model_classified_boxes = []
        self.tracker_classified_boxes = []
    
    def run_evaluation(self):
        """評価実行"""
        print(f"Starting evaluation...")
        print(f"Model: {self.model_name}, Map: {self.map_name}")
        
        # 画像ファイル取得
        image_files = sorted([
            f for f in os.listdir(self.input_image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # グラウンドトゥルース取得
        gt_files = sorted([
            f for f in os.listdir(self.ground_truth_dir)
            if f.endswith('.txt')
        ])
        
        min_frames = min(len(image_files), len(gt_files))
        
        for frame_idx in tqdm(range(min_frames)):
            # 画像読み込み
            image_path = self.input_image_dir / image_files[frame_idx]
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            image_height, image_width = image.shape[:2]
            
            # グラウンドトゥルース読み込み
            gt_path = self.ground_truth_dir / gt_files[frame_idx]
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))
            
            # モデル検出
            model_detections = self.model.predict(image_path)
            
            # Tracker更新
            tracker_result = self.tracker.update(model_detections)
            
            # 分類（TP/FP/FN判定）
            model_classified = self.classifier.classify(gt_boxes, model_detections)
            tracker_classified = self.classifier.classify(gt_boxes, tracker_result)
            
            self.model_classified_boxes.extend(model_classified)
            self.tracker_classified_boxes.extend(tracker_classified)
            
            # ビジュアライゼーション（色分け）
            self._visualize_frame(
                image, frame_idx, gt_boxes, model_detections, tracker_result
            )
        
        # メトリクス計算
        self._compute_metrics()
        
        print(f"Evaluation complete. Results saved to {self.output_dir}")
    
    def _visualize_frame(self, image, frame_idx, gt_boxes, model_dets, tracker_boxes):
        """フレーム毎の可視化（色分けbbox描画）"""
        vis_image = image.copy()
        
        # # グラウンドトゥルース：グリーン
        # for gt_box in gt_boxes:
        #     vis_image = draw_detection_boxes(
        #         vis_image, [gt_box], color=(0, 255, 0), label_prefix="GT"
        #     )
        
        # モデル検出：青
        vis_image = draw_detection_boxes(
            vis_image, model_dets, color=(255, 0, 0), label_prefix="Model"
        )
        
        # Tracker結果：赤
        vis_image = draw_tracker_boxes(
            vis_image, tracker_boxes, color=(0, 0, 255), label_prefix="Tracker"
        )
        
        # 保存
        output_path = self.vis_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(output_path), vis_image)
    
    def _compute_metrics(self):
        """メトリクス計算"""
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
        
        # CSV保存
        csv_path = self.output_dir / "metrics_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Method', 'mAP', 'F1', 'Precision', 'Recall',
                'AP_pedestrian', 'AP_vehicle', 'AP_traffic_light', 'AP_traffic_sign'
            ])
            writer.writerow([
                'Model',
                f"{model_mAP:.4f}", f"{model_f1:.4f}", f"{model_prec:.4f}", f"{model_rec:.4f}",
                f"{model_ap_dict.get(0, 0):.4f}",
                f"{model_ap_dict.get(2, 0):.4f}",
                f"{model_ap_dict.get(9, 0):.4f}",
                f"{model_ap_dict.get(11, 0):.4f}"
            ])
            writer.writerow([
                'Tracker',
                f"{tracker_mAP:.4f}", f"{tracker_f1:.4f}", f"{tracker_prec:.4f}", f"{tracker_rec:.4f}",
                f"{tracker_ap_dict.get(0, 0):.4f}",
                f"{tracker_ap_dict.get(2, 0):.4f}",
                f"{tracker_ap_dict.get(9, 0):.4f}",
                f"{tracker_ap_dict.get(11, 0):.4f}"
            ])
        
        print(f"Metrics saved to {csv_path}")
        print(f"\n=== Results ===")
        print(f"Model  mAP: {model_mAP:.4f}, F1: {model_f1:.4f}")
        print(f"Tracker mAP: {tracker_mAP:.4f}, F1: {tracker_f1:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Tracker Performance")
    parser.add_argument("--model", type=str, default="yolov8n")
    parser.add_argument("--map", type=str, default="Town02")
    parser.add_argument("--max_age", type=int, default=1)
    parser.add_argument("--min_hit", type=int, default=1)
    
    args = parser.parse_args()
    
    cwd: Path = Path(__file__).parent # tracker
    base_dir = cwd.parent.parent.parent
    gt_dir = base_dir / "output" / "label" / args.map / "front"
    image_dir = base_dir / "output" / "image" / args.map / "original" / "front"
    output_dir = cwd / "result"
    
    
    evaluator = TrackerEvaluator(
        model_name=args.model,
        map_name=args.map,
        ground_truth_dir=gt_dir,
        input_image_dir=image_dir,
        output_dir=output_dir,
        max_age=args.max_age,
        min_hit=args.min_hit
    )
    evaluator.run_evaluation()