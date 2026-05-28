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

from src.ObjectDetection.models.ObjectDetector import Detector
from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox
from src.Evaluation.dataset import fileReader
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.metrics import mAP, f1Score
from src.time_aware_exp.config.config import SortConfig
from src.time_aware_exp.data.dataset import ImageDataset
from .tracker import SortTracker
from .bbox_visualization import draw_detection_boxes, draw_tracker_boxes


class TrackerEvaluator:
    """TrackerとSingle Modelの性能を評価（複数データセット対応）"""
    
    def __init__(
        self,
        model_name: str = "yolov8n",
        dataset_type: str = "CARLA",
        map_name: str = "Town02",
        base_dir: Path = None,
        output_dir: Path = None,
        max_age: int = 1,
        min_hit: int = 1,
        iou_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.map_name = map_name
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.max_age = max_age
        self.min_hit = min_hit
        self.iou_threshold = iou_threshold
        
        # データセット取得（ImageDataset で CARLA/KITTI を自動判別）
        self.dataset = self._load_dataset()
        
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
            max_age=self.max_age,
            min_hits=self.min_hit
        )
        self.tracker = SortTracker(sort_cfg)
        
        # 分類器初期化
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)
        
        # 結果格納用（フレーム別）
        self.frame_data = {}  # {frame_idx: {image, detections, tracker_result, gt_boxes}}
        self.model_classified_boxes = []
        self.tracker_classified_boxes = []

    def _load_dataset(self) -> ImageDataset:
        """
        データセットタイプに応じて適切なディレクトリからデータセットをロード
        """
        if self.dataset_type == "CARLA":
            image_dir = self.base_dir / "output" / "image" / self.map_name / "original" / "front"
            gt_dir = self.base_dir / "output" / "label" / self.map_name / "front"
        elif self.dataset_type == "KITTI":
            # KITTI は data_tracking フォーマット（シーケンス 0020 を使用）
            image_dir = self.base_dir.parent.parent.parent / "d" / "data_tracking_image_2" / "training" / "image_02" / "0020"
            gt_dir = self.base_dir.parent.parent.parent / "d" / "data_tracking_label_2" / "training" / "label_02" / "0020"
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported. Use 'CARLA' or 'KITTI'")
        
        print(f"Loading dataset: {self.dataset_type}")
        print(f"  Image dir: {image_dir}")
        print(f"  GT dir: {gt_dir}")
        
        return ImageDataset(image_dir=image_dir, gt_dir=gt_dir)
    
    def run_evaluation(self):
        """評価を3つのフェーズで実行"""
        print(f"Starting evaluation...")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_type}")
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
        for frame_idx, image_path in enumerate(tqdm(self.dataset)):
            # 検出実行
            model_detections = self.model.predict(image_path)
            
            # トラッキング更新
            tracker_result = self.tracker.update(model_detections)
            
            # フレーム結果を保存
            self.frame_data[frame_idx] = {
                'image_path': image_path,
                'model_detections': model_detections,
                'tracker_result': tracker_result,
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
            gt_path = self.dataset.gt_dir / gt_filename
            
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
            
            self.model_classified_boxes.extend(model_classified)
            self.tracker_classified_boxes.extend(tracker_classified)
            
            # ビジュアライゼーション
            self._visualize_frame(frame_idx, frame)

    def _visualize_frame(self, frame_idx: int, frame: dict):
        """フレーム毎の可視化（色分けbbox描画）"""
        vis_image = cv2.imread(frame["image_path"])
        
        # モデル検出：青
        vis_image = draw_detection_boxes(
            vis_image, 
            frame['model_detections'], 
            color=(255, 0, 0), 
            label_prefix="Model"
        )
        
        # Tracker結果：赤
        vis_image = draw_tracker_boxes(
            vis_image, 
            frame['tracker_result'], 
            color=(0, 0, 255), 
            label_prefix="Tracker"
        )
        
        # 保存
        output_path = self.vis_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(output_path), vis_image)

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
    parser.add_argument("--max_age", type=int, default=1, help="Max age for tracker (default: 1)")
    parser.add_argument("--min_hit", type=int, default=1, help="Min hits for tracker (default: 1)")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    cwd: Path = Path(__file__).parent  # tracker フォルダ
    base_dir = cwd.parent.parent.parent  # プロジェクトルート
    output_dir = cwd / "result" / f"{args.dataset}_{args.map}" if args.dataset == "CARLA" else cwd / "result" / "KITTI"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = TrackerEvaluator(
        model_name=args.model,
        dataset_type=args.dataset,
        map_name=args.map,
        base_dir=base_dir,
        output_dir=output_dir,
        max_age=args.max_age,
        min_hit=args.min_hit,
        iou_threshold=args.iou_threshold
    )
    evaluator.run_evaluation()