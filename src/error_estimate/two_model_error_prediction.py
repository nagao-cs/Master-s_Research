"""
two_model_error_prediction_experiment.py

2モデル体制でのフレーム単位FP/FN予測の妥当性を検証する。

責務分担:
- grouping.py: モデルの検出とそのトラッカー予測をグルーピングし、
  agreed / det_only / trk_only に分ける（分類）
- TwoModelTrackIntegrator: グルーピング結果を使って最終的な統合検出結果を
  作る（voting）
- two_model_error_estimator: グルーピング結果（det_only / trk_only）を使って
  解消されずに残ったFP/FN疑いの件数を推定する
このスクリプトは上記3つを順に呼び出し、残数を閾値でエラーフラグに変換した上で、
Affirmative統合された最終出力とGTを比較して実際のFP/FNとの一致度を評価する。

NOTE: 一致しなかったFN候補（＝どちらのモデルもそもそも検出していない）は、
「片方のトラッカーには一時的に痕跡があったが、両モデルとも見逃している」
ケースの検知には使えるが、モデルが一度も捉えたことのない継続的な見逃し
（遠方・恒常的オクルージョン等）はtrk_only候補にすら現れないため、
この手法では原理的に検知できない。
"""
import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from src.boundingBox.boundingBox import ClassifiedBoundingBox

from src.Evaluation.classifier.detectionClassifier import DetectionClassifier, ClassifyCategory
from src.Evaluation.dataset import fileReader

from src.ObjectDetection.models.factory import build_model

from src.util.dataset import build_dataset
from src.config import RESULT_DIR, IOU_THRESHOLD
from src.tracker.tracker import SortTracker
from src.BBox_Integrator.grouping import group_detection_and_tracking
from src.BBox_Integrator.two_model_track_integrator import TwoModelTrackIntegrator
from src.error_estimate.metrics.two_model_error_estimate import estimate_two_model_errors


def count_errors(classified_boxes: list[ClassifiedBoundingBox]) -> tuple[int, int]:
    """
    DetectionClassifier.classify() が返す分類済みboxリストから FP数・FN数 を数える。
    """
    fp = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FP)
    fn = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FN)
    return fp, fn


class TwoModelErrorPredictionExperiment:
    """2モデル体制でのフレーム単位FP/FN予測の妥当性を検証する"""

    def __init__(
        self,
        model_a_name: str = "yolov8n",
        model_b_name: str = "ssd",
        dataset_name: str = "KITTI",
        map_name: str = "Town02",
        error_num_threshold: float = 1,
        iou_threshold: float = IOU_THRESHOLD,
    ):
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.dataset_name = dataset_name
        self.map_name = map_name
        self.error_num_threshold = error_num_threshold
        self.iou_threshold = iou_threshold

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)

        self.model_a = build_model(model_name=model_a_name, dataset=dataset_name, device="cuda")
        self.model_b = build_model(model_name=model_b_name, dataset=dataset_name, device="cuda")
        self.tracker_a = SortTracker()
        self.tracker_b = SortTracker()

        # 最終出力（voting）専任のIntegrator。エラー推定はここでは行わない
        self.integrator = TwoModelTrackIntegrator(iouThreshold=iou_threshold)
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)

        self.output_dir = (
            RESULT_DIR
            / f"two_model_error_prediction/{dataset_name}/{map_name}/{model_a_name}_{model_b_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_records: list[dict] = []

    def run(self):
        print(
            f"Model A: {self.model_a_name}, Model B: {self.model_b_name}, "
            f"Dataset: {self.dataset_name}, threshold: {self.error_num_threshold}"
        )
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        self._save_frame_records()
        self._evaluate()

    def _process_frame(self, frame_idx: int, image_path: Path):
        # モデルA・Bそれぞれの検出
        detections_a = self.model_a.predict(image_path)
        detections_b = self.model_b.predict(image_path)

        # それぞれのトラッカー更新（predicted_boxes = 補正前予測 = Dest）
        dest_a = self.tracker_a.update(detections_a).predicted_boxes
        dest_b = self.tracker_b.update(detections_b).predicted_boxes

        # 1. grouping: モデルごとに検出とトラッカーを分類する
        grouping_a = group_detection_and_tracking(
            model_detections=detections_a, tracking_detections=dest_a, iou_threshold=self.iou_threshold
        )
        grouping_b = group_detection_and_tracking(
            model_detections=detections_b, tracking_detections=dest_b, iou_threshold=self.iou_threshold
        )

        # 2. integrator: 最終的な統合検出結果を構築する（voting）
        combined_result = self.integrator.execute(
            modelADetections=detections_a,
            modelATracking=dest_a,
            modelBDetections=detections_b,
            modelBTracking=dest_b,
        )

        # 3. error_estimator: グルーピング結果からFP/FN疑いの残数を推定する
        error_estimate = estimate_two_model_errors(
            grouping_a=grouping_a,
            grouping_b=grouping_b,
            model_a_detections=detections_a,
            model_a_tracking=dest_a,
            model_b_detections=detections_b,
            model_b_tracking=dest_b,
            iou_threshold=self.iou_threshold,
        )
        residual_fp_count = error_estimate["residual_fp_count"]
        residual_fn_count = error_estimate["residual_fn_count"]

        # 残った怪しい数がそれぞれ閾値を超えたらエラーありと予測する
        fp_estimate_flag = residual_fp_count >= self.error_num_threshold
        fn_estimate_flag = residual_fn_count >= self.error_num_threshold

        # GTと比較して実際のFP/FNを求める
        gt_filename = image_path.stem + ".txt"
        gt_path = self.dataset.label_dir / gt_filename
        if gt_path.exists():
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))
        else:
            gt_boxes = []

        classified = self.classifier.classify(gt_boxes, combined_result)
        fp_count, fn_count = count_errors(classified)

        self.frame_records.append({
            "frame_idx": frame_idx,
            "residual_fp_count": residual_fp_count,
            "residual_fn_count": residual_fn_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "fp_estimate": fp_estimate_flag,
            "fn_estimate": fn_estimate_flag,
        })

    def _save_frame_records(self):
        if not self.frame_records:
            return
        path = self.output_dir / "frame_records.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.frame_records[0].keys()))
            writer.writeheader()
            writer.writerows(self.frame_records)
        print(f"Frame-level records saved to {path}")

    def _confusion_matrix(self, predicted_key: str, actual_count_key: str) -> dict:
        """
        predicted_key（bool の推定フラグ）と actual_count_key（int、0より大きければ
        実際にエラーあり）から、混同行列と precision/recall/f1/accuracy を算出する。
        FP推定・FN推定のどちらの評価にも共通で使う。
        """
        tp = fp = fn = tn = 0
        for r in self.frame_records:
            predicted = r[predicted_key]
            actual = r[actual_count_key] > 0

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(self.frame_records) if self.frame_records else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }

    def _evaluate(self):
        """FP推定・FN推定それぞれについて、予測と実際の一致度を独立に評価する"""
        fp_stats = self._confusion_matrix(predicted_key="fp_estimate", actual_count_key="fp_count")
        fn_stats = self._confusion_matrix(predicted_key="fn_estimate", actual_count_key="fn_count")

        summary = {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "dataset": self.dataset_name,
            "error_num_threshold": self.error_num_threshold,
            "n_frames": len(self.frame_records),
        }
        for prefix, stats in (("fp", fp_stats), ("fn", fn_stats)):
            for key, value in stats.items():
                summary[f"{prefix}_{key}"] = value

        summary_path = RESULT_DIR / f"two_model_error_prediction" / "summary.csv"
        file_exists = summary_path.exists()
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

        print(f"\nSummary saved to {summary_path}")
        for k, v in summary.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2モデル体制でのフレーム単位FP/FN予測の妥当性検証"
    )
    parser.add_argument("--model-a", type=str, default="yolov8n")
    parser.add_argument("--model-b", type=str, default="ssd")
    parser.add_argument("--dataset", type=str, default="KITTI", choices=["CARLA", "KITTI"])
    parser.add_argument("--map", type=str, default="0001")
    parser.add_argument("--threshold", type=float, default=1)
    args = parser.parse_args()

    experiment = TwoModelErrorPredictionExperiment(
        model_a_name=args.model_a,
        model_b_name=args.model_b,
        dataset_name=args.dataset,
        map_name=args.map,
        error_num_threshold=args.threshold,
    )
    experiment.run()