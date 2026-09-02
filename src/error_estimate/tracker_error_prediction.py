"""
tracker_error_prediction_experiment.py

トラッカーの予測（補正前 = Dest）とモデルの検出（Dcur）を比較し、
det_only（検出のみ）・trk_only（トラックのみ）の件数からフレーム単位で
FP/FNそれぞれの疑いを予測する。実際にGTと比較してFP/FNがあったかどうかと
どれだけ一致するかを、FP推定・FN推定それぞれ別に検証する。

フロー:
1. モデル検出 + トラッキング（1フレームずつ）
2. Dcur（モデル検出）と Dest（トラッカーの補正前予測）をグルーピングし、
   det_only件数が閾値を超えたらFP疑い、trk_only件数が閾値を超えたらFN疑い、
   とそれぞれ独立にフラグを立てる
3. GT とモデル検出を比較し、実際に FP/FN があったかを算出
4. FP推定・FN推定それぞれについて、予測 vs 実際 の一致度を混同行列として集計・評価
"""
import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from src.boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory

from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.dataset import fileReader

from src.ObjectDetection.models.factory import build_model

from src.util.dataset import build_dataset
from src.config import DATASET_DIR, IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker
from src.error_estimate.metrics.group_prediction import group_detection_and_tracking


def count_errors(classified_boxes: list[ClassifiedBoundingBox]) -> tuple[int, int]:
    """
    DetectionClassifier.classify() が返す分類済みboxリストから FP数・FN数 を数える。
    """
    fp = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FP)
    fn = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FN)
    return fp, fn


class TrackerErrorPredictionExperiment:
    """モデルとトラッカーの比較によるフレーム単位エラー予測の妥当性を検証する"""

    def __init__(
        self,
        model_name: str = "yolov8n",
        dataset_name: str = "KITTI",
        map_name: str = "0001",
        error_num_threshold: float = 3,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.map_name = map_name
        self.error_num_threshold = error_num_threshold

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
        self.model = build_model(model_name=model_name, dataset=dataset_name, device="cuda")
        self.tracker = SortTracker()
        self.integrator = ConfidenceBaseIntegrator(iouThreshold=IOU_THRESHOLD, confidenceThreshold=0.0)
        self.classifier = DetectionClassifier(iouThreshold=IOU_THRESHOLD)

        self.output_dir = RESULT_DIR / f"tracker_error_prediction/{dataset_name}/{map_name}/{model_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_records: list[dict] = []

    def run(self):
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}, threshold: {self.error_num_threshold}")
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        self._save_frame_records()
        self._evaluate()

    def _process_frame(self, frame_idx: int, image_path: Path):
        # モデル検出
        model_detections = self.model.predict(image_path)

        # トラッキング更新
        tracking_result = self.tracker.update(model_detections)
        dest = tracking_result.predicted_boxes

        # Step 1: Dcur と Dest の一致度からエラーを予測する
        grouping_result = group_detection_and_tracking(model_detections=model_detections, tracking_detections=dest)
        # if frame_idx < 2:
        #     print(grouping_result)
        # else:
        #     exit()
        

        fp_estimate_flag = len(grouping_result.det_only) > self.error_num_threshold
        fn_estimate_flag = len(grouping_result.trk_only) > self.error_num_threshold

        # Step 2: GT と比較して実際にエラーがあったかを確認する
        gt_filename = image_path.stem + ".txt"
        gt_path = self.dataset.label_dir / gt_filename
        if gt_path.exists():
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))
        else:
            gt_boxes = []

        classified = self.classifier.classify(gt_boxes, model_detections)
        fp_count, fn_count = count_errors(classified)

        self.frame_records.append({
            "frame_idx": frame_idx,
            "det_only_count": len(grouping_result.det_only),
            "trk_only_count": len(grouping_result.trk_only),
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
        """
        FP推定・FN推定それぞれについて、フレーム単位で予測と実際の一致度を評価する。
        det_only優勢によるFP予測、trk_only優勢によるFN予測を独立に検証するため、
        2つの混同行列（FP用・FN用）を別々に算出する。
        """
        fp_stats = self._confusion_matrix(predicted_key="fp_estimate", actual_count_key="fp_count")
        fn_stats = self._confusion_matrix(predicted_key="fn_estimate", actual_count_key="fn_count")

        summary = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "error_num_threshold": self.error_num_threshold,
            "n_frames": len(self.frame_records),
        }
        for prefix, stats in (("fp", fp_stats), ("fn", fn_stats)):
            for key, value in stats.items():
                summary[f"{prefix}_{key}"] = value

        summary_path = self.output_dir / "summary.csv"
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
        description="det_only/trk_only件数によるフレーム単位FP/FN予測の妥当性検証"
    )
    parser.add_argument("--model", type=str, default="yolov8n")
    parser.add_argument("--dataset", type=str, default="KITTI", choices=["CARLA", "KITTI"])
    parser.add_argument("--map", type=str, default="Town02")
    parser.add_argument("--threshold", type=float, default=3)
    args = parser.parse_args()

    experiment = TrackerErrorPredictionExperiment(
        model_name=args.model,
        dataset_name=args.dataset,
        map_name=args.map,
        error_num_threshold=args.threshold,
    )
    experiment.run()