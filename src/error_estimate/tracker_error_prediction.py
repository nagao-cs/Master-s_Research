"""
tracker_error_prediction_experiment.py

トラッカーの予測（補正前 = Dest）とモデルの検出（Dcur）を比較し、
det_only（検出のみ）・trk_only（トラックのみ）を抽出する。
confidenceをもとにエラーだと思われるオブジェクトのみに絞り、
それがオブジェクト単位で実際にエラーであったかを評価する。

フロー:
1. モデル検出 + トラッキング（1フレームずつ）
2. Dcur（モデル検出）と Dest（トラッカーの補正前予測）をグルーピングし、
   det_only のうち confidence が閾値未満のものをFP疑いオブジェクト、
   trk_only のうち confidence が閾値未満のものをFN疑いオブジェクトとする
3. GT とモデル検出を比較し、FP疑い・FN疑いオブジェクトそれぞれについて
   実際にFP/FNだったオブジェクトの数を数える
4. オブジェクト単位で「推定した数」「実際に正しかった数（TP）」「実際に
   存在したエラー数」を全フレームで合算し、precision/recall/f1を算出する
   （TP/FP/FNのカウントは物体単位の集計であり、フレーム単位の混同行列
   （TN込み）とは意味が異なるため、accuracy・TNは扱わない）
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
    """モデルとトラッカーの比較によるオブジェクト単位エラー予測の妥当性を検証する"""

    def __init__(
        self,
        model_name: str = "yolov8n",
        dataset_name: str = "KITTI",
        map_name: str = "0001",
        error_conf_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.map_name = map_name
        self.error_conf_threshold = error_conf_threshold

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
        self.model = build_model(model_name=model_name, dataset=dataset_name, device="cuda")
        self.tracker = SortTracker()
        self.integrator = ConfidenceBaseIntegrator(iouThreshold=IOU_THRESHOLD, confidenceThreshold=0.0)
        self.classifier = DetectionClassifier(iouThreshold=IOU_THRESHOLD)

        self.output_dir = RESULT_DIR / f"tracker_error_prediction/{dataset_name}/{map_name}/{model_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_records: list[dict] = []

    def run(self):
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}, threshold: {self.error_conf_threshold}")
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

        # Step 1: Dcur と Dest をグルーピングし、confidenceでエラー疑いを絞り込む
        grouping_result = group_detection_and_tracking(model_detections=model_detections, tracking_detections=dest)
        # estimated_fp = list(filter(lambda bbox: bbox.confidenceScore < self.error_conf_threshold, grouping_result.det_only))
        # estimated_fn = list(filter(lambda bbox: bbox.confidenceScore < self.error_conf_threshold, grouping_result.trk_only))
        estimated_fp = grouping_result.det_only
        estimated_fn = grouping_result.trk_only

        # Step 2: GT と比較して実際にエラーがあったかを確認する
        gt_filename = image_path.stem + ".txt"
        gt_path = self.dataset.label_dir / gt_filename
        if gt_path.exists():
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))
        else:
            gt_boxes = []

        classified = self.classifier.classify(gt_boxes, model_detections)

        # FP疑いオブジェクトのうち、実際にFPだったものの数
        true_estimate_fp = 0
        fps = list(filter(lambda bbox: bbox.classifyCategory == ClassifyCategory.FP, classified))
        for bbox in estimated_fp:
            if bbox in fps:
                true_estimate_fp += 1

        # FN疑いオブジェクト（トラッカー予測位置）のうち、実際のFN（GT側の見逃し）とIoUで一致するものの数
        true_estimate_fn = 0
        fns = list(filter(lambda bbox: bbox.classifyCategory == ClassifyCategory.FN, classified))
        for bbox in estimated_fn:
            if any(bbox.computeIoU(target_fn) > IOU_THRESHOLD for target_fn in fns):
                true_estimate_fn += 1

        fp_count, fn_count = count_errors(classified)

        self.frame_records.append({
            "frame_idx": frame_idx,
            "det_only_count": len(grouping_result.det_only),
            "trk_only_count": len(grouping_result.trk_only),
            "fp_count": fp_count,
            "fn_count": fn_count,
            "fp_estimate": len(estimated_fp),
            "fn_estimate": len(estimated_fn),
            "true_estimate_fp": true_estimate_fp,
            "true_estimate_fn": true_estimate_fn,
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

    def _precision_recall_f1(self, estimated_key: str, true_key: str, actual_key: str) -> dict:
        """
        オブジェクト単位での推定精度を全フレーム合算で算出する。

        :param estimated_key: フレームごとに推定した怪しいオブジェクト数のキー
        :param true_key: そのうち実際に正しかった（TP）オブジェクト数のキー
        :param actual_key: 実際に存在したエラーオブジェクト数のキー
        """
        total_estimated = sum(r[estimated_key] for r in self.frame_records)
        total_true = sum(r[true_key] for r in self.frame_records)
        total_actual = sum(r[actual_key] for r in self.frame_records)

        precision = total_true / total_estimated if total_estimated else 0.0
        recall = total_true / total_actual if total_actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "estimated": total_estimated,
            "true_positive": total_true,
            "actual": total_actual,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def _evaluate(self):
        """
        FP推定・FN推定それぞれについて、オブジェクト単位の精度を独立に評価する。
        全フレームのTP/推定数/実際数を合算してから precision/recall/f1 を算出する
        """
        fp_stats = self._precision_recall_f1("fp_estimate", "true_estimate_fp", "fp_count")
        fn_stats = self._precision_recall_f1("fn_estimate", "true_estimate_fn", "fn_count")

        summary = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "error_conf_threshold": self.error_conf_threshold,
            "n_frames": len(self.frame_records),
        }
        for prefix, stats in (("fp", fp_stats), ("fn", fn_stats)):
            for key, value in stats.items():
                summary[f"{prefix}_{key}"] = value

        summary_path = RESULT_DIR / "tracker_error_prediction" / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
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
        description="confidenceによるオブジェクト単位FP/FN予測の妥当性検証"
    )
    parser.add_argument("--model", type=str, default="yolov8n")
    parser.add_argument("--dataset", type=str, default="KITTI", choices=["CARLA", "KITTI"])
    parser.add_argument("--map", type=str, default="0001")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    experiment = TrackerErrorPredictionExperiment(
        model_name=args.model,
        dataset_name=args.dataset,
        map_name=args.map,
        error_conf_threshold=args.threshold,
    )
    experiment.run()