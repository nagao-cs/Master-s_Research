"""1モデル + 1トラッカーによるFP/FN予測実験。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.Evaluation.dataset import fileReader
from src.ObjectDetection.models.factory import build_model
from src.config import IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker
from src.util.dataset import build_dataset

from src.error_estimate.config import ErrorPredictionConfig, ShapeMatcherConfig
from src.error_estimate.evaluation import (
    GroundTruthEvaluator,
    dominant_accuracy,
    error_stats,
    match_prediction_to_gt,
)
from src.error_estimate.estimation.rule_based import OneSourceRuleBasedEstimator
from src.error_estimate.io import write_csv
from src.error_estimate.matching import ShapeOneToOneMatcher
from src.error_estimate.visualization import save_debug_image


class OneModelOneTrackerExperiment:
    """1モデル1トラッカー実験のオーケストレーション。"""

    def __init__(
        self,
        model_name="yolov8n",
        dataset_name="KITTI",
        map_name="0001",
        iou_threshold=IOU_THRESHOLD,
        tracker_matcher=None,
        estimator=None,
        error_config=None,
        shape_matcher_config=None,
        output_dir=None,
        save_debug=True,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.map_name = map_name
        self.iou_threshold = iou_threshold
        self.save_debug = save_debug

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
        self.model = build_model(model_name=model_name, dataset=dataset_name, device="cuda")
        self.tracker = SortTracker()

        # Detection <-> Tracker は IoU ではなく、形状 + 距離の 1対1 matching。
        self.tracker_matcher = tracker_matcher or ShapeOneToOneMatcher(
            shape_matcher_config or ShapeMatcherConfig()
        )
        self.estimator = estimator or OneSourceRuleBasedEstimator(
            tracker_matcher=self.tracker_matcher,
            config=error_config or ErrorPredictionConfig(),
        )
        self.gt_evaluator = GroundTruthEvaluator(iou_threshold)

        self.output_dir = output_dir or (
            RESULT_DIR
            / "one_model_one_tracker_error_prediction"
            / dataset_name
            / map_name
            / model_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.debug_dir = self.output_dir / "debug_images"
        if save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.records: list[dict] = []

    def run(self):
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}, map: {self.map_name}")
        print(f"GT evaluation IoU: {self.iou_threshold}")
        print(f"Tracker matcher: {type(self.tracker_matcher).__name__}")
        print(f"Low confidence threshold: {self.estimator.config.low_confidence_threshold}")
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        write_csv(self.output_dir / "frame_records.csv", self.records)
        self._evaluate()

    def _load_gt(self, image_path):
        gt_path = self.dataset.label_dir / f"{image_path.stem}.txt"
        if not gt_path.exists():
            return []
        return fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))

    def _process_frame(self, frame_idx: int, image_path: Path):
        detections = self.model.predict(image_path)
        tracker_predictions = self.tracker.update(detections).predicted_boxes

        gt = self._load_gt(image_path)
        actual = self.gt_evaluator.classify(gt, detections)

        prediction = self.estimator.predict(
            current=detections,
            tracker=tracker_predictions,
        )

        record = {
            "frame_idx": frame_idx,
            f"{self.model_name}_count": len(detections),
            f"{self.model_name}_tracker_count": len(tracker_predictions),
            f"{self.model_name}_fp_candidate_count": len(prediction.fp),
            f"{self.model_name}_fn_candidate_count": len(prediction.fn),
            f"{self.model_name}_fp_final_count": sum(c.final for c in prediction.fp),
            f"{self.model_name}_fn_final_count": sum(c.final for c in prediction.fn),
        }

        for error_type in ("fp", "fn"):
            candidates = getattr(prediction, error_type)
            hard = [c.box for c in candidates if c.final]
            actual_boxes = actual[error_type]
            matches = match_prediction_to_gt(hard, actual_boxes, self.iou_threshold)

            record[f"{self.model_name}_{error_type}_tp"] = len(matches)
            record[f"{self.model_name}_{error_type}_actual"] = len(actual_boxes)

        fp_est = record[f"{self.model_name}_fp_final_count"]
        fn_est = record[f"{self.model_name}_fn_final_count"]
        margin = self.estimator.config.dominant_margin
        if abs(fp_est - fn_est) <= margin:
            predicted_type = "balanced"
        elif fp_est > fn_est:
            predicted_type = "FP"
        else:
            predicted_type = "FN"

        actual_fp = len(actual["fp"])
        actual_fn = len(actual["fn"])
        actual_type = "balanced" if actual_fp == actual_fn else ("FP" if actual_fp > actual_fn else "FN")
        record[f"{self.model_name}_dominant_predicted"] = predicted_type
        record[f"{self.model_name}_dominant_actual"] = actual_type
        record[f"{self.model_name}_dominant_correct"] = int(predicted_type == actual_type)

        self.records.append(record)

        if self.save_debug:
            save_debug_image(
                image_path=image_path,
                output_path=self.debug_dir / f"{image_path.stem}.png",
                current={self.model_name: detections},
                trackers={self.model_name: tracker_predictions},
                predictions={
                    self.model_name: {
                        "fp": [c.box for c in prediction.fp if c.final],
                        "fn": [c.box for c in prediction.fn if c.final],
                    }
                },
                title=(
                    f"frame={frame_idx} DET={len(detections)} "
                    f"TRK={len(tracker_predictions)} "
                    f"FP={fp_est} FN={fn_est}"
                ),
            )

    def _evaluate(self):
        print("\n===== One-model one-tracker error prediction result =====")
        summary = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "map": self.map_name,
            "gt_iou": self.iou_threshold,
            "n_frames": len(self.records),
            "tracker_matcher": type(self.tracker_matcher).__name__,
            "low_confidence_threshold": self.estimator.config.low_confidence_threshold,
        }

        for error_type in ("fp", "fn"):
            stats = error_stats(self.records, self.model_name, error_type)
            print(
                f"{self.model_name} {error_type.upper()}: "
                f"precision={stats['precision']:.4f}, "
                f"recall={stats['recall']:.4f}, "
                f"f1={stats['f1']:.4f}, "
                f"pred={stats['predicted']}, actual={stats['actual']}"
            )
            for k, v in stats.items():
                summary[f"{self.model_name}_{error_type}_{k}"] = (
                    round(v, 4) if isinstance(v, float) else v
                )

        acc = dominant_accuracy(self.records, self.model_name)
        print(
            f"{self.model_name} dominant-error accuracy: "
            f"{acc['accuracy']:.4f} ({acc['correct']}/{acc['total']})"
        )
        for k, v in acc.items():
            summary[f"{self.model_name}_dominant_{k}"] = (
                round(v, 4) if isinstance(v, float) else v
            )

        write_csv(self.output_dir / "summary.csv", [summary])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the one-model one-tracker error prediction experiment."
    )
    parser.add_argument("--model", type=str, default="yolov8n")
    parser.add_argument("--dataset", type=str, default="KITTI")
    parser.add_argument("--map", dest="map_name", type=str, default="0001")
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--low-confidence", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-debug", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    experiment = OneModelOneTrackerExperiment(
        model_name=args.model,
        dataset_name=args.dataset,
        map_name=args.map_name,
        iou_threshold=args.iou_threshold,
        error_config=ErrorPredictionConfig(
            low_confidence_threshold=args.low_confidence,
        ),
        output_dir=args.output_dir,
        save_debug=not args.no_debug,
    )
    experiment.run()


if __name__ == "__main__":
    main()
