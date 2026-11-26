from __future__ import annotations

import csv
from pathlib import Path

from tqdm import tqdm

from src.Evaluation.dataset import fileReader
from src.ObjectDetection.models.factory import build_model
from src.config import IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker
from src.util.dataset import build_dataset

from error_estimate.config import ErrorPredictionConfig, ShapeMatcherConfig
from error_estimate.evaluation import GroundTruthEvaluator, match_prediction_to_gt, error_stats, dominant_accuracy
from error_estimate.estimation import TwoSourceRuleBasedEstimator
from error_estimate.io import write_csv
from error_estimate.matching import IoUOneToOneMatcher, ShapeOneToOneMatcher
from error_estimate.visualization import save_debug_image


class TwoModelTwoTrackerExperiment:
    """
    Experiment orchestration only.

    Prediction policy and matching algorithms are injected, so this class does not
    need to change when a different matcher/estimator is tested.
    """

    def __init__(
        self,
        model1_name="yolov8n",
        model2_name="ssd",
        dataset_name="KITTI",
        map_name="0001",
        iou_threshold=IOU_THRESHOLD,
        model_matcher=None,
        tracker_matcher=None,
        evidence_matcher=None,
        estimator=None,
        error_config=None,
        output_dir=None,
        save_debug=True,
    ):
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.dataset_name = dataset_name
        self.map_name = map_name
        self.iou_threshold = iou_threshold
        self.save_debug = save_debug

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
        self.model1 = build_model(model_name=model1_name, dataset=dataset_name, device="cuda")
        self.model2 = build_model(model_name=model2_name, dataset=dataset_name, device="cuda")
        self.tracker1 = SortTracker()
        self.tracker2 = SortTracker()

        self.model_matcher = model_matcher or IoUOneToOneMatcher(iou_threshold)
        self.tracker_matcher = tracker_matcher or ShapeOneToOneMatcher(ShapeMatcherConfig())
        self.evidence_matcher = evidence_matcher or self.tracker_matcher
        self.estimator = estimator or TwoSourceRuleBasedEstimator(
            model_matcher=self.model_matcher,
            tracker_matcher=self.tracker_matcher,
            evidence_matcher=self.evidence_matcher,
            config=error_config or ErrorPredictionConfig(),
        )
        self.gt_evaluator = GroundTruthEvaluator(iou_threshold)

        self.output_dir = output_dir or (
            RESULT_DIR / "m1_m2_two_tracker_error_prediction_refactored"
            / dataset_name / map_name / f"{model1_name}_{model2_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.output_dir / "debug_images"
        if save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.records: list[dict] = []
        self.candidate_records: list[dict] = []

    def run(self):
        print(f"M1: {self.model1_name}")
        print(f"M2: {self.model2_name}")
        print(f"Dataset: {self.dataset_name}, map: {self.map_name}")
        print(f"GT evaluation IoU: {self.iou_threshold}")
        print(f"Model matcher: {type(self.model_matcher).__name__}")
        print(f"Tracker matcher: {type(self.tracker_matcher).__name__}")
        print(f"Evidence matcher: {type(self.evidence_matcher).__name__}")
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        write_csv(self.output_dir / "frame_records.csv", self.records)
        write_csv(self.output_dir / "candidates.csv", self.candidate_records)
        self._evaluate()

    def _load_gt(self, image_path):
        gt_path = self.dataset.label_dir / f"{image_path.stem}.txt"
        if not gt_path.exists():
            return []
        return fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))

    def _process_frame(self, frame_idx, image_path: Path):
        m1 = self.model1.predict(image_path)
        m2 = self.model2.predict(image_path)
        current = {self.model1_name: m1, self.model2_name: m2}

        t1 = self.tracker1.update(m1).predicted_boxes
        t2 = self.tracker2.update(m2).predicted_boxes
        trackers = {self.model1_name: t1, self.model2_name: t2}

        gt = self._load_gt(image_path)
        actual = {
            self.model1_name: self.gt_evaluator.classify(gt, m1),
            self.model2_name: self.gt_evaluator.classify(gt, m2),
        }

        predictions = {}
        for model_name in (self.model1_name, self.model2_name):
            other_name = self.model2_name if model_name == self.model1_name else self.model1_name
            result = self.estimator.predict(
                current=current[model_name],
                other=current[other_name],
                tracker=trackers[model_name],
            )
            predictions[model_name] = result

        record = {
            "frame_idx": frame_idx,
            "m1_count": len(m1),
            "m2_count": len(m2),
            "m1_tracker_count": len(t1),
            "m2_tracker_count": len(t2),
        }

        for model_name in (self.model1_name, self.model2_name):
            prediction = predictions[model_name]
            for error_type in ("fp", "fn"):
                candidates = getattr(prediction, error_type)
                hard = [c.box for c in candidates if c.final]
                actual_boxes = actual[model_name][error_type]
                matches = match_prediction_to_gt(hard, actual_boxes, self.iou_threshold)

                record[f"{model_name}_{error_type}_candidate_count"] = len(candidates)
                record[f"{model_name}_{error_type}_final_count"] = len(hard)
                record[f"{model_name}_{error_type}_tp"] = len(matches)
                record[f"{model_name}_{error_type}_actual"] = len(actual_boxes)

                if error_type == "fn":
                    record[f"{model_name}_fn_other_tracker_count"] = sum(c.sources == {"other", "tracker"} for c in candidates)
                    record[f"{model_name}_fn_other_only_count"] = sum(c.sources == {"other"} for c in candidates)
                    record[f"{model_name}_fn_tracker_only_count"] = sum(c.sources == {"tracker"} for c in candidates)

                for candidate_idx, candidate in enumerate(candidates):
                    row = {
                        "frame_idx": frame_idx,
                        "image": image_path.name,
                        "model": model_name,
                        "error_type": error_type,
                        "candidate_idx": candidate_idx,
                        "final_candidate": int(candidate.final),
                        "sources": "+".join(sorted(candidate.sources)) if candidate.sources else None,
                    }
                    row.update(candidate.metadata)
                    self.candidate_records.append(row)

            fp_est = record[f"{model_name}_fp_final_count"]
            fn_est = record[f"{model_name}_fn_final_count"]
            margin = self.estimator.config.dominant_margin
            if abs(fp_est - fn_est) <= margin:
                predicted_type = "balanced"
            elif fp_est > fn_est:
                predicted_type = "FP"
            else:
                predicted_type = "FN"

            actual_fp = len(actual[model_name]["fp"])
            actual_fn = len(actual[model_name]["fn"])
            actual_type = "balanced" if actual_fp == actual_fn else ("FP" if actual_fp > actual_fn else "FN")
            record[f"{model_name}_dominant_predicted"] = predicted_type
            record[f"{model_name}_dominant_actual"] = actual_type
            record[f"{model_name}_dominant_correct"] = int(predicted_type == actual_type)

        self.records.append(record)

        if self.save_debug:
            debug_path = self.debug_dir / f"{image_path.stem}.jpg"
            current_short = {"m1": m1, "m2": m2}
            trackers_short = {"m1": t1, "m2": t2}
            predictions_short = {
                "m1": {"fp": [c.box for c in predictions[self.model1_name].fp if c.final],
                        "fn": [c.box for c in predictions[self.model1_name].fn if c.final]},
                "m2": {"fp": [c.box for c in predictions[self.model2_name].fp if c.final],
                        "fn": [c.box for c in predictions[self.model2_name].fn if c.final]},
            }
            title = (
                f"frame={frame_idx} M1={len(m1)} M2={len(m2)} "
                f"T1={len(t1)} T2={len(t2)}"
            )
            save_debug_image(debug_path, current_short, trackers_short, predictions_short, title)

    def _evaluate(self):
        print("\n===== Refactored error prediction result =====")
        summary = {
            "model1": self.model1_name,
            "model2": self.model2_name,
            "dataset": self.dataset_name,
            "map": self.map_name,
            "gt_iou": self.iou_threshold,
            "n_frames": len(self.records),
            "model_matcher": type(self.model_matcher).__name__,
            "tracker_matcher": type(self.tracker_matcher).__name__,
            "evidence_matcher": type(self.evidence_matcher).__name__,
        }

        for model in (self.model1_name, self.model2_name):
            for error_type in ("fp", "fn"):
                stats = error_stats(self.records, model, error_type)
                print(
                    f"{model} {error_type.upper()}: "
                    f"precision={stats['precision']:.4f}, "
                    f"recall={stats['recall']:.4f}, "
                    f"f1={stats['f1']:.4f}, "
                    f"pred={stats['predicted']}, actual={stats['actual']}"
                )
                for k, v in stats.items():
                    summary[f"{model}_{error_type}_{k}"] = round(v, 4) if isinstance(v, float) else v

            acc = dominant_accuracy(self.records, model)
            print(f"{model} dominant-error accuracy: {acc['accuracy']:.4f} ({acc['correct']}/{acc['total']})")
            for k, v in acc.items():
                summary[f"{model}_dominant_{k}"] = round(v, 4) if isinstance(v, float) else v

        write_csv(self.output_dir / "summary.csv", [summary])
