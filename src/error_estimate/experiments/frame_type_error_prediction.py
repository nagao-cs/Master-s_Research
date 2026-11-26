"""Frame-level FP/FN prediction using detector/tracker agreement categories.

Two stages. The profile (q_k, lambda) is fitted on several calibration maps and
saved, then loaded to evaluate on other maps. A map used for fitting is refused
at evaluation time so that the test data never leaks into the profile.

Stage 1, fit the profile on several calibration maps:
    python -m src.error_estimate.experiments.frame_type_error_prediction \
      --stage fit --mode two --model1 yolov8n --model2 rtdetr \
      --dataset KITTI --maps 0001 0002 0003

Stage 2, evaluate on test maps with the saved profile:
    python -m src.error_estimate.experiments.frame_type_error_prediction \
      --stage eval --mode two --model1 yolov8n --model2 rtdetr \
      --dataset KITTI --maps 0005 0019

The profile is saved under RESULT_DIR/frame_type_error_prediction/profiles/ unless
--profile is given. One file holds the profile of every target model of the run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from src.Evaluation.dataset import fileReader
from src.ObjectDetection.models.factory import build_model
from src.config import IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker
from src.util.dataset import build_dataset

from src.error_estimate.config import ErrorPredictionConfig, ShapeMatcherConfig
from src.error_estimate.evaluation import GroundTruthEvaluator, match_prediction_to_gt
from src.error_estimate.estimation.frame_type import (
    FN_CATEGORIES,
    FP_CATEGORIES,
    FrameErrorProfile,
    FrameTypeErrorEstimator,
)
from src.error_estimate.estimation.frame_type import FrameEvidence
from src.error_estimate.estimation.frame_metrics import binary_scores, probability_vs_count, safe_div
from src.error_estimate.matching import ShapeOneToOneMatcher


MODEL_TYPES = ("A", "B", "C")
RELATION_BINS = 5


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    """Append one row while allowing later runs to add new columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
        old_fields = list(rows[0].keys()) if rows else []
    fields = list(old_fields)
    for key in row:
        if key not in fields:
            fields.append(key)
    rows.append({k: row.get(k, "") for k in fields})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profile_to_dict(profile: FrameErrorProfile) -> dict:
    return {
        "model_type": profile.model_type,
        "fp_rates": profile.fp_rates,
        "fn_rates": profile.fn_rates,
        "lambda_fn": profile.lambda_fn,
    }


def _profile_from_dict(payload: dict) -> FrameErrorProfile:
    return FrameErrorProfile(
        model_type=payload["model_type"],
        fp_rates={k: float(v) for k, v in payload.get("fp_rates", {}).items()},
        fn_rates={k: float(v) for k, v in payload.get("fn_rates", {}).items()},
        lambda_fn=float(payload.get("lambda_fn", 0.0)),
    )


def save_profiles(path: Path, profiles: dict[str, FrameErrorProfile], dataset: str, maps: list[str]) -> None:
    """Save the profile of every target model, with the maps they were fitted on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "fit_maps": list(maps),
        "profiles": {name: _profile_to_dict(p) for name, p in profiles.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def load_profiles(path: Path) -> tuple[dict[str, FrameErrorProfile], str, list[str]]:
    """Return (profile per model name, dataset, maps used for fitting)."""
    payload = json.loads(path.read_text())
    profiles = {name: _profile_from_dict(p) for name, p in payload["profiles"].items()}
    return profiles, payload["dataset"], list(payload["fit_maps"])


def _match_count(pred_boxes, actual_boxes, iou_threshold: float) -> int:
    return len(match_prediction_to_gt(pred_boxes, actual_boxes, iou_threshold))


def _unique_with_shape(boxes: Iterable, matcher: ShapeOneToOneMatcher):
    result = []
    for box in boxes:
        if not matcher.match([box], result).matches:
            result.append(box)
    return result


def fit_profile(
    frames: list[dict],
    model_type: str,
    iou_threshold: float,
) -> FrameErrorProfile:
    """Count, per category, how often the candidates were real errors.

    q_k^FP = (category-k boxes that match an actual FP) / (category-k boxes), k in FP_CATEGORIES
    q_k^FN = (category-k boxes that match an actual FN) / (category-k boxes), k in FN_CATEGORIES
    lambda = frames holding a GT object captured by no source / all frames
    """
    fp_num: dict[str, int] = {}
    fp_den: dict[str, int] = {}
    fn_num: dict[str, int] = {}
    fn_den: dict[str, int] = {}
    leak_frames = 0

    for item in frames:
        ev: FrameEvidence = item["evidence"]
        actual = item["actual"]
        for k in FP_CATEGORIES:
            boxes = ev.class_boxes.get(k, [])
            fp_den[f"c{k}"] = fp_den.get(f"c{k}", 0) + len(boxes)
            fp_num[f"c{k}"] = fp_num.get(f"c{k}", 0) + _match_count(boxes, actual["fp"], iou_threshold)
        for k in FN_CATEGORIES:
            boxes = ev.class_boxes.get(k, [])
            fn_den[f"c{k}"] = fn_den.get(f"c{k}", 0) + len(boxes)
            fn_num[f"c{k}"] = fn_num.get(f"c{k}", 0) + _match_count(boxes, actual["fn"], iou_threshold)

        # Hidden FN: a GT object that no detection or tracker box captured.
        if item["hidden_fn"] > 0:
            leak_frames += 1

    fp_rates = {k: min(1.0, safe_div(fp_num[k], v)) for k, v in fp_den.items()}
    fn_rates = {k: min(1.0, safe_div(fn_num[k], v)) for k, v in fn_den.items()}
    return FrameErrorProfile(
        model_type=model_type,
        fp_rates=fp_rates,
        fn_rates=fn_rates,
        lambda_fn=safe_div(leak_frames, len(frames)),
    )


class FrameTypeExperiment:
    def __init__(
        self,
        mode: str,
        model1: str,
        model1_type: str,
        dataset_name: str,
        model2: str | None = None,
        model2_type: str | None = None,
        iou_threshold: float = IOU_THRESHOLD,
        low_confidence_threshold: float = 0.5,
        profile_path: Path | None = None,
        decision_threshold: float = 0.5,
    ):
        if model1_type not in MODEL_TYPES or (model2_type and model2_type not in MODEL_TYPES):
            raise ValueError("model type must be A, B, or C")
        if mode not in ("one", "two"):
            raise ValueError("mode must be one or two")
        if mode == "two" and not model2:
            raise ValueError("--model2 is required in two-model mode")

        self.mode = mode
        self.model1_name = model1
        self.model1_type = model1_type
        self.model2_name = model2
        self.model2_type = model2_type
        self.dataset_name = dataset_name
        self.iou_threshold = iou_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.decision_threshold = decision_threshold

        # Models are built once and reused for every map.
        self.model1 = build_model(model_name=model1, dataset=dataset_name, device="cuda")
        self.model2 = build_model(model_name=model2, dataset=dataset_name, device="cuda") if mode == "two" else None

        self.matcher = ShapeOneToOneMatcher(ShapeMatcherConfig())
        self.estimator = FrameTypeErrorEstimator(
            tracker_matcher=self.matcher,
            config=ErrorPredictionConfig(low_confidence_threshold=low_confidence_threshold),
        )
        self.gt_evaluator = GroundTruthEvaluator(iou_threshold)

        self.result_root = RESULT_DIR / "frame_type_error_prediction" / ("one_model" if mode == "one" else "two_model")
        self.combination = model1 if mode == "one" else f"{model1}_{model2}"
        self.profile_path = profile_path or (
            RESULT_DIR / "frame_type_error_prediction" / "profiles"
            / ("one_model" if mode == "one" else "two_model") / f"{self.combination}.json"
        )

    # ---- per-map data collection -------------------------------------------------

    def _load_gt(self, dataset, image_path):
        gt_path = dataset.label_dir / f"{image_path.stem}.txt"
        if not gt_path.exists():
            return []
        return fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))

    def _hidden_fn_count(self, gt, sources: list) -> int:
        """Number of GT objects captured by none of the given detection/tracker boxes."""
        return len(self.gt_evaluator.classify(gt, sources)["fn"])

    def _collect(self, map_name: str) -> list[dict]:
        """Run the detectors and trackers over one map and keep per-frame evidence."""
        dataset = build_dataset(dataset_name=self.dataset_name, map_name=map_name)
        # Trackers carry state, so every map starts with fresh ones.
        tracker1 = SortTracker()
        tracker2 = SortTracker() if self.mode == "two" else None

        print(f"Mode: {self.mode}")
        print(f"M1: {self.model1_name} ({self.model1_type})")
        if self.mode == "two":
            print(f"M2: {self.model2_name} ({self.model2_type})")
        print(f"Dataset: {self.dataset_name}, map: {map_name}")
        print(f"Low confidence threshold: {self.low_confidence_threshold}")
        print(f"Total frames: {len(dataset)}")

        frames: list[dict] = []
        for frame_idx, (image_path, _) in enumerate(tqdm(dataset)):
            d1 = self.model1.predict(image_path)
            t1 = tracker1.update(d1).predicted_boxes
            gt = self._load_gt(dataset, image_path)
            actual1 = self.gt_evaluator.classify(gt, d1)
            if self.mode == "one":
                evidence = self.estimator.one_model_evidence(d1, t1)
                frames.append({
                    "frame_idx": frame_idx, "evidence": evidence, "actual": actual1,
                    "hidden_fn": self._hidden_fn_count(gt, list(d1) + list(t1)),
                })
            else:
                d2 = self.model2.predict(image_path)
                t2 = tracker2.update(d2).predicted_boxes
                actual2 = self.gt_evaluator.classify(gt, d2)
                ev1 = self.estimator.two_model_evidence(d1, t1, d2)
                ev2 = self.estimator.two_model_evidence(d2, t2, d1)
                frames.append({
                    "frame_idx": frame_idx,
                    "evidence1": ev1, "actual1": actual1,
                    "evidence2": ev2, "actual2": actual2,
                    # GT objects that neither model nor either tracker captured.
                    # One value per frame, so both models get the same lambda.
                    "hidden_fn": self._hidden_fn_count(gt, list(d1) + list(d2) + list(t1) + list(t2)),
                })
        return frames

    def _fit_items(self, frames: list[dict]) -> dict[str, list[dict]]:
        """Split collected frames into the fit_profile input of each target model."""
        if self.mode == "one":
            return {self.model1_name: [
                {"evidence": r["evidence"], "actual": r["actual"], "hidden_fn": r["hidden_fn"]} for r in frames
            ]}
        return {
            self.model1_name: [
                {"evidence": r["evidence1"], "actual": r["actual1"], "hidden_fn": r["hidden_fn"]} for r in frames
            ],
            self.model2_name: [
                {"evidence": r["evidence2"], "actual": r["actual2"], "hidden_fn": r["hidden_fn"]} for r in frames
            ],
        }

    def _model_types(self) -> dict[str, str]:
        types = {self.model1_name: self.model1_type}
        if self.mode == "two":
            types[self.model2_name] = self.model2_type
        return types

    # ---- stage 1 ----------------------------------------------------------------

    def fit(self, maps: list[str]):
        """Fit one profile per target model on all frames of every calibration map.

        Counts are pooled over the maps, so q_k and lambda are frequencies over
        all calibration frames, not averages of per-map values.
        """
        pooled: dict[str, list[dict]] = {}
        for map_name in maps:
            for name, items in self._fit_items(self._collect(map_name)).items():
                pooled.setdefault(name, []).extend(items)

        types = self._model_types()
        profiles = {
            name: fit_profile(items, types[name], self.iou_threshold)
            for name, items in pooled.items()
        }
        for name, profile in profiles.items():
            print(f"{name} profile ({len(pooled[name])} frames): {profile}")
        save_profiles(self.profile_path, profiles, self.dataset_name, maps)
        print(f"Profile saved to {self.profile_path}")

    # ---- stage 2 ----------------------------------------------------------------

    def _evaluate_model(self, model_name: str, model_type: str, evidence_key: str, actual_key: str, records: list[dict], profile: FrameErrorProfile, map_name: str):
        rows = []
        for r in records:
            pred = self.estimator.predict(r[evidence_key], profile)
            actual = r[actual_key]
            actual_fp = len(actual["fp"])
            actual_fn = len(actual["fn"])
            rows.append({
                "frame_idx": r["frame_idx"],
                "model": model_name,
                "model_type": model_type,
                "pred_fp_probability": pred["fp_probability"],
                "pred_fn_probability": pred["fn_probability"],
                "pred_expected_fp": pred["expected_fp"],
                "pred_expected_fn": pred["expected_fn"],
                "actual_fp": actual_fp,
                "actual_fn": actual_fn,
                "actual_error_count": actual_fp + actual_fn,
                "pred_expected_error_count": pred["expected_fp"] + pred["expected_fn"],
                **{f"n_cat{k}": r[evidence_key].count(k) for k in (*FP_CATEGORIES, *FN_CATEGORIES)},
                "pred_dominant": "balanced" if abs(pred["expected_fp"] - pred["expected_fn"]) < 1e-9 else ("FP" if pred["expected_fp"] > pred["expected_fn"] else "FN"),
                "actual_dominant": "balanced" if actual_fp == actual_fn else ("FP" if actual_fp > actual_fn else "FN"),
            })
        return rows

    def _summary(self, rows: list[dict], model_name: str, model_type: str, profile: FrameErrorProfile, map_name: str):
        th = self.decision_threshold
        summary = {
            "model": model_name,
            "model_type": model_type,
            "dataset": self.dataset_name,
            "map": map_name,
            "n_frames": len(rows),
            "calibration_lambda_fn": round(profile.lambda_fn, 6),
            "decision_threshold": th,
            "fp_probability_mean": round(float(np.mean([r["pred_fp_probability"] for r in rows])), 6),
            "fn_probability_mean": round(float(np.mean([r["pred_fn_probability"] for r in rows])), 6),
        }
        # Frame-level decision: "this frame has at least one FP (FN)".
        for kind in ("fp", "fn"):
            scores = binary_scores(
                [r[f"pred_{kind}_probability"] >= th for r in rows],
                [r[f"actual_{kind}"] > 0 for r in rows],
            )
            for key, value in scores.items():
                summary[f"{kind}_{key}"] = round(value, 6) if isinstance(value, float) else value

        dominant_correct = sum(r["pred_dominant"] == r["actual_dominant"] for r in rows)
        summary["dominant_accuracy"] = round(safe_div(dominant_correct, len(rows)), 6)
        summary["dominant_correct"] = dominant_correct
        return summary

    def _load_profiles_for_eval(self, maps: list[str]) -> dict[str, FrameErrorProfile]:
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {self.profile_path}. Run --stage fit first.")
        profiles, dataset, fit_maps = load_profiles(self.profile_path)
        if dataset != self.dataset_name:
            raise ValueError(f"Profile was fitted on {dataset}, not {self.dataset_name}")
        leaked = sorted(set(maps) & set(fit_maps))
        if leaked:
            raise ValueError(f"Maps {leaked} were used to fit the profile; choose other test maps")
        missing = [n for n in self._model_types() if n not in profiles]
        if missing:
            raise ValueError(f"Profile has no entry for {missing}")
        print(f"Loaded profile {self.profile_path} (fitted on maps {fit_maps})")
        return profiles

    def evaluate(self, maps: list[str]):
        """Evaluate every test map with the saved profile (all frames are test data)."""
        profiles = self._load_profiles_for_eval(maps)
        for map_name in maps:
            self._evaluate_map(map_name, profiles)

    def _evaluate_map(self, map_name: str, profiles: dict[str, FrameErrorProfile]):
        frames = self._collect(map_name)
        output_dir = self.result_root / self.dataset_name / map_name / self.combination
        if self.mode == "one":
            targets = [(self.model1_name, self.model1_type, "evidence", "actual")]
        else:
            targets = [
                (self.model1_name, self.model1_type, "evidence1", "actual1"),
                (self.model2_name, self.model2_type, "evidence2", "actual2"),
            ]

        all_rows, summaries, relation_rows = [], [], []
        for name, mtype, evidence_key, actual_key in targets:
            rows = self._evaluate_model(name, mtype, evidence_key, actual_key, frames, profiles[name], map_name)
            all_rows.extend(rows)
            summaries.append(self._summary(rows, name, mtype, profiles[name], map_name))
            relation_rows.extend(self._relation_rows(rows, name))
        write_csv(output_dir / "frame_records.csv", all_rows)
        print(f"Frame records saved to {output_dir / 'frame_records.csv'}")
        self._save_relation(relation_rows, output_dir, map_name)

        # One summary.csv per model-count, with every subsequent model appended.
        summary_path = self.result_root / "summary.csv"
        for row in summaries:
            append_csv(summary_path, row)
            print(row)
        print(f"Summary saved to {summary_path}")

    def _relation_rows(self, rows, model_name):
        """Predicted probability vs. actual error count, one row per (kind, probability bin)."""
        out = []
        for kind in ("fp", "fn"):
            table = probability_vs_count(
                [r[f"pred_{kind}_probability"] for r in rows],
                [r[f"actual_{kind}"] for r in rows],
                RELATION_BINS,
            )
            out.extend({"model": model_name, "kind": kind, **t} for t in table)
        return out

    def _save_relation(self, relation_rows, output_dir: Path, map_name: str):
        """Per-run file, plus one shared file that accumulates every run."""
        write_csv(output_dir / "relation.csv", relation_rows)
        shared = self.result_root / "relation.csv"
        run_id = {
            "combination": self.model1_name if self.mode == "one" else f"{self.model1_name}+{self.model2_name}",
            "dataset": self.dataset_name,
            "map": map_name,
        }
        for row in relation_rows:
            append_csv(shared, {**run_id, **row})
        print(f"Probability-bin results saved to {output_dir / 'relation.csv'} and {shared}")


def build_parser():
    p = argparse.ArgumentParser(description="Frame-level FP/FN prediction from detector/tracker evidence")
    p.add_argument("--stage", choices=["fit", "eval"], required=True,
                   help="fit: build and save the profile from --maps; eval: test on --maps with the saved profile")
    p.add_argument("--mode", choices=["one", "two"], default="one")
    p.add_argument("--model", dest="model1", default="yolov8n")
    p.add_argument("--model1", dest="model1_alias", default=None, help=argparse.SUPPRESS)
    p.add_argument("--model2", default=None)
    p.add_argument("--dataset", default="KITTI")
    p.add_argument("--maps", nargs="+", required=True,
                   help="calibration maps for --stage fit, test maps for --stage eval")
    p.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    p.add_argument("--low-confidence-threshold", type=float, default=0.5)
    p.add_argument("--profile", type=Path, default=None,
                   help="profile JSON path (default: under RESULT_DIR/frame_type_error_prediction/profiles)")
    p.add_argument("--decision-threshold", type=float, default=0.5,
                   help="probability at or above which a frame is predicted to contain an error")
    return p


def main():
    args = build_parser().parse_args()
    model1 = args.model1_alias or args.model1
    exp = FrameTypeExperiment(
        mode=args.mode,
        model1=model1,
        model1_type="A",
        model2=args.model2,
        model2_type="A",
        dataset_name=args.dataset,
        iou_threshold=args.iou_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        profile_path=args.profile,
        decision_threshold=args.decision_threshold,
    )
    if args.stage == "fit":
        exp.fit(args.maps)
    else:
        exp.evaluate(args.maps)


if __name__ == "__main__":
    main()