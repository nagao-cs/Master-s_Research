"""
M1/M2 + 2 Trackers による FP/FN error-type prediction experiment.

v3: IoU に依存しない Shape-aware 1-to-1 matching
------------------------------------------------
SORT は位置・速度から bbox を予測するため、実際の検出 bbox と予測 bbox の
IoU が 0 になることがある。この場合でも、同一物体らしい候補を失わないように、

    - bbox 中心間距離
    - bbox の幅・高さの比
    - bbox の aspect ratio の比

を使って tracking bbox と detection bbox の近さを評価する。

重要:
    最終的な detection <-> tracker の対応は必ず 1 対 1。
    Hungarian assignment により、同じ detection / tracker を複数候補に使わない。

FP/FN prediction 自体は score を使わず、最終候補数だけを使う。

FP:
    current detection が other model と own tracker の両方に
    shape-aware matching しなければ FP candidate。
    また、どちらか一方にしか match せず confidence <= 0.5 の場合も
    FP candidate とする。

FN:
    current detection に match しない other-model detection または
    own tracker prediction を FN evidence とする。
    other と tracker の evidence が同一物体らしい場合は 1 候補に統合する。

GT は評価専用であり、prediction には使わない。
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
from tqdm import tqdm

from src.boundingBox.boundingBox import ClassifyCategory
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.dataset import fileReader
from src.ObjectDetection.models.factory import build_model
from src.util.dataset import build_dataset
from src.config import IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker


def _safe_div(a, b):
    return a / b if b else 0.0


def _get_confidence(box):
    """Detection confidence の属性名の実装差を吸収する。"""
    for name in ("confidence", "conf", "score", "confidenceScore"):
        value = getattr(box, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    # Tracker prediction 等 confidence を持たない box は低 confidence 扱いしない。
    return 1.0


def _center(box):
    return float(box.xCenter), float(box.yCenter)


def _shape_features(box):
    """
    bbox shape に関する特徴量。

    戻り値:
        width, height, aspect_ratio
    """
    w = max(float(box.width), 1e-6)
    h = max(float(box.height), 1e-6)
    return w, h, w / h


def _shape_distance(det, track, center_weight=1.0, size_weight=0.75, ratio_weight=0.75):
    """
    Detection と tracking bbox の shape-aware distance。

    IoU は一切使わない。

    1. center distance:
       tracking bbox の幅・高さで正規化する。
       したがって bbox が大きい物体ほど、多少中心がずれていても候補にできる。

    2. size distance:
       width / height の log-ratio。
       例: width が 2 倍なら log(2)、0.5 倍なら -log(2) の絶対値。

    3. aspect-ratio distance:
       縦横比の log-ratio。

    小さいほど同一物体らしい。
    """
    dx = float(det.xCenter) - float(track.xCenter)
    dy = float(det.yCenter) - float(track.yCenter)

    tw, th, tar = _shape_features(track)

    center_distance = math.sqrt(
        (dx / tw) ** 2 +
        (dy / th) ** 2
    )

    dw, dh, dar = _shape_features(det)

    size_distance = 0.5 * (
        abs(math.log(dw / tw)) +
        abs(math.log(dh / th))
    )

    ratio_distance = abs(math.log(dar / tar))

    total = (
        center_weight * center_distance
        + size_weight * size_distance
        + ratio_weight * ratio_distance
    )

    return {
        "distance": float(total),
        "center_distance": float(center_distance),
        "size_distance": float(size_distance),
        "ratio_distance": float(ratio_distance),
        "width_ratio": float(dw / tw),
        "height_ratio": float(dh / th),
        "aspect_ratio": float(dar / tar),
    }


def _shape_match_one_to_one(
    detections,
    references,
    max_distance=2.5,
    max_center_distance=2.0,
    max_size_ratio=2.5,
    max_aspect_ratio=2.0,
    center_weight=1.0,
    size_weight=0.75,
    ratio_weight=0.75,
):
    """
    shape-aware な 1 対 1 matching。

    IoU >= threshold は要求しない。
    ただし、無制限に遠い bbox を対応付けることは避けるため、
    center / size / aspect ratio の gate を設ける。

    Hungarian assignment により global に最小 distance の対応を選ぶ。
    同じ detection / reference は最大 1 回しか match されない。

    Returns:
        matches:
            (det_index, ref_index, info)
        unmatched_detection_indices
        unmatched_reference_indices
    """
    if not detections or not references:
        return [], list(range(len(detections))), list(range(len(references)))

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "numpy と scipy が必要です。`pip install numpy scipy` を実行してください。"
        ) from e

    INVALID = 1e9
    cost = np.full((len(detections), len(references)), INVALID, dtype=float)
    infos = {}

    for i, det in enumerate(detections):
        for j, ref in enumerate(references):
            if det.classId != ref.classId:
                continue

            info = _shape_distance(
                det,
                ref,
                center_weight=center_weight,
                size_weight=size_weight,
                ratio_weight=ratio_weight,
            )

            width_ratio = info["width_ratio"]
            height_ratio = info["height_ratio"]
            aspect_ratio = info["aspect_ratio"]

            # どちら方向の比でも同じ許容幅にする。
            width_ok = (1.0 / max_size_ratio) <= width_ratio <= max_size_ratio
            height_ok = (1.0 / max_size_ratio) <= height_ratio <= max_size_ratio
            aspect_ok = (1.0 / max_aspect_ratio) <= aspect_ratio <= max_aspect_ratio

            if (
                info["distance"] <= max_distance
                and info["center_distance"] <= max_center_distance
                and width_ok
                and height_ok
                and aspect_ok
            ):
                cost[i, j] = info["distance"]
                infos[(i, j)] = info

    rows, cols = linear_sum_assignment(cost)

    matches = []
    used_det = set()
    used_ref = set()

    for i, j in zip(rows, cols):
        if cost[i, j] >= INVALID:
            continue
        matches.append((int(i), int(j), infos[(int(i), int(j))]))
        used_det.add(int(i))
        used_ref.add(int(j))

    unmatched_det = [
        i for i in range(len(detections))
        if i not in used_det
    ]
    unmatched_ref = [
        j for j in range(len(references))
        if j not in used_ref
    ]

    return matches, unmatched_det, unmatched_ref



def _iou_match_one_to_one(a_boxes, b_boxes, iou_threshold):
    """同一クラス・IoU閾値による通常の1対1 matching（モデル間用）。"""
    if not a_boxes or not b_boxes:
        return [], list(range(len(a_boxes))), list(range(len(b_boxes)))

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    INVALID = 1e9
    cost = np.full((len(a_boxes), len(b_boxes)), INVALID, dtype=float)
    ious = np.zeros_like(cost)

    for i, a in enumerate(a_boxes):
        for j, b in enumerate(b_boxes):
            if a.classId != b.classId:
                continue
            iou = float(a.computeIoU(b))
            ious[i, j] = iou
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou

    rows, cols = linear_sum_assignment(cost)
    matches = []
    used_a, used_b = set(), set()
    for i, j in zip(rows, cols):
        if cost[i, j] >= INVALID:
            continue
        matches.append((int(i), int(j), {
            "distance": 1.0 - float(ious[i, j]),
            "iou": float(ious[i, j]),
        }))
        used_a.add(int(i))
        used_b.add(int(j))

    ua = [i for i in range(len(a_boxes)) if i not in used_a]
    ub = [j for j in range(len(b_boxes)) if j not in used_b]
    return matches, ua, ub

def _merge_evidence_one_to_one(
    boxes_a,
    boxes_b,
    max_distance=2.5,
    max_center_distance=2.0,
    max_size_ratio=2.5,
    max_aspect_ratio=2.0,
    center_weight=1.0,
    size_weight=0.75,
    ratio_weight=0.75,
):
    """
    other-model evidence と tracker evidence を 1 対 1 で統合する。

    両方が同じ物体を示している場合は 1 candidate にまとめる。
    match しないものはそれぞれ独立 candidate として残す。

    Returns:
        candidates:
            {
                "box": 代表 bbox,
                "sources": {"other", "tracker"} の集合
                ...
            }
    """
    matches, unmatched_a, unmatched_b = _shape_match_one_to_one(
        boxes_a,
        boxes_b,
        max_distance=max_distance,
        max_center_distance=max_center_distance,
        max_size_ratio=max_size_ratio,
        max_aspect_ratio=max_aspect_ratio,
        center_weight=center_weight,
        size_weight=size_weight,
        ratio_weight=ratio_weight,
    )

    candidates = []

    for ai, bi, info in matches:
        # tracker bbox を代表にする。予測位置を current FN candidate の位置として使う。
        candidates.append({
            "box": boxes_b[bi],
            "sources": {"other", "tracker"},
            "other_index": ai,
            "tracker_index": bi,
            "shape_distance": info["distance"],
            "center_distance": info["center_distance"],
            "size_distance": info["size_distance"],
            "ratio_distance": info["ratio_distance"],
        })

    for ai in unmatched_a:
        candidates.append({
            "box": boxes_a[ai],
            "sources": {"other"},
            "other_index": ai,
            "tracker_index": None,
            "shape_distance": None,
            "center_distance": None,
            "size_distance": None,
            "ratio_distance": None,
        })

    for bi in unmatched_b:
        candidates.append({
            "box": boxes_b[bi],
            "sources": {"tracker"},
            "other_index": None,
            "tracker_index": bi,
            "shape_distance": None,
            "center_distance": None,
            "size_distance": None,
            "ratio_distance": None,
        })

    return candidates


class M1M2TwoTrackerErrorPredictionShapeMatchingExperiment:
    def __init__(
        self,
        model1_name="yolov8n",
        model2_name="ssd",
        dataset_name="KITTI",
        map_name="0001",
        iou_threshold=IOU_THRESHOLD,
        max_match_distance=2.5,
        max_center_distance=2.0,
        max_size_ratio=2.5,
        max_aspect_ratio=2.0,
        center_weight=1.0,
        size_weight=0.75,
        ratio_weight=0.75,
        low_confidence_threshold=0.5,
        dominant_margin=0.0,
        save_debug=True,
    ):
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.dataset_name = dataset_name
        self.map_name = map_name

        # GT 評価用 IoU。prediction の detection-tracker matching には使わない。
        self.iou_threshold = iou_threshold

        self.max_match_distance = max_match_distance
        self.max_center_distance = max_center_distance
        self.max_size_ratio = max_size_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.center_weight = center_weight
        self.size_weight = size_weight
        self.ratio_weight = ratio_weight

        self.low_confidence_threshold = low_confidence_threshold
        self.dominant_margin = dominant_margin
        self.save_debug = save_debug

        self.dataset = build_dataset(
            dataset_name=dataset_name,
            map_name=map_name,
        )
        self.model1 = build_model(
            model_name=model1_name,
            dataset=dataset_name,
            device="cuda",
        )
        self.model2 = build_model(
            model_name=model2_name,
            dataset=dataset_name,
            device="cuda",
        )

        self.tracker1 = SortTracker()
        self.tracker2 = SortTracker()
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)

        self.output_dir = (
            RESULT_DIR
            / f"m1_m2_two_tracker_error_prediction_shape_matching/"
              f"{dataset_name}/{map_name}/{model1_name}_{model2_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.debug_dir = self.output_dir / "debug_images"
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.records = []
        self.candidate_records = []

    @property
    def models(self):
        return {
            self.model1_name: self.model1,
            self.model2_name: self.model2,
        }

    def _match(self, detections, references):
        return _shape_match_one_to_one(
            detections,
            references,
            max_distance=self.max_match_distance,
            max_center_distance=self.max_center_distance,
            max_size_ratio=self.max_size_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            center_weight=self.center_weight,
            size_weight=self.size_weight,
            ratio_weight=self.ratio_weight,
        )

    def run(self):
        print(f"M1: {self.model1_name}")
        print(f"M2: {self.model2_name}")
        print(f"Dataset: {self.dataset_name}, map: {self.map_name}")
        print(f"GT evaluation IoU: {self.iou_threshold}")
        print("Detection/Tracker matching: Shape-aware 1-to-1 (NO IoU)")
        print(f"Max match distance: {self.max_match_distance}")
        print(f"Max center distance: {self.max_center_distance}")
        print(f"Max size ratio: {self.max_size_ratio}")
        print(f"Max aspect ratio: {self.max_aspect_ratio}")
        print(f"Distance weights: center={self.center_weight}, "
              f"size={self.size_weight}, ratio={self.ratio_weight}")
        print(f"Low confidence threshold: {self.low_confidence_threshold}")
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        self._save_records()
        self._evaluate()

    def _load_gt(self, image_path):
        gt_path = self.dataset.label_dir / f"{image_path.stem}.txt"
        if not gt_path.exists():
            return []
        return fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))

    def _fp_candidates(
        self,
        model_name,
        detections,
        current_detections,
        tracker_predictions,
    ):
        """
        FP candidate を生成する。

        current detection と、
            - other model
            - own tracker
        をそれぞれ独立に shape-aware 1-to-1 matching。

        最終条件:
          A. other にも tracker にも match しない
             -> FP candidate
          B. 片方にしか match しない AND confidence <= threshold
             -> FP candidate

        これにより、IoU=0 でも shape/position が近ければ
        tracker-supported と判定できる。
        """
        other_name = (
            self.model2_name
            if model_name == self.model1_name
            else self.model1_name
        )

        other = current_detections[other_name]
        tracker = tracker_predictions[model_name]

        # モデル間は従来どおり IoU 1-to-1。
        match_other, unmatched_current_other, _ = _iou_match_one_to_one(
            detections, other, self.iou_threshold
        )
        # detection <-> tracker だけ shape-aware 1-to-1。
        match_tracker, unmatched_current_tracker, _ = self._match(
            detections, tracker
        )

        matched_other = {i: info for i, _, info in match_other}
        matched_tracker = {i: info for i, _, info in match_tracker}

        candidates = []

        for idx, box in enumerate(detections):
            has_other = idx in matched_other
            has_tracker = idx in matched_tracker

            suspicious = not (has_other and has_tracker)
            confidence = _get_confidence(box)

            final_candidate = (
                (not has_other and not has_tracker)
                or (
                    suspicious
                    and confidence <= self.low_confidence_threshold
                )
            )

            if not suspicious:
                continue

            other_info = matched_other.get(idx)
            print(other_info)
            tracker_info = matched_tracker.get(idx)

            candidates.append({
                "box": box,
                "source_index": idx,
                "other_matched": int(has_other),
                "tracker_matched": int(has_tracker),
                "confidence": confidence,
                "final": bool(final_candidate),

                "other_shape_distance": (
                    other_info["distance"] if other_info else None
                ),
                "other_center_distance": (
                    other_info["center_distance"] if other_info else None
                ),
                "other_size_distance": (
                    other_info["size_distance"] if other_info else None
                ),
                "other_ratio_distance": (
                    other_info["ratio_distance"] if other_info else None
                ),

                "tracker_shape_distance": (
                    tracker_info["distance"] if tracker_info else None
                ),
                "tracker_center_distance": (
                    tracker_info["center_distance"] if tracker_info else None
                ),
                "tracker_size_distance": (
                    tracker_info["size_distance"] if tracker_info else None
                ),
                "tracker_ratio_distance": (
                    tracker_info["ratio_distance"] if tracker_info else None
                ),
            })

        return candidates

    def _fn_candidates(
        self,
        model_name,
        detections,
        current_detections,
        tracker_predictions,
    ):
        """
        FN candidate を生成する。

        other model と own tracker をそれぞれ current detection に
        1 対 1 matching し、unmatched evidence を取り出す。

        その後、
            unmatched other <-> unmatched tracker
        も shape-aware 1-to-1 matching して、

            other + tracker -> 1 candidate

        と統合する。

        片方の evidence しかなくても FN candidate とする。
        """
        other_name = (
            self.model2_name
            if model_name == self.model1_name
            else self.model1_name
        )

        other = current_detections[other_name]
        tracker = tracker_predictions[model_name]

        # other model <-> current detection は IoU 1-to-1。
        _, _, unmatched_other = _iou_match_one_to_one(
            other, detections, self.iou_threshold
        )
        # tracker <-> current detection は shape-aware 1-to-1。
        _, _, unmatched_tracker = self._match(
            tracker, detections
        )

        # current detection と match しなかった evidence。
        other_unmatched_boxes = [other[i] for i in unmatched_other]
        tracker_unmatched_boxes = [tracker[i] for i in unmatched_tracker]

        merged = _merge_evidence_one_to_one(
            other_unmatched_boxes,
            tracker_unmatched_boxes,
            max_distance=self.max_match_distance,
            max_center_distance=self.max_center_distance,
            max_size_ratio=self.max_size_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            center_weight=self.center_weight,
            size_weight=self.size_weight,
            ratio_weight=self.ratio_weight,
        )

        return merged

    def _process_frame(self, frame_idx, image_path: Path):
        m1 = self.model1.predict(image_path)
        m2 = self.model2.predict(image_path)

        current = {
            self.model1_name: m1,
            self.model2_name: m2,
        }

        # Tracker は現在 detection から更新しつつ、
        # update() が返す predicted_boxes は current detection より前の予測として利用。
        t1 = self.tracker1.update(m1).predicted_boxes
        t2 = self.tracker2.update(m2).predicted_boxes

        tracker_predictions = {
            self.model1_name: t1,
            self.model2_name: t2,
        }

        # GT は評価専用。
        gt_boxes = self._load_gt(image_path)
        c1 = self.classifier.classify(gt_boxes, m1)
        c2 = self.classifier.classify(gt_boxes, m2)

        actual = {
            self.model1_name: {
                "fp": [
                    b for b in c1
                    if b.classifyCategory == ClassifyCategory.FP
                ],
                "fn": [
                    b for b in c1
                    if b.classifyCategory == ClassifyCategory.FN
                ],
            },
            self.model2_name: {
                "fp": [
                    b for b in c2
                    if b.classifyCategory == ClassifyCategory.FP
                ],
                "fn": [
                    b for b in c2
                    if b.classifyCategory == ClassifyCategory.FN
                ],
            },
        }

        pred = {}

        for model_name in (self.model1_name, self.model2_name):
            fp_candidates = self._fp_candidates(
                model_name,
                current[model_name],
                current,
                tracker_predictions,
            )
            fn_candidates = self._fn_candidates(
                model_name,
                current[model_name],
                current,
                tracker_predictions,
            )

            pred[model_name] = {
                "fp": fp_candidates,
                "fn": fn_candidates,
                "hard_fp": [
                    c["box"] for c in fp_candidates if c["final"]
                ],
                "hard_fn": [
                    c["box"] for c in fn_candidates if c["final"]
                ],
            }

        record = {
            "frame_idx": frame_idx,
            "m1_count": len(m1),
            "m2_count": len(m2),
            "m1_tracker_count": len(t1),
            "m2_tracker_count": len(t2),
        }

        for model_name in (self.model1_name, self.model2_name):
            for error_type in ("fp", "fn"):
                candidates = pred[model_name][error_type]
                hard = pred[model_name][f"hard_{error_type}"]
                actual_boxes = actual[model_name][error_type]

                for candidate_idx, candidate in enumerate(candidates):
                    self.candidate_records.append({
                        "frame_idx": frame_idx,
                        "image": image_path.name,
                        "model": model_name,
                        "error_type": error_type,
                        "candidate_idx": candidate_idx,
                        "final_candidate": int(candidate["final"]),

                        "confidence": candidate.get("confidence"),

                        "other_matched": candidate.get(
                            "other_matched", None
                        ),
                        "tracker_matched": candidate.get(
                            "tracker_matched", None
                        ),

                        "other_shape_distance": candidate.get(
                            "other_shape_distance", None
                        ),
                        "other_center_distance": candidate.get(
                            "other_center_distance", None
                        ),
                        "other_size_distance": candidate.get(
                            "other_size_distance", None
                        ),
                        "other_ratio_distance": candidate.get(
                            "other_ratio_distance", None
                        ),

                        "tracker_shape_distance": candidate.get(
                            "tracker_shape_distance", None
                        ),
                        "tracker_center_distance": candidate.get(
                            "tracker_center_distance", None
                        ),
                        "tracker_size_distance": candidate.get(
                            "tracker_size_distance", None
                        ),
                        "tracker_ratio_distance": candidate.get(
                            "tracker_ratio_distance", None
                        ),

                        "sources": (
                            "+".join(sorted(candidate["sources"]))
                            if "sources" in candidate
                            else None
                        ),
                        "shape_distance": candidate.get(
                            "shape_distance", None
                        ),
                        "center_distance": candidate.get(
                            "center_distance", None
                        ),
                        "size_distance": candidate.get(
                            "size_distance", None
                        ),
                        "ratio_distance": candidate.get(
                            "ratio_distance", None
                        ),
                    })

                # 評価時だけ GT と IoU matching。
                matches = self._evaluate_prediction_vs_gt(
                    hard,
                    actual_boxes,
                )

                record[f"{model_name}_{error_type}_candidate_count"] = len(
                    candidates
                )
                record[f"{model_name}_{error_type}_final_count"] = len(hard)
                record[f"{model_name}_{error_type}_tp"] = len(matches)
                record[f"{model_name}_{error_type}_actual"] = len(
                    actual_boxes
                )

                # FN candidate source 内訳。
                if error_type == "fn":
                    record[
                        f"{model_name}_fn_other_tracker_count"
                    ] = sum(
                        1 for c in candidates
                        if c.get("sources") == {"other", "tracker"}
                    )
                    record[
                        f"{model_name}_fn_other_only_count"
                    ] = sum(
                        1 for c in candidates
                        if c.get("sources") == {"other"}
                    )
                    record[
                        f"{model_name}_fn_tracker_only_count"
                    ] = sum(
                        1 for c in candidates
                        if c.get("sources") == {"tracker"}
                    )

        # Frame-level dominant error type: final candidate count only.
        for model_name in (self.model1_name, self.model2_name):
            fp_est = record[f"{model_name}_fp_final_count"]
            fn_est = record[f"{model_name}_fn_final_count"]

            if abs(fp_est - fn_est) <= self.dominant_margin:
                predicted = "balanced"
            elif fp_est > fn_est:
                predicted = "FP"
            else:
                predicted = "FN"

            actual_fp = len(actual[model_name]["fp"])
            actual_fn = len(actual[model_name]["fn"])

            if actual_fp == actual_fn:
                actual_type = "balanced"
            elif actual_fp > actual_fn:
                actual_type = "FP"
            else:
                actual_type = "FN"

            record[f"{model_name}_dominant_predicted"] = predicted
            record[f"{model_name}_dominant_actual"] = actual_type
            record[f"{model_name}_dominant_correct"] = int(
                predicted == actual_type
            )

        self.records.append(record)

        if self.save_debug:
            self._save_debug_image(
                image_path,
                frame_idx,
                current,
                tracker_predictions,
                pred,
            )

    def _evaluate_prediction_vs_gt(self, predicted_boxes, actual_boxes):
        """
        prediction と GT error boxes の評価。

        ここでは通常の IoU を使う。
        これは detection <-> tracker の prediction matching とは別物。
        """
        if not predicted_boxes or not actual_boxes:
            return []

        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise ImportError(
                "numpy と scipy が必要です。"
            ) from e

        INVALID = 1e6
        cost = np.full(
            (len(predicted_boxes), len(actual_boxes)),
            INVALID,
            dtype=float,
        )

        for i, pred in enumerate(predicted_boxes):
            for j, gt_error in enumerate(actual_boxes):
                if pred.classId != gt_error.classId:
                    continue

                iou = float(pred.computeIoU(gt_error))
                if iou >= self.iou_threshold:
                    cost[i, j] = 1.0 - iou

        rows, cols = linear_sum_assignment(cost)

        return [
            (int(i), int(j))
            for i, j in zip(rows, cols)
            if cost[i, j] < INVALID
        ]

    @staticmethod
    def _box_to_xyxy(box, width, height):
        return (
            int((box.xCenter - box.width / 2) * width),
            int((box.yCenter - box.height / 2) * height),
            int((box.xCenter + box.width / 2) * width),
            int((box.yCenter + box.height / 2) * height),
        )

    def _draw_boxes(self, image, boxes, color, label):
        h, w = image.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = self._box_to_xyxy(box, w, h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    def _save_debug_image(
        self,
        image_path,
        frame_idx,
        current,
        tracker_predictions,
        pred,
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            return

        m1 = current[self.model1_name]
        m2 = current[self.model2_name]
        t1 = tracker_predictions[self.model1_name]
        t2 = tracker_predictions[self.model2_name]

        self._draw_boxes(image, m1, (255, 0, 0), "M1")
        self._draw_boxes(image, m2, (0, 255, 0), "M2")
        self._draw_boxes(image, t1, (0, 0, 255), "T1")
        self._draw_boxes(image, t2, (0, 128, 255), "T2")

        self._draw_boxes(
            image,
            pred[self.model1_name]["hard_fp"],
            (0, 255, 255),
            "M1-FP",
        )
        self._draw_boxes(
            image,
            pred[self.model1_name]["hard_fn"],
            (255, 0, 255),
            "M1-FN",
        )
        self._draw_boxes(
            image,
            pred[self.model2_name]["hard_fp"],
            (0, 200, 255),
            "M2-FP",
        )
        self._draw_boxes(
            image,
            pred[self.model2_name]["hard_fn"],
            (200, 0, 255),
            "M2-FN",
        )

        text = (
            f"frame={frame_idx} "
            f"M1={len(m1)} M2={len(m2)} "
            f"T1={len(t1)} T2={len(t2)} "
            f"M1 FP={len(pred[self.model1_name]['hard_fp'])} "
            f"FN={len(pred[self.model1_name]['hard_fn'])} "
            f"M2 FP={len(pred[self.model2_name]['hard_fp'])} "
            f"FN={len(pred[self.model2_name]['hard_fn'])}"
        )

        cv2.putText(
            image,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(self.debug_dir / f"{image_path.stem}.jpg"),
            image,
        )

    def _save_records(self):
        if not self.records:
            return

        path = self.output_dir / "frame_records.csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(self.records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(self.records)

        print(f"Frame records saved to {path}")

    def _error_stats(self, model, error_type):
        pred = sum(
            r[f"{model}_{error_type}_final_count"]
            for r in self.records
        )
        tp = sum(
            r[f"{model}_{error_type}_tp"]
            for r in self.records
        )
        actual = sum(
            r[f"{model}_{error_type}_actual"]
            for r in self.records
        )

        precision = _safe_div(tp, pred)
        recall = _safe_div(tp, actual)
        f1 = _safe_div(
            2 * precision * recall,
            precision + recall,
        )

        return {
            "predicted": pred,
            "tp": tp,
            "actual": actual,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _dominant_accuracy(self, model):
        correct = sum(
            r[f"{model}_dominant_correct"]
            for r in self.records
        )
        total = len(self.records)

        return {
            "correct": correct,
            "total": total,
            "accuracy": _safe_div(correct, total),
        }

    def _evaluate(self):
        summary = {
            "model1": self.model1_name,
            "model2": self.model2_name,
            "dataset": self.dataset_name,
            "map": self.map_name,
            "gt_iou": self.iou_threshold,
            "max_match_distance": self.max_match_distance,
            "max_center_distance": self.max_center_distance,
            "max_size_ratio": self.max_size_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "center_weight": self.center_weight,
            "size_weight": self.size_weight,
            "ratio_weight": self.ratio_weight,
            "low_confidence_threshold": self.low_confidence_threshold,
            "n_frames": len(self.records),
        }

        print("\n===== Shape-aware error prediction result =====")

        for model in (self.model1_name, self.model2_name):
            for error_type in ("fp", "fn"):
                stats = self._error_stats(
                    model,
                    error_type,
                )

                print(
                    f"{model} {error_type.upper()}: "
                    f"precision={stats['precision']:.4f}, "
                    f"recall={stats['recall']:.4f}, "
                    f"f1={stats['f1']:.4f}, "
                    f"pred={stats['predicted']}, "
                    f"actual={stats['actual']}"
                )

                for k, v in stats.items():
                    summary[
                        f"{model}_{error_type}_{k}"
                    ] = (
                        round(v, 4)
                        if isinstance(v, float)
                        else v
                    )

            acc = self._dominant_accuracy(model)

            print(
                f"{model} dominant-error accuracy: "
                f"{acc['accuracy']:.4f} "
                f"({acc['correct']}/{acc['total']})"
            )

            for k, v in acc.items():
                summary[
                    f"{model}_dominant_{k}"
                ] = (
                    round(v, 4)
                    if isinstance(v, float)
                    else v
                )

        candidate_path = self.output_dir / "candidates.csv"

        if self.candidate_records:
            with open(candidate_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        self.candidate_records[0].keys()
                    ),
                )
                writer.writeheader()
                writer.writerows(self.candidate_records)

            print(f"Candidate records saved to {candidate_path}")

        summary_path = self.output_dir / "summary.csv"

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(summary.keys()),
            )
            writer.writeheader()
            writer.writerow(summary)

        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "M1/M2 + independent trackers + "
            "shape-aware 1-to-1 matching によるFP/FN prediction"
        )
    )

    parser.add_argument(
        "--model1",
        "--base-model",
        dest="model1",
        default="yolov8n",
    )
    parser.add_argument(
        "--model2",
        "--aux-model",
        dest="model2",
        default="ssd",
    )
    parser.add_argument(
        "--dataset",
        default="KITTI",
        choices=["CARLA", "KITTI"],
    )
    parser.add_argument("--map", default="0001")

    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help="GT error evaluation only. Detection-tracker matchingでは使用しない。",
    )

    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=2.5,
        help="shape-aware total distance の最大値",
    )
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=2.0,
        help="tracking bbox の幅・高さで正規化した中心距離の最大値",
    )
    parser.add_argument(
        "--max-size-ratio",
        type=float,
        default=2.5,
        help="width/height ratio の許容範囲。2.5なら0.4～2.5倍",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=2.0,
        help="aspect ratio の許容範囲。2なら0.5～2倍",
    )

    parser.add_argument(
        "--center-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--size-weight",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--ratio-weight",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--dominant-margin",
        type=float,
        default=0.0,
        help="FP/FN candidate count差がこの値以下ならbalanced",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="debug imageを保存しない",
    )

    args = parser.parse_args()

    exp = M1M2TwoTrackerErrorPredictionShapeMatchingExperiment(
        model1_name=args.model1,
        model2_name=args.model2,
        dataset_name=args.dataset,
        map_name=args.map,
        iou_threshold=args.iou,
        max_match_distance=args.max_match_distance,
        max_center_distance=args.max_center_distance,
        max_size_ratio=args.max_size_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        center_weight=args.center_weight,
        size_weight=args.size_weight,
        ratio_weight=args.ratio_weight,
        low_confidence_threshold=args.low_confidence_threshold,
        dominant_margin=args.dominant_margin,
        save_debug=not args.no_debug,
    )
    exp.run()
