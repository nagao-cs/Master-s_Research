from __future__ import annotations

from dataclasses import dataclass, field

from src.error_estimate.config import ErrorPredictionConfig
from src.error_estimate.estimation.rule_based import get_confidence
from src.error_estimate.matching import ShapeOneToOneMatcher
from src.error_estimate.types import Box

# Evidence categories (see the report, Table VII).
#   1 target detection only, low confidence
#   2 target detection only, high confidence
#   3 other model's detection only
#   4 tracker only
#   5 other model's detection and the tracker agree, target did not detect
# A target detection that agrees with the other model or the tracker is
# treated as reliable and is not an error candidate.
FP_CATEGORIES = (1, 2)
FN_CATEGORIES = (3, 4, 5)


@dataclass
class FrameEvidence:
    """Per-frame error-candidate boxes, keyed by category 1-5."""
    class_boxes: dict[int, list[Box]] = field(default_factory=dict)

    def count(self, category: int) -> int:
        return len(self.class_boxes.get(category, []))


@dataclass
class FrameErrorProfile:
    """Empirical per-category error rates of one model, fitted on calibration frames.

    fp_rates / fn_rates are keyed ``c1`` ... ``c5`` (q_k^FP for categories 1-2,
    q_k^FN for categories 3-5). A category never seen in calibration has rate 0.
    lambda_fn is the probability that a frame holds a GT object captured by none
    of the sources used to build the categories.
    """
    model_type: str
    fp_rates: dict[str, float] = field(default_factory=dict)
    fn_rates: dict[str, float] = field(default_factory=dict)
    lambda_fn: float = 0.0

    def fp_rate(self, category: int) -> float:
        return float(self.fp_rates.get(f"c{category}", 0.0))

    def fn_rate(self, category: int) -> float:
        return float(self.fn_rates.get(f"c{category}", 0.0))

    def __str__(self):
        return f"fp_rates: {self.fp_rates}\nfn_rates: {self.fn_rates}"

class FrameTypeErrorEstimator:
    """Frame-level FP/FN estimator from detector/tracker agreement.

    With one model, only categories 1, 2 and 4 exist.
    With two models, the other model's detections add categories 3 and 5.
    """

    def __init__(
        self,
        tracker_matcher: ShapeOneToOneMatcher,
        config: ErrorPredictionConfig | None = None,
        low_confidence_threshold: float | None = None,
    ):
        self.tracker_matcher = tracker_matcher
        self.config = config or ErrorPredictionConfig()
        self.low_confidence_threshold = (
            self.config.low_confidence_threshold
            if low_confidence_threshold is None
            else low_confidence_threshold
        )

    def _confidence_category(self, box: Box) -> int:
        return 1 if get_confidence(box) <= self.low_confidence_threshold else 2

    def _matched_left(self, left: list[Box], right: list[Box]) -> set[int]:
        """Indices of ``left`` boxes that have a counterpart in ``right``."""
        return {m.left_index for m in self.tracker_matcher.match(left, right).matches}

    def one_model_evidence(self, detections: list[Box], tracker: list[Box]) -> FrameEvidence:
        boxes: dict[int, list[Box]] = {1: [], 2: [], 4: []}
        result = self.tracker_matcher.match(detections, tracker)
        for i in result.unmatched_left:
            box = detections[i]
            boxes[self._confidence_category(box)].append(box)
        for j in result.unmatched_right:
            boxes[4].append(tracker[j])
        return FrameEvidence(class_boxes=boxes)

    def two_model_evidence(
        self,
        target_detections: list[Box],
        target_tracker: list[Box],
        other_detections: list[Box],
    ) -> FrameEvidence:
        boxes: dict[int, list[Box]] = {1: [], 2: [], 3: [], 4: [], 5: []}

        # Target detections not backed by the other model or the tracker (1, 2).
        backed = self._matched_left(target_detections, other_detections)
        backed |= self._matched_left(target_detections, target_tracker)
        for i, box in enumerate(target_detections):
            if i not in backed:
                boxes[self._confidence_category(box)].append(box)

        # Boxes the target did not detect: other-model-only and tracker-only,
        # then pair them up (5 = both capture the same object).
        seen_by_target = self._matched_left(other_detections, target_detections)
        other_only = [b for i, b in enumerate(other_detections) if i not in seen_by_target]
        seen_by_target = self._matched_left(target_tracker, target_detections)
        tracker_only = [b for i, b in enumerate(target_tracker) if i not in seen_by_target]

        pairs = self.tracker_matcher.match(other_only, tracker_only)
        for m in pairs.matches:
            boxes[5].append(other_only[m.left_index])
        for i in pairs.unmatched_left:
            boxes[3].append(other_only[i])
        for j in pairs.unmatched_right:
            boxes[4].append(tracker_only[j])

        return FrameEvidence(class_boxes=boxes)

    def predict(self, evidence: FrameEvidence, profile: FrameErrorProfile) -> dict[str, float]:
        fp_survival = 1.0
        fn_survival = max(0.0, 1.0 - profile.lambda_fn)
        expected_fp = 0.0
        expected_fn = profile.lambda_fn

        for k in FP_CATEGORIES:
            n, q = evidence.count(k), min(max(profile.fp_rate(k), 0.0), 1.0)
            fp_survival *= (1.0 - q) ** n
            expected_fp += n * q
        for k in FN_CATEGORIES:
            n, q = evidence.count(k), min(max(profile.fn_rate(k), 0.0), 1.0)
            fn_survival *= (1.0 - q) ** n
            expected_fn += n * q

        return {
            "fp_probability": 1.0 - fp_survival,
            "fn_probability": 1.0 - fn_survival,
            "expected_fp": expected_fp,
            "expected_fn": expected_fn,
        }