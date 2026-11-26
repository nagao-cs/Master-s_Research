from __future__ import annotations

from src.error_estimate.config import ErrorPredictionConfig
from src.error_estimate.types import Box, ErrorCandidate, FramePrediction, Matcher


def get_confidence(box: Box) -> float:
    for name in ("confidence", "conf", "score", "confidenceScore"):
        value = getattr(box, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 1.0


class OneSourceRuleBasedEstimator:
    """1モデル + 1トラッカー専用の rule-based FP/FN estimator.

    判定規則:
      FP: detection が tracker とマッチせず、かつ confidence <= threshold
      FN: tracker prediction が detection とマッチしない

    detection-tracker の対応付けは注入された matcher に任せるため、
    ShapeOneToOneMatcher など別の 1-to-1 matcher に交換できる。
    """

    def __init__(
        self,
        tracker_matcher: Matcher,
        config: ErrorPredictionConfig | None = None,
    ):
        self.tracker_matcher = tracker_matcher
        self.config = config or ErrorPredictionConfig()

    def predict(
        self,
        current: list[Box],
        tracker: list[Box],
    ) -> FramePrediction:
        result = self.tracker_matcher.match(current, tracker)

        matched_current = {m.left_index: m for m in result.matches}
        matched_tracker = {m.right_index: m for m in result.matches}

        fp: list[ErrorCandidate] = []
        fn: list[ErrorCandidate] = []
        threshold = self.config.low_confidence_threshold

        # FP: detection-only かつ low confidence
        for i in result.unmatched_left:
            box = current[i]
            confidence = get_confidence(box)
            if confidence <= threshold:
                fp.append(
                    ErrorCandidate(
                        box=box,
                        final=True,
                        sources={"current"},
                        metadata={
                            "confidence": confidence,
                            "tracker_matched": 0,
                        },
                    )
                )

        # FN: tracker-only（detection と不一致なら confidence 条件なし）
        for j in result.unmatched_right:
            box = tracker[j]
            fn.append(
                ErrorCandidate(
                    box=box,
                    final=True,
                    sources={"tracker"},
                    metadata={"tracker_matched": 0},
                )
            )

        return FramePrediction(fp=fp, fn=fn)


class TwoSourceRuleBasedEstimator:
    """Current detection + other model + own tracker による estimator."""

    def __init__(
        self,
        model_matcher: Matcher,
        tracker_matcher: Matcher,
        evidence_matcher: Matcher | None = None,
        config: ErrorPredictionConfig | None = None,
    ):
        self.model_matcher = model_matcher
        self.tracker_matcher = tracker_matcher
        self.evidence_matcher = evidence_matcher or tracker_matcher
        self.config = config or ErrorPredictionConfig()

    def predict(self, current: list[Box], other: list[Box], tracker: list[Box]) -> FramePrediction:
        fp = self._predict_fp(current, other, tracker)
        fn = self._predict_fn(current, other, tracker)
        return FramePrediction(fp=fp, fn=fn)

    def _predict_fp(self, current: list[Box], other: list[Box], tracker: list[Box]) -> list[ErrorCandidate]:
        model_result = self.model_matcher.match(current, other)
        tracker_result = self.tracker_matcher.match(current, tracker)
        other_by_current = {m.left_index: m for m in model_result.matches}
        tracker_by_current = {m.left_index: m for m in tracker_result.matches}

        candidates: list[ErrorCandidate] = []
        threshold = self.config.low_confidence_threshold

        for i, box in enumerate(current):
            model_match = other_by_current.get(i)
            tracker_match = tracker_by_current.get(i)
            has_other = model_match is not None
            has_tracker = tracker_match is not None
            suspicious = not (has_other and has_tracker)
            confidence = get_confidence(box)

            final = (not has_other and not has_tracker) or (suspicious and confidence <= threshold)
            if not suspicious:
                continue

            metadata = {
                "confidence": confidence,
                "other_matched": int(has_other),
                "tracker_matched": int(has_tracker),
            }
            if model_match:
                metadata.update({f"other_{k}": v for k, v in model_match.metrics.items()})
            if tracker_match:
                metadata.update({f"tracker_{k}": v for k, v in tracker_match.metrics.items()})

            candidates.append(ErrorCandidate(box=box, final=final, metadata=metadata))

        return candidates

    def _predict_fn(self, current: list[Box], other: list[Box], tracker: list[Box]) -> list[ErrorCandidate]:
        other_result = self.model_matcher.match(other, current)
        tracker_result = self.tracker_matcher.match(tracker, current)

        other_boxes = [other[i] for i in other_result.unmatched_left]
        # tracker_boxes = [tracker[i] for i in tracker_result.unmatched_left]

        # evidence_result = self.evidence_matcher.match(other_boxes, tracker_boxes)
        candidates: list[ErrorCandidate] = []

        # for m in evidence_result.matches:
        #     candidates.append(ErrorCandidate(
        #         box=tracker_boxes[m.right_index],
        #         sources={"other", "tracker"},
        #         metadata=dict(m.metrics),
        #     ))

        # for i in evidence_result.unmatched_left:
        for i in range(len(other_boxes)):
            box = other_boxes[i]
            conf = get_confidence(box)
            if conf >= self.config.low_confidence_threshold:
                candidates.append(ErrorCandidate(box=box, sources={"other"}))

        # for j in evidence_result.unmatched_right:
        #     box = tracker_boxes[j]
        #     conf = get_confidence(box)
        #     if conf >= self.config.low_confidence_threshold:
        #         candidates.append(ErrorCandidate(box=box, sources={"tracker"}))

        return candidates
