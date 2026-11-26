from __future__ import annotations

import numpy as np

from src.error_estimate.config import ShapeMatcherConfig
from src.error_estimate.matching._hungarian import INVALID_COST, solve_one_to_one
from src.error_estimate.types import Box, MatchResult


class IoUOneToOneMatcher:
    """Same-class IoU-gated Hungarian one-to-one matcher."""

    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold

    def match(self, left: list[Box], right: list[Box]) -> MatchResult:
        if not left or not right:
            return MatchResult([], list(range(len(left))), list(range(len(right))))

        cost = np.full((len(left), len(right)), INVALID_COST, dtype=float)
        metrics: dict[tuple[int, int], dict[str, float]] = {}

        for i, a in enumerate(left):
            for j, b in enumerate(right):
                if a.classId != b.classId:
                    continue
                iou = float(a.computeIoU(b))
                if iou >= self.iou_threshold:
                    cost[i, j] = 1.0 - iou
                    metrics[(i, j)] = {"iou": iou, "distance": 1.0 - iou}

        return solve_one_to_one(cost, metrics)
