from __future__ import annotations

import math
import numpy as np

from src.error_estimate.config import ShapeMatcherConfig
from src.error_estimate.matching._hungarian import INVALID_COST, solve_one_to_one
from src.error_estimate.types import Box, MatchResult


def _shape(box: Box) -> tuple[float, float, float]:
    width = max(float(box.width), 1e-6)
    height = max(float(box.height), 1e-6)
    return width, height, width / height


def shape_distance(det: Box, track: Box, config: ShapeMatcherConfig) -> dict[str, float]:
    dx = float(det.xCenter) - float(track.xCenter)
    dy = float(det.yCenter) - float(track.yCenter)
    tw, th, track_ar = _shape(track)
    dw, dh, det_ar = _shape(det)

    center = math.hypot(dx / tw, dy / th)
    size = 0.5 * (abs(math.log(dw / tw)) + abs(math.log(dh / th)))
    ratio = abs(math.log(det_ar / track_ar))
    total = (
        config.center_weight * center
        + config.size_weight * size
        + config.ratio_weight * ratio
    )

    return {
        "distance": float(total),
        "center_distance": float(center),
        "size_distance": float(size),
        "ratio_distance": float(ratio),
        "width_ratio": float(dw / tw),
        "height_ratio": float(dh / th),
        "aspect_ratio": float(det_ar / track_ar),
    }


class ShapeOneToOneMatcher:
    """IoU-independent shape/position matcher with strict one-to-one assignment."""

    def __init__(self, config: ShapeMatcherConfig | None = None):
        self.config = config or ShapeMatcherConfig()

    def _valid(self, metrics: dict[str, float]) -> bool:
        c = self.config
        width_ok = 1.0 / c.max_size_ratio <= metrics["width_ratio"] <= c.max_size_ratio
        height_ok = 1.0 / c.max_size_ratio <= metrics["height_ratio"] <= c.max_size_ratio
        aspect_ok = 1.0 / c.max_aspect_ratio <= metrics["aspect_ratio"] <= c.max_aspect_ratio
        return (
            metrics["distance"] <= c.max_distance
            and metrics["center_distance"] <= c.max_center_distance
            and width_ok and height_ok and aspect_ok
        )

    def match(self, left: list[Box], right: list[Box]) -> MatchResult:
        if not left or not right:
            return MatchResult([], list(range(len(left))), list(range(len(right))))

        cost = np.full((len(left), len(right)), INVALID_COST, dtype=float)
        metrics: dict[tuple[int, int], dict[str, float]] = {}

        for i, a in enumerate(left):
            for j, b in enumerate(right):
                if a.classId != b.classId:
                    continue
                info = shape_distance(a, b, self.config)
                if self._valid(info):
                    cost[i, j] = info["distance"]
                    metrics[(i, j)] = info

        return solve_one_to_one(cost, metrics)
