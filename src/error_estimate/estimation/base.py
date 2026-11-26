from __future__ import annotations

from typing import Protocol

from src.error_estimate.types import Box, FramePrediction


class ErrorEstimator(Protocol):
    def predict(
        self,
        current: list[Box],
        other: list[Box],
        tracker: list[Box],
    ) -> FramePrediction:
        ...
