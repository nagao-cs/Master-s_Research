from __future__ import annotations

from src.error_estimate.tracking.base import predicted_boxes


class SortPredictionAdapter:
    """Wraps the project's SortTracker without coupling the estimator to it."""

    def __init__(self, tracker):
        self.tracker = tracker

    def update(self, detections):
        return predicted_boxes(self.tracker.update(detections))
