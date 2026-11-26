from __future__ import annotations

from typing import Protocol

from error_estimate.types import Box


class Tracker(Protocol):
    def update(self, detections: list[Box]) -> object:
        ...


def predicted_boxes(update_result: object) -> list[Box]:
    return list(getattr(update_result, "predicted_boxes"))
