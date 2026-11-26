from __future__ import annotations

from src.boundingBox.boundingBox import ClassifyCategory
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier


class GroundTruthEvaluator:
    """Project-specific GT classification kept outside the prediction library."""

    def __init__(self, iou_threshold: float):
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)

    def classify(self, gt_boxes, detections):
        classified = self.classifier.classify(gt_boxes, detections)
        return {
            "fp": [b for b in classified if b.classifyCategory == ClassifyCategory.FP],
            "fn": [b for b in classified if b.classifyCategory == ClassifyCategory.FN],
        }
