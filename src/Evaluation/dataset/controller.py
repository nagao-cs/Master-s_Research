from typing import Dict, List, Callable, Tuple
import numpy as np


class AdaptiveController:
    def __init__(self, adaptive: bool, instance_threshold: int, confidence_threshold: float):
        self.adaptive = adaptive
        self.instance_threshold = instance_threshold
        self.confidence_threshold = confidence_threshold

    def control_mode(self, dets: Dict[int, Dict[int, Tuple[float]]]) -> str:
        if not self.adaptive:
            return "multi-version"
        if self.rule(dets):
            return "multi-version"
        else:
            return "1version"

    def select_rule(self, rule: Callable[[Dict[int, Dict[int, Tuple[float]]]], bool]) -> None:
        self.rule = rule

    def instance_rule(self, dets: Dict[int, Dict[int, Tuple[float]]]) -> bool:
        base_det = dets[0]
        total_instances = sum(len(boxes) for boxes in base_det.values())
        if total_instances >= self.instance_threshold:
            return True
        else:
            return False

    def confidence_rule(self, dets: Dict[int, Dict[int, Tuple[float]]]) -> bool:
        base_det = dets[0]
        confidences = list()
        for boxes in base_det.values():
            boxes = np.array(boxes)
            confidences.extend(boxes[:, -1])
        confidences.sort()
        worst_conf = confidences[0]
        if (not confidences) or worst_conf < self.confidence_threshold:
            return True
        else:
            return False
