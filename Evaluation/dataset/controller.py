class AdaptiveController:
    def __init__(self, adaptive: bool, instance_threshold: int, confidence_threshold: float):
        self.adaptive = adaptive
        self.instance_threshold = instance_threshold
        self.confidence_threshold = confidence_threshold

    def control_mode(self, gt: dict, dets: dict[dict]) -> str:
        if not self.adaptive:
            return "standard"
        base_det = dets[0]
        total_instances = sum(len(boxes) for boxes in base_det.values())
        if total_instances >= self.instance_threshold:
            return "adaptive"
        else:
            return "standard"
