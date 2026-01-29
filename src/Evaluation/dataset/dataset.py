from .loader import DetectionResultLoader
from .analyzer import DetectionAnalyzer
from .metrics import DetectionMetrics
from .controller import AdaptiveController
from .stats import DetectionStats
from .integrater import DetectionIntegrator


class Dataset:
    def __init__(self, gt_dir, det_dirs, iou_th, adaptive, instance_threshold, confidence_threshold):
        self.loader = DetectionResultLoader(gt_dir, det_dirs)
        self.analyzer = DetectionAnalyzer(iou_th)
        self.metrics = DetectionMetrics()
        self.controller = AdaptiveController(adaptive=adaptive,
                                             instance_threshold=instance_threshold, confidence_threshold=confidence_threshold)
        self.stats = DetectionStats()
        self.integrator = DetectionIntegrator(
            iou_th, num_version=len(det_dirs))
