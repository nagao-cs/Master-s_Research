from .loader import DetectionResultLoader
from .analyzer import DetectionAnalyzer
from .statistics import DetectionStats
from .controller import AdaptiveController


class Dataset:
    def __init__(self, gt_dir, det_dirs, iou_th, adaptive, instance_threshold, confidence_threshold):
        self.loader = DetectionResultLoader(gt_dir, det_dirs)
        self.analyzer = DetectionAnalyzer(iou_th)
        self.stats = DetectionStats()
        self.controller = AdaptiveController(adaptive=adaptive,
                                             instance_threshold=instance_threshold, confidence_threshold=confidence_threshold)
