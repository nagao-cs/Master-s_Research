from .loader import DetectionResultLoader
from .analyzer import DetectionAnalyzer
from .statistics import DetectionStats


class Dataset:
    def __init__(self, gt_dir, det_dirs, iou_th):
        self.loader = DetectionResultLoader(gt_dir, det_dirs)
        self.analyzer = DetectionAnalyzer(iou_th)
        self.stats = DetectionStats()
