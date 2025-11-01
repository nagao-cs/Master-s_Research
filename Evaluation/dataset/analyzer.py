class DetectionAnalyzer:
    def __init__(self, iou_th: float):
        self.iou_th = iou_th

    def analyze_frame(self, gt_path: str, det_paths: list[str]) -> dict:
        frame_results = dict()
