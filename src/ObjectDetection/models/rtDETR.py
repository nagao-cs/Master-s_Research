from ultralytics import RTDETR
import numpy as np
import torch

from .ObjectDetector import Detector, BBoxFormat


class RTDETRDetector(Detector):
    def load_model(self):
        try:
            self.model = RTDETR("rtdetr-l.pt")
            print("RTDETR model loaded")
        except Exception as e:
            raise RuntimeError(f"Error loading rtdetr-l model: {e}")

    def _run_model(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BBoxFormat]:
        detectionResult = self.model.predict(
            source=image,
            device=self.device,
            verbose=False
        )[0]
        
        rawBoundingBoxList = detectionResult.boxes
        
        return (
            rawBoundingBoxList.xywhn,   # 正規化xywh形式
            rawBoundingBoxList.conf,
            rawBoundingBoxList.cls,
            BBoxFormat.XYWH_NORM
        )
    
    def _get_class_id_offset(self) -> int:
        return 0  # Ultralytics 0-indexed