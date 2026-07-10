from ultralytics import YOLO
import torch
import numpy as np
from typing import Optional

from .ObjectDetector import Detector, BBoxFormat

class Yolov11nDetector(Detector):
    def load_model(self, model):
        self.model = YOLO(model)
        
    def _run_model(self, image: np.ndarray) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], BBoxFormat]:
        detectionResult = self.model.predict(
            source=image,
            device=self.device,
            verbose=False
        )[0]
        
        rawBoundingBoxList = detectionResult.boxes
        
        if rawBoundingBoxList is None:
            return (None, None, None, BBoxFormat.XYWH_NORM)
        xywhn = torch.as_tensor(rawBoundingBoxList.xywhn, device=self.device)
        conf = torch.as_tensor(rawBoundingBoxList.conf, device=self.device)
        cls = torch.as_tensor(rawBoundingBoxList.cls, device=self.device)

        return (
            xywhn,
            conf,
            cls,
            BBoxFormat.XYWH_NORM
        )
    
    def _get_class_id_offset(self) -> int:
        return 0  # Ultralytics 0-indexed