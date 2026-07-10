from ultralytics import YOLO
import torch
import numpy as np

from typing import Optional

from .ObjectDetector import Detector, BBoxFormat

class Yolov8lDetector(Detector):
    def load_model(self):
        self.model = YOLO("yolov8l")

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
    
    def _get_flops_input_tensor(self, image: np.ndarray) -> Optional[tuple]:
        # Ultralyticsはモデルの推論方式が特殊なため、FLOPs計算をスキップ
        return None