import numpy as np
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from .ObjectDetector import Detector, BBoxFormat

class SSDDetector(Detector):
    def load_model(self):
        weights = SSD300_VGG16_Weights.COCO_V1
        self.model = ssd300_vgg16(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def _run_model(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BBoxFormat]:
        imageTensor = self._convert_to_rgb_tensor(image)
        with torch.no_grad():
            detections = self.model([imageTensor])[0]
        
        return (
            detections["boxes"],        # xyxy形式
            detections["scores"],
            detections["labels"],
            BBoxFormat.XYXY
        )
    
    def _get_class_id_offset(self) -> int:
        return -1
    
    def _get_flops_input_tensor(self, image: np.ndarray) -> tuple:
        """FLOPs計算用の入力（モデルと同じ形式）"""
        imageTensor = self._convert_to_rgb_tensor(image)
        return (imageTensor,)  # モデルへの入力はリストだが、profile用にタプルで返す