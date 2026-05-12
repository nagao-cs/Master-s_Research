import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from .ObjectDetector import Detector, BBoxFormat


class FasterRCNNDetector(Detector):

    def load_model(self):
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            print(f"PyTorch FasterRCNN model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading FasterRCNN model: {e}")

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
        return -1  # COCO 1-indexed → 0-indexed