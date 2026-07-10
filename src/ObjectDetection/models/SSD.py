import numpy as np
import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

from .ObjectDetector import Detector, BBoxFormat

class SSDDetector(Detector):
    def load_model(self, model):
        try:
            self.model = model
            print(self.model.score_thresh)
            print(self.model.nms_thresh)
            print(self.model.detections_per_img)
            self.model.to(self.device)
            self.model.eval()
            print(f"PyTorch SSD model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading SSD model: {e}")

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