import numpy as np
import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

from .ObjectDetector import Detector, BBoxFormat


class RetinanetDetector(Detector):
    def load_model(self, model):
        try:
            self.model = model
            print(self.model.score_thresh)
            print(self.model.nms_thresh)
            print(self.model.detections_per_img)
            self.model.to(self.device)
            self.model.eval()
            print(f"PyTorch RetinaNet model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading RetinaNet model: {e}")


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