from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass
import torch
from torchvision.transforms import functional as F    
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

from src.boundingBox.boundingBox import DetectionBoundingBox
from ..metrics import MetricsCollector, PerformanceMetrics

class BBoxFormat(Enum):
    """バウンディングボックスの座標形式"""
    XYXY = "xyxy"              # [x_min, y_min, x_max, y_max] (ピクセル絶対)
    XYWH = "xywh"              # [x_center, y_center, width, height] (ピクセル)
    XYWH_NORM = "xywh_norm"    # [x_center, y_center, width, height] (正規化 0-1)


@dataclass
class RawDetection:
    """検出結果の統一中間形式（正規化座標）"""
    bboxes: torch.Tensor              # [N, 4] (x_center, y_center, width, height) 正規化
    confidence_scores: torch.Tensor   # [N]
    class_ids: torch.Tensor           # [N]
    image_height: int
    image_width: int
    device: str


class Detector:
    def __init__(self, model):
        self._setup_device()
        self.load_model(model)
        self.metrics_collector = MetricsCollector(device=self.device)

    @abstractmethod
    def load_model(self, model):
        """モデルをロード"""
        pass

    @abstractmethod
    def _run_model(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BBoxFormat]:
        """
        モデル推論を実行し、フレームワーク固有の形式で結果を返す
        
        Returns:
            (bboxes, confidence_scores, class_ids, bbox_format)
        """
        pass

    @abstractmethod
    def _get_class_id_offset(self) -> int:
        """クラスID調整オフセット（0-indexed vs 1-indexed の差分）"""
        pass
    
    def predict(self, image_path: Path) -> list[DetectionBoundingBox]:
        """5ステップの統一検出パイプライン"""
        # ステップ1: 検出の実行
        image, image_height, image_width = self._read_image(image_path)
        bboxes, confidence_scores, class_ids, bbox_format = self._run_model(image)
        
        # ステップ2: 共通形式に変換
        raw_detection = self._to_raw_detection(
            bboxes, confidence_scores, class_ids,
            image_height, image_width, bbox_format
        )
        
        # ステップ3: マスク適用
        filtered = self._apply_masks_to_raw_detection(raw_detection)
        
        # ステップ5: 最終的な結果の生成
        if len(filtered.bboxes) == 0:
            return []
        
        final_bbox_list: list[DetectionBoundingBox] = self._raw_detection_to_bbox_list(filtered, self._get_class_id_offset())
        return final_bbox_list
    
    def _setup_device(self) -> None:
        """デバイス（GPU/CPU）をセットアップ"""
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
    # ========== 共通の画像処理 ==========
    
    def _read_image(self, image_path: Path) -> tuple[np.ndarray, int, int]:
        """画像を読み込んでサイズ情報を取得"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        height, width = image.shape[:2]
        return image, height, width
    
    def _convert_to_rgb_tensor(self, image: np.ndarray) -> torch.Tensor:
        """画像をRGB Tensorに変換（torchvision用）"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(image_rgb).to(self.device)
        return tensor
    
    # ========== 形式変換メソッド ==========
    
    def _to_raw_detection(
        self,
        bboxes: torch.Tensor,
        confidence_scores: torch.Tensor,
        class_ids: torch.Tensor,
        image_height: int,
        image_width: int,
        bbox_format: BBoxFormat
    ) -> RawDetection:
        """フレームワーク固有形式 → RawDetection（統一形式）"""
        if bbox_format == BBoxFormat.XYXY:
            # torchvision: [x_min, y_min, x_max, y_max] (ピクセル)
            x_min = bboxes[:, 0]
            y_min = bboxes[:, 1]
            x_max = bboxes[:, 2]
            y_max = bboxes[:, 3]
            
            widths_norm = (x_max - x_min) / image_width
            heights_norm = (y_max - y_min) / image_height
            x_centers_norm = ((x_min + x_max) / 2) / image_width
            y_centers_norm = ((y_min + y_max) / 2) / image_height
            
            xywh_norm = torch.stack([x_centers_norm, y_centers_norm, widths_norm, heights_norm], dim=1)
        
        elif bbox_format == BBoxFormat.XYWH:
            # ピクセルxywh → 正規化xywh
            widths_norm = bboxes[:, 2] / image_width
            heights_norm = bboxes[:, 3] / image_height
            x_centers_norm = bboxes[:, 0] / image_width
            y_centers_norm = bboxes[:, 1] / image_height
            
            xywh_norm = torch.stack([x_centers_norm, y_centers_norm, widths_norm, heights_norm], dim=1)
        
        elif bbox_format == BBoxFormat.XYWH_NORM:
            # Ultralytics: 既に正規化済み
            xywh_norm = bboxes
        
        else:
            raise ValueError(f"Unknown bbox format: {bbox_format}")
        
        return RawDetection(
            bboxes=xywh_norm,
            confidence_scores=confidence_scores,
            class_ids=class_ids,
            image_height=image_height,
            image_width=image_width,
            device=self.device
        )
    
    # ========== マスク適用 ==========
    
    def _apply_masks_to_raw_detection(
        self,
        raw_detection: RawDetection,
        confidence_threshold: Optional[float] = None,
        target_classes: Optional[list[int]] = None,
        size_threshold_pixel: Optional[float] = None
    ) -> RawDetection:
        """統一フォーマットにマスク適用"""
        from ..utils import utils
        
        # デフォルト値の取得
        if confidence_threshold is None:
            confidence_threshold = utils.CONF_THRESHOLD
        if size_threshold_pixel is None:
            size_threshold_pixel = utils.SIZE_THRESHOLD
        
        bboxes = raw_detection.bboxes
        
        # 信頼度マスク
        conf_mask = raw_detection.confidence_scores >= confidence_threshold
        
        # クラスマスク
        if target_classes is None:
            class_mask = torch.ones(len(raw_detection.class_ids), dtype=torch.bool, device=raw_detection.device)
        else:
            target_tensor = torch.tensor(target_classes, device=raw_detection.device)
            class_mask = torch.isin(raw_detection.class_ids, target_tensor)
        
        # サイズマスク（正規化座標から計算）
        widths_pixel = bboxes[:, 2] * raw_detection.image_width
        heights_pixel = bboxes[:, 3] * raw_detection.image_height
        size_pixel = widths_pixel * heights_pixel
        size_mask = size_pixel >= size_threshold_pixel
        
        # 統合マスク
        combined_mask = conf_mask & class_mask & size_mask
        valid_indices = torch.where(combined_mask)[0]
        
        # マスク適用
        if len(valid_indices) == 0:
            filtered = RawDetection(
                bboxes=torch.empty((0, 4), device=raw_detection.device),
                confidence_scores=torch.empty(0, device=raw_detection.device),
                class_ids=torch.empty(0, dtype=torch.int64, device=raw_detection.device),
                image_height=raw_detection.image_height,
                image_width=raw_detection.image_width,
                device=raw_detection.device
            )
        else:
            filtered = RawDetection(
                bboxes=bboxes[valid_indices],
                confidence_scores=raw_detection.confidence_scores[valid_indices],
                class_ids=raw_detection.class_ids[valid_indices],
                image_height=raw_detection.image_height,
                image_width=raw_detection.image_width,
                device=raw_detection.device
            )
        
        return filtered
    
    # ========== 最終結果生成 ==========
    
    def _raw_detection_to_bbox_list(
        self,
        raw_detection: RawDetection,
        class_id_offset: int = 0
    ) -> list[DetectionBoundingBox]:
        """RawDetection → DetectionBoundingBoxリスト"""
        bboxes, conf_scores, class_ids = self._cpu_to_numpy(
            raw_detection.bboxes,
            raw_detection.confidence_scores,
            raw_detection.class_ids
        )
        
        detections = []
        for i in range(len(bboxes)):
            bbox = DetectionBoundingBox(
                float(bboxes[i, 0]),              # x_center
                float(bboxes[i, 1]),              # y_center
                float(bboxes[i, 2]),              # width
                float(bboxes[i, 3]),              # height
                int(class_ids[i]) + class_id_offset,
                float(conf_scores[i])
            )
            detections.append(bbox)
        
        return detections
    
    def _cpu_to_numpy(
        self,
        *tensors: torch.Tensor
    ) -> tuple[np.ndarray, ...]:
        """テンソルをNumPy配列に変換"""
        return tuple(tensor.cpu().numpy() for tensor in tensors)