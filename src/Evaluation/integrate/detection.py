from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """1つの検出結果を表現するクラス"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    frame_id: int
    model_id: int
