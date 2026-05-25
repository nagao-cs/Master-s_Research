"""
bbox_visualization.py
色分けbbox描画ユーティリティ
"""
import cv2
import numpy as np
from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox


def draw_detection_boxes(
    image: np.ndarray,
    boxes: list,
    color: tuple = (255, 0, 0),
    label_prefix: str = "Det",
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    検出bboxを画像に描画
    
    Args:
        image: 入力画像
        boxes: DetectionBoundingBoxリスト
        color: (B, G, R)
        label_prefix: ラベルプレフィックス
        thickness: ボックスの厚さ
        font_scale: フォントスケール
    
    Returns:
        描画済み画像
    """
    result = image.copy()
    img_h, img_w = image.shape[:2]
    
    for box in boxes:
        # 正規化座標 → ピクセル座標
        x_min = int((box.xCenter - box.width / 2) * img_w)
        y_min = int((box.yCenter - box.height / 2) * img_h)
        x_max = int((box.xCenter + box.width / 2) * img_w)
        y_max = int((box.yCenter + box.height / 2) * img_h)
        
        # ボックス描画
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # ラベル描画
        if hasattr(box, 'confidenceScore'):
            label = f"{label_prefix} {box.classId} {box.confidenceScore:.2f}"
        else:
            label = f"{label_prefix} {box.classId}"
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        bg_color = color
        cv2.rectangle(
            result,
            (x_min, y_min - text_size[1] - 4),
            (x_min + text_size[0] + 4, y_min),
            bg_color,
            -1
        )
        cv2.putText(
            result, label,
            (x_min + 2, y_min - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1
        )
    
    return result


def draw_tracker_boxes(
    image: np.ndarray,
    boxes: list,
    color: tuple = (0, 0, 255),
    label_prefix: str = "Track",
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Tracker結果のbboxを描画（draw_detection_boxesのラッパー）
    """
    return draw_detection_boxes(image, boxes, color, label_prefix, thickness, font_scale)


def draw_comparison_frame(
    image: np.ndarray,
    gt_boxes: list,
    model_boxes: list,
    tracker_boxes: list
) -> np.ndarray:
    """
    GT（グリーン）、モデル（青）、Tracker（赤）を重ねて描画
    """
    result = image.copy()
    
    # GT
    result = draw_detection_boxes(result, gt_boxes, color=(0, 255, 0), label_prefix="GT")
    # Model
    result = draw_detection_boxes(result, model_boxes, color=(255, 0, 0), label_prefix="M")
    # Tracker
    result = draw_tracker_boxes(result, tracker_boxes, color=(0, 0, 255), label_prefix="T")
    
    return result