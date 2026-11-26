from __future__ import annotations

from pathlib import Path

import cv2


def box_to_xyxy(box, width: int, height: int):
    return (
        int((box.xCenter - box.width / 2) * width),
        int((box.yCenter - box.height / 2) * height),
        int((box.xCenter + box.width / 2) * width),
        int((box.yCenter + box.height / 2) * height),
    )


def draw_boxes(image, boxes, color, label):
    h, w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box_to_xyxy(box, w, h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, label, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )


def save_debug_image(
    image_path: str | Path,
    output_path: str | Path,
    current: dict,
    trackers: dict,
    predictions: dict,
    title: str = "",
):
    """任意数のモデル/トラッカーを描画して debug image を保存する。

    image_path は読み込み元、output_path は書き込み先として明確に分離する。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    for name, boxes in current.items():
        draw_boxes(image, boxes, (255, 0, 0), name)

    for name, boxes in trackers.items():
        draw_boxes(image, boxes, (0, 0, 255), f"{name}-TRK")

    for name, errors in predictions.items():
        draw_boxes(image, errors.get("fp", []), (0, 255, 255), f"{name}-FP")
        draw_boxes(image, errors.get("fn", []), (255, 0, 255), f"{name}-FN")

    if title:
        cv2.putText(image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"debug imageを保存できません: {output_path}")
    return image
