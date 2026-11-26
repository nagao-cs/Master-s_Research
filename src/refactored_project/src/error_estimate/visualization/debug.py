from __future__ import annotations

import cv2


def box_to_xyxy(box, width, height):
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
        cv2.putText(image, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_debug_image(path, current, trackers, predictions, title):
    image = cv2.imread(str(path))
    if image is None:
        return None

    draw_boxes(image, current["m1"], (255, 0, 0), "M1")
    draw_boxes(image, current["m2"], (0, 255, 0), "M2")
    draw_boxes(image, trackers["m1"], (0, 0, 255), "T1")
    draw_boxes(image, trackers["m2"], (0, 128, 255), "T2")
    draw_boxes(image, predictions["m1"]["fp"], (0, 255, 255), "M1-FP")
    draw_boxes(image, predictions["m1"]["fn"], (255, 0, 255), "M1-FN")
    draw_boxes(image, predictions["m2"]["fp"], (0, 200, 255), "M2-FP")
    draw_boxes(image, predictions["m2"]["fn"], (200, 0, 255), "M2-FN")

    cv2.putText(image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return image
