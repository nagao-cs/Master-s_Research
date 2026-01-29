SIZE_THRESHOLD = 100
CONF_THRESHOLD = 0.25

# utils.py

# 1. モデル出力ID -> 標準COCO ID へのマッピング
# (TensorFlow HubのSSD MobileNet V1 FPNなど、欠番を詰めたIDを出力するモデル用)
COCO_ID_MAPPER = {
    0: 0,
    1: 0, 2: 2, 3: 2, 4: 4, 5: 2, 6: 6, 7: 2, 8: 8, 9: 9, 10: 9,
    11: 13, 12: 11, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20,
    19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
    27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39,
    35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48,
    43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
    51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64,
    59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76,
    67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 83, 74: 84,
    75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90
}

VALID_CLASS_ID = (0, 2, 9, 11)


# 2. 標準COCO ID -> ラベル名 へのマッピング
COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    11: 'reserved', 12: 'reserved', 13: 'fire hydrant', 13: 'stop sign',
    15: 'parking meter', 16: 'bench', 17: 'bird', 18: 'cat', 19: 'dog',
    20: 'horse', 21: 'sheep', 22: 'cow', 23: 'elephant', 24: 'bear',
    25: 'zebra', 26: 'reserved', 27: 'giraffe', 28: 'backpack',
    29: 'reserved', 30: 'reserved', 31: 'umbrella', 32: 'handbag',
    33: 'tie', 34: 'suitcase', 35: 'frisbee', 36: 'skis',
    37: 'snowboard', 38: 'sports ball', 39: 'kite', 40: 'baseball bat',
    41: 'baseball glove', 42: 'skateboard', 43: 'surfboard', 44: 'tennis racket',
    45: 'reserved', 46: 'bottle', 47: 'wine glass', 48: 'cup',
    49: 'fork', 50: 'knife', 51: 'spoon', 52: 'bowl', 53: 'banana',
    54: 'apple', 55: 'sandwich', 56: 'orange', 57: 'broccoli', 58: 'carrot',
    59: 'hot dog', 60: 'pizza', 61: 'donut', 62: 'cake', 63: 'chair',
    64: 'couch', 65: 'potted plant', 66: 'reserved', 67: 'bed',
    68: 'reserved', 69: 'reserved', 70: 'dining table', 71: 'reserved',
    72: 'toilet', 73: 'tv', 74: 'laptop', 75: 'mouse', 76: 'remote',
    77: 'keyboard', 78: 'cell phone', 79: 'microwave', 80: 'oven',
    81: 'toaster', 82: 'sink', 83: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}
