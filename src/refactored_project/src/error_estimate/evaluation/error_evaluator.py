from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from error_estimate.matching._hungarian import INVALID_COST


def match_prediction_to_gt(predicted_boxes, actual_boxes, iou_threshold: float):
    if not predicted_boxes or not actual_boxes:
        return []

    cost = np.full((len(predicted_boxes), len(actual_boxes)), INVALID_COST, dtype=float)
    for i, pred in enumerate(predicted_boxes):
        for j, gt_error in enumerate(actual_boxes):
            if pred.classId != gt_error.classId:
                continue
            iou = float(pred.computeIoU(gt_error))
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou

    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < INVALID_COST]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def error_stats(records, model: str, error_type: str):
    predicted = sum(r[f"{model}_{error_type}_final_count"] for r in records)
    tp = sum(r[f"{model}_{error_type}_tp"] for r in records)
    actual = sum(r[f"{model}_{error_type}_actual"] for r in records)
    precision = safe_div(tp, predicted)
    recall = safe_div(tp, actual)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"predicted": predicted, "tp": tp, "actual": actual,
            "precision": precision, "recall": recall, "f1": f1}


def dominant_accuracy(records, model: str):
    correct = sum(r[f"{model}_dominant_correct"] for r in records)
    total = len(records)
    return {"correct": correct, "total": total, "accuracy": safe_div(correct, total)}
