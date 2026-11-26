"""Frame-level evaluation helpers (pure functions, no project imports).

Place at src/error_estimate/estimation/frame_metrics.py.
"""
from __future__ import annotations


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_scores(predicted: list[bool], actual: list[bool]) -> dict[str, float]:
    """Accuracy, precision and recall of a frame-level "has error" decision.

    Positive means the frame contains at least one error of the evaluated kind.
    Precision / recall are 0.0 when their denominator is empty.
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    hit = sum(1 for p, a in zip(predicted, actual) if p and a)
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    n_pred = sum(1 for p in predicted if p)
    n_actual = sum(1 for a in actual if a)
    return {
        "accuracy": safe_div(correct, len(actual)),
        "precision": safe_div(hit, n_pred),
        "recall": safe_div(hit, n_actual),
        "n_predicted_positive": n_pred,
        "n_actual_positive": n_actual,
    }


def probability_vs_count(
    probabilities: list[float],
    counts: list[int],
    n_bins: int = 5,
) -> list[dict]:
    """Group frames by predicted probability and summarise the actual error counts.

    If the probability is informative, error_frame_rate and mean_error_count
    rise from the low bins to the high bins. A constant prediction ends up in a
    single bin, which makes it visible.
    """
    if len(probabilities) != len(counts):
        raise ValueError("probabilities and counts must have the same length")
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(probabilities):
        b = min(max(int(p * n_bins), 0), n_bins - 1)
        groups.setdefault(b, []).append(i)

    rows = []
    for b in sorted(groups):
        idx = groups[b]
        n = len(idx)
        rows.append({
            "probability_bin": f"{b / n_bins:.2f}-{(b + 1) / n_bins:.2f}",
            "n_frames": n,
            "mean_probability": round(sum(probabilities[i] for i in idx) / n, 6),
            "error_frame_rate": round(sum(1 for i in idx if counts[i] > 0) / n, 6),
            "mean_error_count": round(sum(counts[i] for i in idx) / n, 6),
        })
    return rows