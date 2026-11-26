from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from error_estimate.types import MatchInfo, MatchResult

INVALID_COST = 1e9


def solve_one_to_one(cost: np.ndarray, metrics: dict[tuple[int, int], dict[str, float]]) -> MatchResult:
    if cost.size == 0:
        return MatchResult(
            matches=[],
            unmatched_left=list(range(cost.shape[0])),
            unmatched_right=list(range(cost.shape[1])),
        )

    rows, cols = linear_sum_assignment(cost)
    matches: list[MatchInfo] = []
    used_left: set[int] = set()
    used_right: set[int] = set()

    for left, right in zip(rows, cols):
        if cost[left, right] >= INVALID_COST:
            continue
        left = int(left)
        right = int(right)
        matches.append(MatchInfo(left, right, metrics[(left, right)]))
        used_left.add(left)
        used_right.add(right)

    return MatchResult(
        matches=matches,
        unmatched_left=[i for i in range(cost.shape[0]) if i not in used_left],
        unmatched_right=[j for j in range(cost.shape[1]) if j not in used_right],
    )
