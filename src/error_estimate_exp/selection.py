"""
selection.py
エラー推定と解消率から追加モデルを選ぶ（純粋関数のみ）。

    Delta(M, t) = sum_i ( E_FP^(Mi)(t) * R_FP^(Mi,M) + E_FN^(Mi)(t) * R_FN^(Mi,M) )
    M*          = argmax_M  Delta(M, t) / C(M)   （同点なら C(M) の小さい候補）

C(M) は事前に計測した実行時間。FLOPs ではモデル間の差が大きすぎて
エラーの種類に関係なく軽いモデルばかり選ばれたため、実測時間を使う。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class ErrorEstimate:
    """フレーム t における 1 モデルの FP / FN 推定件数。"""
    fp: float
    fn: float


@dataclass(frozen=True)
class Resolution:
    """候補モデルを加えたとき、対象モデルの FP / FN が解消される割合。"""
    fp: float
    fn: float


ResolutionTable = Mapping[tuple[str, str], Resolution]  # (対象モデル, 候補モデル) → 解消率


def expected_reduction(
    estimates: Mapping[str, ErrorEstimate],
    resolution: ResolutionTable,
    candidate: str,
) -> float:
    """候補モデルを加えたときの期待エラー削減量 Delta(M, t)。"""
    total = 0.0
    for target, e in estimates.items():
        r = resolution[(target, candidate)]
        total += e.fp * r.fp + e.fn * r.fn
    return total


def select_model(
    estimates: Mapping[str, ErrorEstimate],
    candidates: Iterable[str],
    resolution: ResolutionTable,
    cost: Callable[[str], float],
) -> str:
    """Delta / C が最大の候補を返す。同点なら計算コストの小さい候補を選ぶ。"""
    pool = list(candidates)
    if not pool:
        raise ValueError("candidates is empty")
    return max(
        pool,
        key=lambda m: (expected_reduction(estimates, resolution, m) / cost(m), -cost(m)),
    )