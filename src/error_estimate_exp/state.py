"""
state.py
エラー推定でモデルを選ぶ AdROD の State。

- SingleState  基本モデル M1 とそのトラッカーの一致度が theta_track 以下なら、
               M1 のエラー推定から第 2 モデルを選び、同フレーム内で PairState へ移る。
- PairState    2 モデルの一致度が theta_high 以上なら SingleState へ戻る。
               theta_low 以下なら両モデルのエラー推定から第 3 モデルを選び、
               このフレームだけ 3 モデルで統合する（ペアはそのまま維持）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .frame import Frame

if TYPE_CHECKING:
    from .context import SelectionContext


def agreement(bbox_groups: list[list]) -> float:
    """統合グループのうち 2 ソースが一致した割合（time_aware_exp の calc_jaccard と同じ）。"""
    if not bbox_groups:
        return 1.0
    return sum(1 for g in bbox_groups if len(g) == 2) / len(bbox_groups)


class SelectionState(ABC):
    @abstractmethod
    def process(self, ctx: SelectionContext, frame: Frame) -> list:
        """フレームを処理して統合済みの検出結果を返す。"""


class SingleState(SelectionState):
    def process(self, ctx, frame):
        base = ctx.active[0]
        result, groups = ctx.integrator({base: frame.det(base), "track": frame.track(base)})
        if agreement(groups) > ctx.thresholds.theta_track:
            return result

        partner = ctx.select_partner(frame)
        if partner is None:
            return result
        ctx.transition(PairState(), (base, partner))
        return ctx.state.process(ctx, frame)


class PairState(SelectionState):
    def process(self, ctx, frame):
        a, b = ctx.active
        result, groups = ctx.integrator({a: frame.det(a), b: frame.det(b)})
        score = agreement(groups)

        if score >= ctx.thresholds.theta_high:
            ctx.transition(ctx.single_state(), (a,))
            return result
        if score > ctx.thresholds.theta_low:
            return result

        third = ctx.select_third(frame)
        if third is None:
            return result
        fused, _ = ctx.integrator({a: frame.det(a), b: frame.det(b), third: frame.det(third)})
        return fused
