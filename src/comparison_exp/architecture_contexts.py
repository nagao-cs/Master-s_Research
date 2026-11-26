"""
architecture_contexts.py
比較手法 2: 先行研究の構成を AdROD の枠組み（Frame・統合器・コスト計上）に移したもの。
どちらもフレームごとに完結し、フレームをまたいだ状態は持たない。
出力は config の integrate_way の統合器（conf_base なら確信度の和が tau_p 以上のグループを採用）を通す。
1 モデルだけの出力も同じ統合器に 1 モデルで通し、後処理をそろえる。

- RecoveryBlockContext   Afrin & Machida のリカバリブロック [2]
    主ブロック（YOLOv8n + YOLOv11n）を常に動かし、2 モデルの一致度が threshold 以下なら
    主ブロックの出力は捨て、予備ブロック（RT-DETR）単独で検出し直してその出力を使う。
- ConfidenceExitContext  Inoue の確信度による Early Exit [13]
    order の先頭モデルから始め、直前に追加したモデルの検出の平均確信度が
    confidence_threshold 以下なら次のモデルを追加する（最大 len(order) = 3 モデル）。
    出力はそのフレームで実行した全モデルのアンサンブル。
"""
from __future__ import annotations

from src.error_estimate_exp.context import SelectionContext
from src.error_estimate_exp.state import SelectionState, agreement


# ── リカバリブロック ─────────────────────────────────────────

class RecoveryBlockState(SelectionState):
    def process(self, ctx: "RecoveryBlockContext", frame):
        a, b = ctx.primary
        result, groups = ctx.integrator({a: frame.det(a), b: frame.det(b)})
        if agreement(groups) > ctx.rb_threshold:
            return result
        # 主ブロックが不一致: 予備ブロックの出力に置き換える
        backup, _ = ctx.integrator({ctx.backup: frame.det(ctx.backup)})
        return backup


class RecoveryBlockContext(SelectionContext):
    def __init__(self, *args, primary: tuple[str, str], backup: str, threshold: float, **kwargs) -> None:
        self.primary = tuple(primary)
        self.backup = backup
        self.rb_threshold = threshold
        super().__init__(*args, **kwargs)
        self.active = self.primary

    def single_state(self) -> SelectionState:
        return RecoveryBlockState()


# ── 確信度による Early Exit ──────────────────────────────────

def mean_confidence(dets: list) -> float:
    """検出の平均確信度。検出がないフレームは 0（不確実として扱い、次のモデルを追加する）。"""
    if not dets:
        return 0.0
    return sum(d.confidenceScore for d in dets) / len(dets)


class ConfidenceExitState(SelectionState):
    def process(self, ctx: "ConfidenceExitContext", frame):
        used: dict[str, list] = {}
        for model in ctx.order:
            used[model] = frame.det(model)
            if mean_confidence(used[model]) > ctx.confidence_threshold:
                break
        result, _ = ctx.integrator(used)
        return result


class ConfidenceExitContext(SelectionContext):
    def __init__(self, *args, order: tuple[str, ...], confidence_threshold: float, **kwargs) -> None:
        self.order = tuple(order)
        self.confidence_threshold = confidence_threshold
        super().__init__(*args, **kwargs)
        self.active = (self.order[0],)

    def single_state(self) -> SelectionState:
        return ConfidenceExitState()
