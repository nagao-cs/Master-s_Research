"""
selectors.py
比較手法 1: 状態遷移は提案手法（error_estimate_exp の SingleState / PairState）のまま、
追加モデルの選び方だけを替える Context。

- FixedContext   常に同じモデルを決まった順で追加する（time_aware_exp と同じ）
- RandomContext  候補から一様に選ぶ（シードを変えて複数回実行し平均する）
- OracleContext  正解データを使い、そのフレームで実際に最も多くエラーを減らすモデルを選ぶ
                 （選択で到達できる上限。候補の検出は「覗き見」として計算コストに数えない）

提案手法（エラー推定で選ぶ）は error_estimate_exp.context.SelectionContext そのもの。
"""
from __future__ import annotations

import random
from pathlib import Path

from src.Evaluation.dataset import fileReader
from src.error_estimate_exp.context import SelectionContext, SelectionFrameResult
from src.error_estimate_exp.frame import Frame
from src.eval_lib.reliability import classify_version


def candidate_pool(ctx: SelectionContext) -> list[str]:
    """追加できる候補（SelectionContext._select と同じ条件）。"""
    if len(ctx.active) >= ctx.max_version:
        return []
    return [m for m in ctx.candidates if m not in ctx.active]


class FixedContext(SelectionContext):
    def __init__(self, *args, fixed_models: tuple[str, ...], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_models = tuple(fixed_models)

    def select_partner(self, frame: Frame) -> str | None:
        return self._next_fixed()

    def select_third(self, frame: Frame) -> str | None:
        return self._next_fixed()

    def _next_fixed(self) -> str | None:
        """fixed_models のうち、まだ動いていない最初のモデル。"""
        pool = candidate_pool(self)
        return next((m for m in self.fixed_models if m in pool), None)


class RandomContext(SelectionContext):
    def __init__(self, *args, seed: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng = random.Random(seed)

    def select_partner(self, frame: Frame) -> str | None:
        return self._choice()

    def select_third(self, frame: Frame) -> str | None:
        return self._choice()

    def _choice(self) -> str | None:
        pool = candidate_pool(self)
        return self.rng.choice(pool) if pool else None


class OracleContext(SelectionContext):
    """
    候補 M ごとに、統合結果の実際のエラー数（FP + FN）がどれだけ減るかを正解データで数え、
    最も減るモデルを選ぶ。同数なら selection_cost の小さい方。
    第 2 モデル: err(M1 の検出) - err(M1 + M の統合)
    第 3 モデル: err(M1 + M2 の統合) - err(M1 + M2 + M の統合)
    追加するかどうかは提案手法と同じ状態遷移が決める（オラクルは「何を」足すかだけ）。
    """

    def __init__(self, *args, gt_dir: Path, target_class_ids: list[int], iou_threshold: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gt_dir = Path(gt_dir)
        self.targets = set(target_class_ids)
        self.iou_threshold = iou_threshold
        self._gt: list = []
        self._peek = None

    def process_image(self, image_path: Path) -> SelectionFrameResult:
        gt_path = self.gt_dir / f"{Path(image_path).stem}.txt"
        gt = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path)) if gt_path.exists() else []
        self._gt = [b for b in gt if b.classId in self.targets]

        # 候補の検出は 1 フレームに 1 回だけ実行し、選ばれたら Frame からも同じ結果を使う。
        # Frame.det を通ったモデルだけが executed（計算コスト）に数えられる。
        cache: dict[str, list] = {}

        def detect(model: str) -> list:
            if model not in cache:
                cache[model] = self.models[model].predict(image_path)
            return cache[model]

        self._peek = detect
        return self._process(detect)

    def process_cache(self, frame: dict) -> SelectionFrameResult:
        raise NotImplementedError("OracleContext は画像からの実行（online）だけに対応しています")

    def select_partner(self, frame: Frame) -> str | None:
        base = self.active[0]
        current = frame.det(base)
        return self._best(
            current,
            lambda m: self.integrator({base: frame.det(base), m: self._peek(m)})[0],
        )

    def select_third(self, frame: Frame) -> str | None:
        a, b = self.active
        current, _ = self.integrator({a: frame.det(a), b: frame.det(b)})
        return self._best(
            current,
            lambda m: self.integrator({a: frame.det(a), b: frame.det(b), m: self._peek(m)})[0],
        )

    def _best(self, current: list, fused_with) -> str | None:
        pool = candidate_pool(self)
        if not pool:
            return None
        base_err = self._errors(current)
        return max(pool, key=lambda m: (base_err - self._errors(fused_with(m)), -self.selection_cost(m)))

    def _errors(self, dets: list) -> int:
        dets = [d for d in dets if d.classId in self.targets]
        r = classify_version(self._gt, dets, self.iou_threshold)
        return len(r.fp) + len(r.fn_gt)
