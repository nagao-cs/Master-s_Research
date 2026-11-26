"""
context.py
エラー推定によるモデル選択を行う AdROD の Context。

モデル別にトラッカーを持ち、そのモデルを実行したフレームでだけ更新する。
SORT は連続フレームを前提とするため、前フレームで実行していないモデルの
トラッカーは作り直す。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.error_estimate.estimation.frame_type import FrameErrorProfile, FrameEvidence
from src.time_aware_exp.state.AdrodState import FrameResult

from .calibration import Calibration
from .frame import Frame
from .selection import ErrorEstimate, select_model
from .state import SelectionState, SingleState


@dataclass
class SelectionFrameResult(FrameResult):
    models: tuple[str, ...] = ()  # このフレームで実行したモデル


class SelectionContext:
    def __init__(
        self,
        thresholds,
        integrator,
        base_model: str,
        candidates: tuple[str, ...],
        max_version: int,
        calibration: Calibration,
        estimator,
        flops: Callable[[str], float],
        selection_cost: Callable[[str], float],
        tracker_factory: Callable[[], object],
        detector_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.integrator = integrator
        self.base_model = base_model
        self.candidates = candidates
        self.max_version = max_version
        self.calibration = calibration
        self.estimator = estimator
        self.flops = flops                    # 記録用（FrameResult.flops）
        self.selection_cost = selection_cost  # モデル選択用の C(M)（実行時間）
        self._tracker_factory = tracker_factory
        self._detector_factory = detector_factory

        self.state: SelectionState = self.single_state()
        self.active: tuple[str, ...] = (base_model,)
        self.models: dict[str, object] = {}
        self.trackers: dict[str, object] = {}
        self.history: list[tuple[str, ...]] = []
        self._prev_executed: set[str] = set()

    # ── 外部エントリポイント ──────────────────────────────────

    def ready_model(self) -> None:
        """選ばれうる全モデルを先に読み込む（キャッシュモードでは何もしない）。"""
        if self._detector_factory is None:
            return
        for name in dict.fromkeys((self.base_model, *self.candidates)):
            if name not in self.models:
                self.models[name] = self._detector_factory(name)
                print(f"✓ Model '{name}' loaded")

    def process_image(self, image_path: Path) -> SelectionFrameResult:
        return self._process(lambda m: self.models[m].predict(image_path))

    def process_cache(self, frame: dict) -> SelectionFrameResult:
        return self._process(frame.__getitem__)

    # ── 状態遷移（State から呼ぶ） ───────────────────────────

    def transition(self, next_state: SelectionState, active: tuple[str, ...]) -> None:
        self.state = next_state
        self.active = active

    def single_state(self) -> SelectionState:
        """単一モデル状態（初期状態と PairState からの戻り先）。比較手法で差し替える。"""
        return SingleState()

    # ── モデル選択（State から呼ぶ） ─────────────────────────

    def select_partner(self, frame: Frame) -> str | None:
        """基本モデルの単一モデル時エラー推定から第 2 モデルを選ぶ。"""
        base = self.active[0]
        evidence = self.estimator.one_model_evidence(frame.det(base), frame.track(base))
        estimates = {base: self._estimate(evidence, self.calibration.one_model[base])}
        return self._select(estimates)

    def select_third(self, frame: Frame) -> str | None:
        """2 モデルそれぞれのエラー推定から第 3 モデルを選ぶ。"""
        a, b = self.active
        estimates = {
            target: self._estimate(
                self.estimator.two_model_evidence(frame.det(target), frame.track(target), frame.det(other)),
                self.calibration.two_model[(target, other)],
            )
            for target, other in ((a, b), (b, a))
        }
        return self._select(estimates)

    # ── 内部処理 ─────────────────────────────────────────────

    def _process(self, detect: Callable[[str], list]) -> SelectionFrameResult:
        frame = Frame(detect, self._update_tracker)
        detections = self.state.process(self, frame)

        # 実行したモデルのトラッカーは推定に使ったかどうかに関係なく毎フレーム更新する
        for m in frame.executed:
            frame.track(m)
        self._prev_executed = set(frame.executed)
        self.history.append(frame.executed)

        return SelectionFrameResult(
            detections=detections,
            state=len(frame.executed),
            flops=sum(self.flops(m) for m in frame.executed),
            version_detections=frame.detections,
            models=frame.executed,
        )

    def _update_tracker(self, model: str, dets: list) -> list:
        if model not in self._prev_executed:
            self.trackers[model] = self._tracker_factory()
        return self.trackers[model].update(dets).predicted_boxes

    def _estimate(self, evidence: FrameEvidence, profile: FrameErrorProfile) -> ErrorEstimate:
        p = self.estimator.predict(evidence, profile)
        return ErrorEstimate(fp=p["expected_fp"], fn=p["expected_fn"])

    def _select(self, estimates: dict[str, ErrorEstimate]) -> str | None:
        if len(self.active) >= self.max_version:
            return None
        pool = [m for m in self.candidates if m not in self.active]
        if not pool:
            return None
        return select_model(estimates, pool, self.calibration.resolution, self.selection_cost)