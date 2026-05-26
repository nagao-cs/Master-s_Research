"""
adrodState.py
AdROD の State パターン実装。

各 State は ctx.detect(model, frame_ref) を呼ぶだけで、
キャッシュ・モデル推論の違いを意識しない。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.ObjectDetection.models.FLOPsDict import FLOPs_Dict

if TYPE_CHECKING:
    from .stateContext import StateContext


# ── フレーム処理結果 ─────────────────────────────────────────

@dataclass
class FrameResult:
    detections: list[DetectionBoundingBox]
    state:      int # num_version
    flops:      float


# ── State 基底クラス ─────────────────────────────────────────

class AdrodState(ABC):
    @abstractmethod
    def process_cache(self, ctx: StateContext, frame: dict) -> FrameResult:
        pass
        
    @abstractmethod
    def exe_detection(self, ctx: StateContext, frame_ref: Any) -> FrameResult:
        pass
    
    def calc_jaccard(self, bbox_groups: list[list[DetectionBoundingBox]]) -> float:
        if not bbox_groups:
            return 1.0
        agreed = sum(1 for g in bbox_groups if len(g) == 2)
        return agreed / len(bbox_groups)


# ── SOLO 状態 ────────────────────────────────────────────────

class SingleState(AdrodState):
    """
    M1 のみで検出する。
    不確実性が theta_track を超えたら同フレーム内で TwoState へ遷移する。
    """
    def process_cache(self, ctx, frame):
        det_m1 = frame[ctx.m1]
        track_result = ctx.tracker.update(det_m1)
        det_dict = {ctx.m1: det_m1, "track": track_result}
        result, bbox_groups = ctx.integrator(det_dict)
        uncertainy: float = self.calc_jaccard(bbox_groups)
        
        if uncertainy <= ctx.thresholds.theta_track:
            ctx.transition(TwoState())
            return ctx.state.process_cache(ctx, frame)

        return FrameResult(
            detections=det_m1,
            state=1,
            flops=FLOPs_Dict[ctx.m1],
        )

    def exe_detection(self, ctx: StateContext, frame_ref: Any) -> FrameResult:
        dets_m1 = ctx.detect(ctx.m1, frame_ref)
        u = ctx.uncertainty.calculate(dets_m1)

        if u >= ctx.thresholds.theta_track:
            ctx.transition(TwoState())
            return ctx.state.exe_detection(ctx, frame_ref)

        return FrameResult(
            detections=dets_m1,
            state=1,
            flops=FLOPs_Dict[ctx.m1],
        )


# ── PAIR 状態 ────────────────────────────────────────────────

class TwoState(AdrodState):
    """
    M1+M2 で検出する。
    Jaccard 係数に基づいて次状態を決定する。
    """
    def process_cache(self, ctx, frame):
        dets = {
            ctx.m1: frame[ctx.m1],
            ctx.m2: frame[ctx.m2],
        }
        result, bbox_groups = ctx.integrator(dets)
        jaccard = self.calc_jaccard(bbox_groups)

        if jaccard >= ctx.thresholds.theta_high:
            ctx.transition(SingleState())
        elif jaccard <= ctx.thresholds.theta_low:
            ctx.transition(ThreeState())
            return ctx.state.process_cache(ctx, frame)

        return FrameResult(
            detections=result,
            state=2,
            flops=FLOPs_Dict[ctx.m1] + FLOPs_Dict[ctx.m2],
        )
    
    def exe_detection(self, ctx: StateContext, frame_ref: Any) -> FrameResult:
        dets = {
            ctx.m1: ctx.detect(ctx.m1, frame_ref),
            ctx.m2: ctx.detect(ctx.m2, frame_ref),
        }
        result, bbox_groups = ctx.integrator(dets)
        jaccard = self.calc_jaccard(bbox_groups)

        if jaccard >= ctx.thresholds.theta_high:
            ctx.transition(SingleState())
        elif jaccard <= ctx.thresholds.theta_low:
            ctx.transition(ThreeState())
            return ctx.state.exe_detection(ctx, frame_ref)

        return FrameResult(
            detections=result,
            state=2,
            flops=FLOPs_Dict[ctx.m1] + FLOPs_Dict[ctx.m2],
        )

# ── ENSEMBLE 状態 ────────────────────────────────────────────

class ThreeState(AdrodState):
    """
    M1+M2+M3 の三版で検出する。
    実行後は無条件で TwoState へ戻る。
    """
    def process_cache(self, ctx, frame):
        dets = {
            ctx.m1: frame[ctx.m1],
            ctx.m2: frame[ctx.m2],
            ctx.m3: frame[ctx.m3]
        }
        result, bbox_groups = ctx.integrator(dets)
        
        ctx.transition(TwoState())
        
        return FrameResult(
            detections=result,
            state=3,
            flops=FLOPs_Dict[ctx.m1] + FLOPs_Dict[ctx.m2],
        )
        
    def exe_detection(self, ctx: StateContext, frame_ref: Any) -> FrameResult:
        dets = {
            ctx.m1: ctx.detect(ctx.m1, frame_ref),
            ctx.m2: ctx.detect(ctx.m2, frame_ref),
            ctx.m3: ctx.detect(ctx.m3, frame_ref),
        }
        result, _ = ctx.integrator(dets)
        ctx.transition(TwoState())

        return FrameResult(
            detections=result,
            state=3,
            flops=sum(FLOPs_Dict[m] for m in [ctx.m1, ctx.m2, ctx.m3]),
        )