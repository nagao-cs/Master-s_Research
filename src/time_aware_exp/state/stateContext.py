"""
stateContext.py
State パターンにおける Context クラス。

【設計方針】
- DetectionStrategy を注入することで、キャッシュ・モデル推論を切り替え可能にする。
- State は ctx.detect(model, frame_ref) を呼ぶだけで、
  取得方法（キャッシュ or 推論）を意識しなくてよい。
- process(frame_ref) が唯一の外部エントリポイント。
  呼び出し元はフレームインデックスか画像を渡すだけでよい。
"""
from __future__ import annotations

from pathlib import Path

from src.boundingBox.boundingBox import DetectionBoundingBox
from ..config.config import ThresholdConfig
from ..strategy.detectionStrategy import DetectionStrategy
from ..tracker.tracker import SortTracker
from ...boundingBox.integrator.integrator import Integrator

from .AdrodState import AdrodState, FrameResult


class StateContext:
    """
    AdROD の State パターン Context。

    Args:
        thresholds:    遷移判定に使う閾値群
        uncertainty:   SOLO→PAIR の不確実性メトリクス（Strategy）
        integrator:    複数モデルの検出結果を統合する callable
        m1, m2, m3:    モデル名（strategy へのキーと対応）
        initial_state: 起動時の状態オブジェクト（通常 SoloState()）
    """

    def __init__(
        self,
        thresholds:    ThresholdConfig,
        integrator:    Integrator,
        tracker:       SortTracker,
        m1: str,
        m2: str,
        m3: str,
        initial_state: AdrodState
    ) -> None:
        self.thresholds  = thresholds
        self.integrator  = integrator
        self.tracker = tracker
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.state: AdrodState = initial_state
        
        self.models = {}

    # ── 外部エントリポイント ──────────────────────────────────

    def process_cache(self, frame):
        return self.state.process_cache(self, frame)
        
    def process_image(self, input_image_path: Path) -> FrameResult:
        """
        1フレームを処理して FrameResult を返す。
        """
        return self.state.exe_detection(self, input_image_path)

    def ready_model(self):
        """必要なモデルを起動しておく"""
        from ..factory.detection_factory import build_single_model
        
        for model_name in [self.m1, self.m2, self.m3]:
            if model_name not in self.models:
                self.models[model_name] = build_single_model(model_name)
                print(f"✓ Model '{model_name}' loaded")
                
    # ── 状態遷移（State から呼ぶ） ───────────────────────────

    def transition(self, next_state: AdrodState) -> None:
        """現在の状態を次の状態オブジェクトに切り替える。"""
        self.state = next_state

    # ── リセット ─────────────────────────────────────────────

    def reset(self) -> None:
        """シーン切り替え時などに状態と不確実性メトリクスをリセットする。"""
        from .AdrodState import SingleState
        self.state = SingleState()
        if hasattr(self.uncertainty, "reset"):
            self.uncertainty.reset()