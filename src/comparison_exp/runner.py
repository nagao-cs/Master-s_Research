"""
runner.py
提案手法と同じ Runner（データセット・評価・結果保存）で比較手法の Context を動かす。
結果は提案手法と同じ形式（result.csv, selected_models.csv, result/, figure/）。
"""
from __future__ import annotations

from pathlib import Path

from src.error_estimate_exp.runner import _SelectionMixin
from src.time_aware_exp.runner.online_runner import OnlineRunner

from .config import MethodConfig
from .factory import build_comparison_context


class ComparisonRunner(_SelectionMixin, OnlineRunner):
    preload = True

    def __init__(
        self,
        cfg,
        method: MethodConfig,
        base_dir: Path,
        cfg_path: Path,
        seed: int | None = None,
        shared_models: dict | None = None,
    ) -> None:
        super().__init__(cfg, base_dir, cfg_path)
        self.method = method
        self.seed = seed
        # 複数の手法・シードを続けて実行するとき、読み込んだ検出器を使い回す
        self.shared_models = shared_models

    def build_context(self):
        ctx = build_comparison_context(
            self.cfg, self.method, preload=self.preload, gt_dir=self.dataset.gt_dir, seed=self.seed
        )
        if self.shared_models is not None:
            ctx.models = self.shared_models
        return ctx
