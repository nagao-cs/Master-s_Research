"""
factory.py
method.name に応じた Context を作る。
検出器・トラッカー・統合器・キャリブレーション・コストは提案手法（error_estimate_exp.factory）と共通。
"""
from __future__ import annotations

from pathlib import Path

from src.error_estimate_exp.config import ExpConfig
from src.error_estimate_exp.context import SelectionContext
from src.error_estimate_exp.factory import build_context
from src.eval_lib.result_csv import target_class_ids

from .architecture_contexts import ConfidenceExitContext, RecoveryBlockContext
from .config import MethodConfig
from .selector_contexts import FixedContext, OracleContext, RandomContext


def context_spec(cfg: ExpConfig, method: MethodConfig, gt_dir: Path | None, seed: int | None):
    """(Context クラス, 追加の引数)。"""
    name = method.name
    if name == "proposed":
        return SelectionContext, {}
    if name == "fixed":
        return FixedContext, {"fixed_models": method.fixed_models}
    if name == "random":
        return RandomContext, {"seed": 0 if seed is None else seed}
    if name == "oracle":
        if gt_dir is None:
            raise ValueError("oracle には正解ラベルのディレクトリが必要です")
        return OracleContext, {
            "gt_dir": gt_dir,
            "target_class_ids": target_class_ids(cfg.dataset),
            "iou_threshold": cfg.iou_threshold,
        }
    if name == "recovery_block":
        threshold = cfg.thresholds.theta_low if method.threshold is None else method.threshold
        return RecoveryBlockContext, {"primary": method.primary, "backup": method.backup, "threshold": threshold}
    if name == "confidence_exit":
        threshold = cfg.thresholds.tau_p if method.confidence_threshold is None else method.confidence_threshold
        return ConfidenceExitContext, {"order": method.order, "confidence_threshold": threshold}
    raise ValueError(f"未対応の method: {name}")


def build_comparison_context(
    cfg: ExpConfig,
    method: MethodConfig,
    preload: bool,
    gt_dir: Path | None = None,
    seed: int | None = None,
) -> SelectionContext:
    cls, kwargs = context_spec(cfg, method, gt_dir, seed)
    return build_context(cfg, preload=preload, context_cls=cls, **kwargs)
