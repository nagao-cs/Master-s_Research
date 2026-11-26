"""
factory.py
検出器・トラッカー・エラー推定器・Context の生成をまとめる。
calibrate.py と実行時で同じ検出器と推定器を使うため、生成はここに一本化する。
"""
from __future__ import annotations

from src.ObjectDetection.models.factory import build_model
from src.ObjectDetection.models.FLOPsDict import FLOPs_Dict
from src.error_estimate.config import ErrorPredictionConfig, ShapeMatcherConfig
from src.error_estimate.estimation.frame_type import FrameTypeErrorEstimator
from src.error_estimate.matching import ShapeOneToOneMatcher
from src.time_aware_exp.factory.context_factory import build_integrator
from src.tracker.tracker import SortTracker

from .calibration import Calibration
from .config import ExpConfig, TrackerConfig
from .context import SelectionContext
from .cost import load_execution_time


def build_detector(model_name: str, dataset: str):
    return build_model(model_name=model_name, dataset=dataset, device="cuda")


def build_tracker(cfg: TrackerConfig) -> SortTracker:
    return SortTracker(max_age=cfg.max_age, min_hits=cfg.min_hits, iou_threshold=cfg.iou_threshold)


def build_estimator(low_confidence_threshold: float) -> FrameTypeErrorEstimator:
    return FrameTypeErrorEstimator(
        tracker_matcher=ShapeOneToOneMatcher(ShapeMatcherConfig()),
        config=ErrorPredictionConfig(low_confidence_threshold=low_confidence_threshold),
    )


def build_context(cfg: ExpConfig, preload: bool, context_cls=SelectionContext, **extra) -> SelectionContext:
    """context_cls と extra は比較手法（src/comparison_exp）が Context を差し替えるときに使う。"""
    cal_path = cfg.calibration.path
    if not cal_path.exists():
        raise FileNotFoundError(f"{cal_path} がありません。calibrate.py を先に実行してください")
    calibration = Calibration.load(cal_path)
    if calibration.dataset != cfg.dataset:
        raise ValueError(f"キャリブレーションのデータセット {calibration.dataset} が設定 {cfg.dataset} と異なります")
    # 共有ファイルなので、この config の calibration 設定で作られたものか確かめる
    if list(calibration.fit_maps) != list(cfg.calibration.maps):
        raise ValueError(f"{cal_path} の系列 {calibration.fit_maps} が設定 {list(cfg.calibration.maps)} と異なります")
    if calibration.low_confidence_threshold != cfg.calibration.low_confidence_threshold:
        raise ValueError(
            f"{cal_path} の low_confidence_threshold {calibration.low_confidence_threshold} が"
            f"設定 {cfg.calibration.low_confidence_threshold} と異なります"
        )
    # キャリブレーションに使った系列をテストに使わない
    if cfg.test_map in calibration.fit_maps:
        raise ValueError(f"系列 {cfg.test_map} はキャリブレーションに使われています")

    exec_time = load_execution_time(cfg.dataset, list(cfg.cost.maps))
    needed = set(cfg.selection.models)
    for name, known in (("calibration", calibration.models), ("実行時間", exec_time), ("FLOPs_Dict", FLOPs_Dict)):
        missing = sorted(needed - set(known))
        if missing:
            raise ValueError(f"{name} に含まれないモデルがあります: {missing}")

    return context_cls(
        thresholds=cfg.thresholds,
        integrator=build_integrator(cfg),
        base_model=cfg.selection.base_model,
        candidates=cfg.selection.candidates,
        max_version=cfg.max_version,
        calibration=calibration,
        estimator=build_estimator(calibration.low_confidence_threshold),
        flops=FLOPs_Dict.__getitem__,
        selection_cost=exec_time.__getitem__,
        tracker_factory=lambda: build_tracker(cfg.tracker),
        detector_factory=(lambda m: build_detector(m, cfg.dataset)) if preload else None,
        **extra,
    )
