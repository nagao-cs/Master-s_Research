"""
calibrate.py
候補モデルの事前計測を行い calibration/<calibration.name>.json を作る。
出力は config ディレクトリではなく共有の calibration/ に置くので、
同じ name を指定した config どうしで使い回せる。

各キャリブレーション系列で全モデルの検出とトラッキングを一度だけ実行し、次を求める。
- 単一モデル時のエラー推定プロファイル（対象モデルごと）
- 2 モデル時のエラー推定プロファイル（(対象, 相手) ごと）
- 解消率 R_FP, R_FN（(対象, 候補) ごと）
  対象モデルの FP / FN のうち、対象+候補 を統合器で統合した結果に残らなかった割合。
  統合器とトラッカーは実行時と同じ config.yaml の設定で作る。
件数はすべての系列で合算してから割合にする。
対象モデルは selection の base_model と candidates、系列は calibration.maps。

    python -m src.error_estimate_exp.calibrate --cfg_name default
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from src.Evaluation.dataset import fileReader
from src.error_estimate.estimation.frame_metrics import safe_div
from src.error_estimate.evaluation import GroundTruthEvaluator, match_prediction_to_gt
from src.error_estimate.experiments.frame_type_error_prediction import fit_profile
from src.time_aware_exp.factory.context_factory import build_integrator
from src.util.dataset import build_dataset

from .calibration import Calibration, Pair
from .config import config_path, load_config
from .factory import build_detector, build_estimator, build_tracker
from .selection import Resolution

MODEL_TYPE = "A"


@dataclass
class FrameRecord:
    gt: list
    dets: dict[str, list]
    tracks: dict[str, list]
    actual: dict[str, dict]  # モデル → {"fp": [...], "fn": [...]}


@dataclass
class _Count:
    n_fp: int = 0
    res_fp: int = 0
    n_fn: int = 0
    res_fn: int = 0

    def rate(self) -> Resolution:
        return Resolution(fp=safe_div(self.res_fp, self.n_fp), fn=safe_div(self.res_fn, self.n_fn))


def _match_count(a: list, b: list, iou: float) -> int:
    return len(match_prediction_to_gt(a, b, iou))


def _load_gt(label_dir: Path, image_path: Path) -> list:
    gt_path = label_dir / f"{image_path.stem}.txt"
    if not gt_path.exists():
        return []
    return fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))


def collect_map(detectors: dict, dataset_name: str, map_name: str, gt_evaluator, tracker_cfg) -> list[FrameRecord]:
    """1 系列で全モデルの検出とトラッキングを行う。トラッカーは系列ごとに作り直す。"""
    dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
    trackers = {m: build_tracker(tracker_cfg) for m in detectors}
    records = []
    for image_path, _ in tqdm(dataset, desc=map_name):
        gt = _load_gt(dataset.label_dir, image_path)
        dets = {m: d.predict(image_path) for m, d in detectors.items()}
        tracks = {m: trackers[m].update(dets[m]).predicted_boxes for m in detectors}
        actual = {m: gt_evaluator.classify(gt, dets[m]) for m in detectors}
        records.append(FrameRecord(gt, dets, tracks, actual))
    return records


def _hidden_fn(gt_evaluator, gt: list, sources: list) -> int:
    return len(gt_evaluator.classify(gt, sources)["fn"])


def fit_one_model(records, model, estimator, gt_evaluator, iou):
    items = [{
        "evidence": estimator.one_model_evidence(r.dets[model], r.tracks[model]),
        "actual": r.actual[model],
        "hidden_fn": _hidden_fn(gt_evaluator, r.gt, [*r.dets[model], *r.tracks[model]]),
    } for r in records]
    return fit_profile(items, MODEL_TYPE, iou)


def fit_two_model(records, target, other, estimator, gt_evaluator, iou):
    items = [{
        "evidence": estimator.two_model_evidence(r.dets[target], r.tracks[target], r.dets[other]),
        "actual": r.actual[target],
        "hidden_fn": _hidden_fn(
            gt_evaluator, r.gt,
            [*r.dets[target], *r.dets[other], *r.tracks[target], *r.tracks[other]],
        ),
    } for r in records]
    return fit_profile(items, MODEL_TYPE, iou)


def measure_resolution(records, a, b, integrator, gt_evaluator, iou) -> dict[Pair, _Count]:
    """a+b の統合結果で、a と b それぞれの FP / FN がどれだけ消えたかを数える。"""
    counts = {(a, b): _Count(), (b, a): _Count()}
    for r in records:
        fused, _ = integrator({a: r.dets[a], b: r.dets[b]})
        fused_actual = gt_evaluator.classify(r.gt, fused)
        for (target, _), c in counts.items():
            fp, fn = r.actual[target]["fp"], r.actual[target]["fn"]
            # 統合後にも同じ FP / FN が残っていなければ解消とみなす
            c.n_fp += len(fp)
            c.res_fp += len(fp) - _match_count(fp, fused_actual["fp"], iou)
            c.n_fn += len(fn)
            c.res_fn += len(fn) - _match_count(fn, fused_actual["fn"], iou)
    return counts


def main():
    p = ArgumentParser(description="Calibrate error profiles and resolution rates")
    p.add_argument("--cfg_name", default="default")
    args = p.parse_args()

    cfg = load_config(config_path(args.cfg_name))
    cal_cfg = cfg.calibration
    iou = cfg.iou_threshold
    integrator = build_integrator(cfg)
    gt_evaluator = GroundTruthEvaluator(iou)
    estimator = build_estimator(cal_cfg.low_confidence_threshold)
    models = list(cfg.selection.models)

    detectors = {m: build_detector(m, cfg.dataset) for m in models}
    records = [r for map_name in cal_cfg.maps
               for r in collect_map(detectors, cfg.dataset, map_name, gt_evaluator, cfg.tracker)]
    print(f"Calibration frames: {len(records)}")

    one_model = {m: fit_one_model(records, m, estimator, gt_evaluator, iou) for m in models}
    two_model = {
        (t, o): fit_two_model(records, t, o, estimator, gt_evaluator, iou)
        for t in models for o in models if t != o
    }
    resolution = {}
    for a, b in combinations(models, 2):
        for (target, cand), c in measure_resolution(records, a, b, integrator, gt_evaluator, iou).items():
            resolution[(target, cand)] = c.rate()
            print(f"R({target}, +{cand})  FP {c.res_fp}/{c.n_fp}  FN {c.res_fn}/{c.n_fn}")

    out = cal_cfg.path
    out.parent.mkdir(parents=True, exist_ok=True)
    Calibration(
        dataset=cfg.dataset,
        fit_maps=list(cal_cfg.maps),
        low_confidence_threshold=cal_cfg.low_confidence_threshold,
        one_model=one_model,
        two_model=two_model,
        resolution=resolution,
    ).save(out)
    print(f"Calibration saved to {out}")


if __name__ == "__main__":
    main()
