"""
cost.py
単一モデル評価の CSV から、モデル選択に使う計算コスト（実行時間）を読む。

CSV は oneVersionObjectDetection.py が系列ごとに書き出す
DATASET_DIR/single_model_detection/<dataset>/<map>/eval_result/results.csv
（timestamp,dataset,model,map,...,Execution_Time_s の形式）。
同じモデルの行が複数（複数系列・複数回の計測）あれば平均する。
argmax の比較にしか使わないので正規化はしない。
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.config import DATASET_DIR

TIME_COLUMN = "Execution_Time_s"


def execution_time_csv(dataset: str, map_name: str) -> Path:
    return DATASET_DIR / "single_model_detection" / dataset / map_name / "eval_result" / "results.csv"


def load_execution_time(dataset: str, maps: list[str]) -> dict[str, float]:
    """モデル名 → 平均実行時間 [s]。maps の系列の CSV を合わせて平均する。"""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for map_name in maps:
        path = execution_time_csv(dataset, map_name)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["dataset"] != dataset or row["map"] != map_name:
                    continue
                model = row["model"]
                sums[model] = sums.get(model, 0.0) + float(row[TIME_COLUMN])
                counts[model] = counts.get(model, 0) + 1

    if not sums:
        raise ValueError(f"dataset={dataset}, maps={maps} の実行時間がありません")
    return {m: sums[m] / counts[m] for m in sums}
