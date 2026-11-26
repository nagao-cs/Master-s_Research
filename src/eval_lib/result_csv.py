"""
result_csv.py
評価結果を CSV に書くための共通処理。

- データセットごとの評価対象クラスとクラス名
- 評価対象クラスすべての AP 列
- 既存の results.csv に列を追加しながら 1 行追記する
  （古いヘッダーのファイルでも、列名で読む側 cost.py などが壊れないように書き直す）
"""
from __future__ import annotations

import csv
from pathlib import Path

from .evaluation_result import EvaluationResult

# AP 列名に使うクラス名（ない ID は番号のまま）
CLASS_NAMES = {
    "KITTI": {
        0: "car", 1: "van", 2: "truck", 3: "pedestrian",
        4: "person_sitting", 5: "cyclist", 6: "tram", 7: "misc",
    },
    "CARLA": {0: "pedestrian", 2: "vehicle", 9: "traffic_light", 11: "traffic_sign"},
}


def target_class_ids(dataset: str) -> list[int]:
    """評価対象のクラス ID。"""
    if dataset == "KITTI":
        return list(range(8))
    return [0, 2, 9, 11]


def ap_values(dataset: str, result: EvaluationResult) -> dict[str, float]:
    """{"AP_<クラス名>": AP}。GT が 1 つもないクラス（computeAP が -1）は nan。"""
    names = CLASS_NAMES.get(dataset, {})
    out = {}
    for c in target_class_ids(dataset):
        ap = result.class_ap_dict.get(c, -1.0)
        out[f"AP_{names.get(c, c)}"] = float("nan") if ap < 0 else ap
    return out


def append_row(csv_path: Path, row: dict) -> None:
    """
    row を csv_path に 1 行追記する。
    既存ファイルのヘッダーに row の列がなければ、列を足してファイル全体を書き直す
    （既存の行の新しい列は空欄）。列の順は既存の列 → 新しい列。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    header: list[str] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            rows = list(reader)

    new_cols = [k for k in row if k not in header]
    if header and not new_cols:
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
        return

    header = header + new_cols
    with csv_path.open("w", newline="") as f:
        # 列数がヘッダーと合わない古い行の余り（キー None）は捨てる
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)
