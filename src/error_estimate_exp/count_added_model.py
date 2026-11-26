"""
count_selected_models.py
selected_models.csv から，各モデルがどの段階（第 1・第 2・第 3 モデル）で
何フレーム動作し，何回新たに起動されたかを集計する。

  frames       その段階で動作したフレーム数
  activations  前フレームでその段階に入っていなかったモデルが入った回数

    python count_selected_models.py selected_models.csv [-o counts.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from src.config import ERROR_ESTIMATE_EXP_ROOT
import pandas as pd

STAGES = {1: "1st", 2: "2nd", 3: "3rd"}
CONFIG_ROOT = ERROR_ESTIMATE_EXP_ROOT / "config"

def to_stages(models: pd.Series) -> pd.DataFrame:
    """'yolov8n+ssd' を段階ごとの列に分ける（存在しない段階は NaN）。"""
    stages = models.str.split("+", expand=True)
    stages.columns = [STAGES.get(i + 1, f"{i + 1}th") for i in stages.columns]
    return stages


def count(stages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in stages.columns:
        cur = stages[stage]
        started = cur.notna() & (cur != cur.shift())
        frames = cur.value_counts()
        activations = cur[started].value_counts()
        for model, n in frames.items():
            rows.append({
                "stage": stage,
                "model": model,
                "frames": int(n),
                "activations": int(activations.get(model, 0)),
            })
    return pd.DataFrame(rows).sort_values(["stage", "frames"], ascending=[True, False], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="default")
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args()

    config_name = args.cfg
    csv_path = CONFIG_ROOT / config_name / "selected_models.csv"
    df = pd.read_csv(csv_path)
    table = count(to_stages(df["models"]))
    table["frame_ratio"] = (table.frames / len(df)).round(3)

    print(f"Total frames: {len(df)}")
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()