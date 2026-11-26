"""
runner.py
time_aware_exp の Runner に、エラー推定でモデルを選ぶ Context を差し込む。
評価・結果保存は BaseRunner のものを使い、フレームごとの実行モデルだけ追加で保存する。
"""
from __future__ import annotations

import csv

from src.time_aware_exp.data.dataset import ImageDataset
from src.time_aware_exp.runner.online_runner import OnlineRunner
from src.util.dataset import build_dataset

from .factory import build_context


class _SelectionMixin:
    preload: bool

    def build_dataset(self) -> ImageDataset:
        # calibrate.py と同じデータソースから test_map の系列を読む
        dataset = build_dataset(dataset_name=self.cfg.dataset, map_name=self.cfg.test_map)
        return ImageDataset(dataset.image_dir, dataset.label_dir)

    def build_context(self):
        return build_context(self.cfg, preload=self.preload)

    def save_result(self):
        super().save_result()
        path = self.cfg_path.parent / "selected_models.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "models"])
            for idx, models in enumerate(self.context.history):
                writer.writerow([idx, "+".join(models)])
        print(f"Selected models saved to {path}")


class OnlineSelectionRunner(_SelectionMixin, OnlineRunner):
    preload = True
