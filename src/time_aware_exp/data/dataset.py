"""
dataset.py
データソースの取得・管理を担う Dataset クラス群。
Runner からデータ取得責務を分離する。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from src.Evaluation.dataset import fileReader
from ..config.config import AdrodConfig


# ---------------------------------------------------------------------------
# 抽象基底
# ---------------------------------------------------------------------------

class BaseDataset(ABC):
    """
    Runner が反復処理するデータソースの共通インターフェース。
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError
    
    @abstractmethod
    def build_from_config():
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 実装: 画像ファイル一覧（OnlineRunner 用）
# ---------------------------------------------------------------------------

class ImageDataset(BaseDataset):
    """
    ディレクトリ内の画像ファイルパス一覧を提供する Dataset。

    Parameters
    ----------
    image_dir : Path
        画像ファイルが格納されたディレクトリ。
    """
    def __init__(
        self,
        image_dir: Path,
        gt_dir: Path
    ) -> None:
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {image_dir}")

        self._paths: list[Path] = sorted(
            image_dir / name
            for name in os.listdir(image_dir)
        )
        self.gt_dir = gt_dir

        if not self._paths:
            raise FileNotFoundError(f"No image files found in: {image_dir}")

    # --- ファクトリ ---
    @classmethod
    def build_from_config(cls, cfg: AdrodConfig, base_dir: Path) -> "ImageDataset":
        """
        AdrodConfig の規約ディレクトリ構造からインスタンスを生成する。
        """
        if cfg.dataset == "CARLA":
            image_dir = base_dir / "output" / "image" / "Town02" / "original" / "front"
            gt_dir = base_dir / "output" / "label" / "Town02" / "front"
        elif cfg.dataset == "KITTI":
            image_dir = base_dir.parent.parent.parent / "d" / "data_tracking_image_2" / "training" / "image_02" / "0020"
            gt_dir = base_dir.parent.parent.parent / "d" / "data_tracking_label_2" / "training" / "label_02" / "0020"
        return cls(image_dir, gt_dir)

    # --- シーケンスプロトコル ---

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[Path]:
        return iter(self._paths)


# ---------------------------------------------------------------------------
# 実装: キャッシュ済み検出結果（CacheRunner 用）
# ---------------------------------------------------------------------------

class CachedDetectionDataset(BaseDataset):
    """
    事前に計算済みの検出結果ファイル群をフレーム単位で提供する Dataset。

    各フレームは ``{model_name: List[BoundingBox]}`` の辞書として返される。

    Parameters
    ----------
    base_dir : Path
        プロジェクトルートディレクトリ。
    map_name : str
        使用するマップ名（ディレクトリ名に使用）。
    model_names : list[str]
        ロードするモデル名のリスト。
    """

    def __init__(
        self,
        base_dir: Path,
        map_name: str,
        model_names: list[str],
    ) -> None:
        self._model_names = model_names
        self._frames: list[dict] = self._load(base_dir, map_name, model_names)

    # --- ファクトリ ---

    @classmethod
    def from_config(cls, base_dir: Path, cfg) -> "CachedDetectionDataset":
        """
        AdrodConfig からインスタンスを生成する。
        """
        model_names = [cfg.model_1, cfg.model_2, cfg.model_3]
        return cls(base_dir, cfg.map, model_names)

    # --- 内部ロード ---

    def _load(
        self,
        base_dir: Path,
        map_name: str,
        model_names: list[str],
    ) -> list[dict]:
        det_base_dir = (
            base_dir
            / "oneVersionDetectionResult"
            / "labels"
            / map_name
        )

        # 各モデルの .txt ファイル一覧をソート済みで取得
        model_files: dict[str, list[str]] = {}
        for model_name in model_names:
            model_dir = det_base_dir / model_name
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Detection cache directory not found: {model_dir}"
                )
            model_files[model_name] = sorted(
                f for f in os.listdir(model_dir) if f.endswith(".txt")
            )

        # 全モデルのフレーム数が一致することを検証
        frame_counts = {m: len(files) for m, files in model_files.items()}
        if len(set(frame_counts.values())) != 1:
            raise ValueError(
                f"Frame count mismatch across models: {frame_counts}"
            )

        num_frames = frame_counts[model_names[0]]

        # フレームごとに {model_name: BoundingBoxList} の辞書を構築
        frames: list[dict] = []
        for frame_idx in range(num_frames):
            frame_detections: dict = {}
            for model_name in model_names:
                file_path = (
                    det_base_dir
                    / model_name
                    / model_files[model_name][frame_idx]
                )
                frame_detections[model_name] = (
                    fileReader.convertDetectionFileToBoundingBoxList(
                        str(file_path)
                    )
                )
            frames.append(frame_detections)

        return frames

    # --- シーケンスプロトコル ---

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._frames)