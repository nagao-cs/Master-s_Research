import os
from pathlib import Path

from src.Evaluation.dataset import fileReader

from .base_runner import BaseRunner


class CacheRunner(BaseRunner):
    def build_detection_source(self):
        self.model_names = [
            self.cfg.model_1, 
            self.cfg.model_2, 
            self.cfg.model_3
        ]
        detection_cache = DetectionCache(self.base_dir, self.cfg.map, self.model_names)
        
        return detection_cache
    
    def execute_detection(self):
        for frame in self.frame_source:
            frame_result = self.context.process_cache(frame)
            self.recorder.record_frame_result(frame_result)
        
class DetectionCache:
    def __init__(
        self,
        base_dir: Path,
        map_name: str,
        model_names: list[str]
    ):
        self.model_names = model_names
        self.cache = self._load(
            base_dir,
            map_name,
            model_names
        )

    def _load(
        self,
        base_dir,
        map_name,
        model_names
    ):
        # フレームごとのリスト構造を作成
        det_base_dir = (
            base_dir
            / "oneVersionDetectionResult"
            / "labels"
            / map_name
        )

        # 各モデルのファイル一覧を読み込む
        model_files = {}
        for model_name in model_names:
            model_dir = det_base_dir / model_name
            files = sorted(
                f for f in os.listdir(model_dir)
                if f.endswith(".txt")
            )
            model_files[model_name] = files

        # フレームごとにまとめる
        num_frames = len(model_files[model_names[0]])
        cache = []
        
        for frame_idx in range(num_frames):
            frame_detections = {}
            
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
            
            cache.append(frame_detections)

        return cache

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)