from .base_runner import BaseRunner
from ..data.dataset import CachedDetectionDataset


class CacheRunner(BaseRunner):
    def build_dataset(self) -> CachedDetectionDataset:
        return CachedDetectionDataset.from_config(self.base_dir, self.cfg)

    def execute_detection(self):
        for frame in self.dataset:
            frame_result = self.context.process_cache(frame)
            self.recorder.record_frame_result(frame_result)