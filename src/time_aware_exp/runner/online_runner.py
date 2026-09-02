from tqdm import tqdm

from .base_runner import BaseRunner
from ..data.dataset import ImageDataset


class OnlineRunner(BaseRunner):
    def build_dataset(self) -> ImageDataset:
        return ImageDataset.build_from_config(self.cfg, self.base_dir)

    def execute_detection(self):
        for image_path in tqdm(iterable=self.dataset, total=len(self.dataset)):
            frame_result = self.context.process_image(image_path)
            self.recorder.record_frame_result(frame_result)