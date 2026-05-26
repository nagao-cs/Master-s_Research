import os
from .base_runner import BaseRunner

from ..factory.detection_factory import (
    build_model_detection
)


class OnlineRunner(BaseRunner):
    def build_detection_source(self):
        input_image_dir = self.base_dir / "output" / "image" / self.cfg.map / "original" / "front"
        
        if not input_image_dir.exists():
            raise FileNotFoundError(F"directory does not exits: {input_image_dir}")
        
        input_image_paths = [
            input_image_dir / input_image_name 
            for input_image_name in os.listdir(input_image_dir)
        ]
        
        return input_image_paths
    
    def execute_detection(self):
        self.context.ready_model()
        for input_image_path in self.frame_source:
            frame_result = self.context.process_image(input_image_path)
            self.recorder.record_frame_result(frame_result)