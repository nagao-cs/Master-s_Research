from .base_runner import BaseRunner

from ..factory.detection_factory import (
    build_model_detection
)

from ..strategy.frame_input import FrameInput


class VideoFrameSource:

    def __init__(self, cap):

        self.cap = cap

    def get(self, frame_idx):

        success, image = self.cap.read()

        if not success:
            raise RuntimeError("Failed to read frame")

        return FrameInput(
            frame_idx=frame_idx,
            image=image
        )


class OnlineRunner(BaseRunner):

    def build_detection(self):

        detection = build_model_detection(self.cfg)

        import cv2

        cap = cv2.VideoCapture("video.mp4")

        num_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_source = VideoFrameSource(cap)

        return detection, frame_source, num_frames