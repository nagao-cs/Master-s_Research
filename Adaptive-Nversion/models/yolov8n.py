from ultralytics import YOLO
from detecor import Detector


class Yolov8n(Detector):
    def __init__(self):
        try:
            self.model = YOLO('yolov8n')
        except Exception as e:
            raise RuntimeError
