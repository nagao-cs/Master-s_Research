from ultralytics import YOLO
import cv2
import numpy as np
import csv
from models.AbstractObjectDetector import AbstractObjectDetector
import os
import utils.utils as utils


class Yolo11nDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = YOLO(
                r"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\yolo11n.pt")
            print(f"YOLOv11 model loaded")

        except Exception as e:
            print(f"Error loading YOLOv11n model: {e}")
            self.model = None
            raise e

    def predict(self, image):
        if self.model is None:
            print("Model is not loaded")
            return []
        image = cv2.imread(image)
        if image is None:
            print(f"Could not read image: {image}")
            return []
        im_widht = image.shape[1]
        im_height = image.shape[0]
        detections = self.model(image)
        bboxes = list()
        bboxes = detections[0].boxes
        output = list()
        for bbox in bboxes:
            if bbox.conf > utils.CONF_THRESHOLD:
                xmin, ymin, xmax, ymax = bbox.xyxy[0].tolist()
                size = (xmax - xmin) * (ymax - ymin)
                if size < utils.SIZE_THRESHOLD:
                    continue
                xmin, xmax, ymin, ymax = xmin/im_widht, xmax / \
                    im_widht, ymin/im_height, ymax/im_height
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                class_id = int(bbox.cls[0])
                conf = bbox.conf[0].item()
                label = self.model.names[class_id] if class_id < len(
                    self.model.names) else 'unknown'
                output.append({
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': conf,
                    'label': label
                })
        return output
