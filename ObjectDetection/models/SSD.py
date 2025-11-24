import tensorflow_hub as hub
import cv2
import numpy as np
from models.AbstractObjectDetector import AbstractObjectDetector
import os
import csv
import tensorflow as tf
from PIL import Image
import utils.utils as utils


class SSDDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = hub.load(
                "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/ssd-mobilenet-v2/1")
            print(f"SSD model loaded")
        except Exception as e:
            print(f"Error loading SSD model: {e}")
            self.model = None

    def predict(self, image_path):
        if self.model is None:
            print("Model is not loaded")
            return []
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
        image = np.array(image, dtype=np.uint8)
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        detections = self.model(image_tensor)
        # - detection_boxes: バウンディングボックスの座標 (ymin, xmin, ymax, xmax)
        # - detection_classes: 検出されたオブジェクトのクラスID
        # - detection_scores: 検出の信頼度スコア
        # - num_detections: 検出されたオブジェクトの数
        num_detections = int(detections['num_detections'][0])
        bboxes = detections['detection_boxes'][0].numpy()[:num_detections]
        classes = detections['detection_classes'][0].numpy().astype(int)[
            :num_detections]
        scores = detections['detection_scores'][0].numpy()[:num_detections]

        output = list()
        for i in range(num_detections):
            if scores[i] > 0.25:
                ymin, xmin, ymax, xmax = bboxes[i]
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                class_id = classes[i]
                conf = scores[i]
                label = f"Class {class_id}"
                output.append({
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'class_id': class_id,
                    'confidence': conf,
                    'label': label
                })
        return output
