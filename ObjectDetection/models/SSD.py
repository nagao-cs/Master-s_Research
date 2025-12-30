import tensorflow_hub as hub
import cv2
import numpy as np
from .AbstractObjectDetector import AbstractObjectDetector
import tensorflow as tf
from ..utils import utils
from ..boundingbox.boundingBox import BoundingBox


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
            raise RuntimeError(f"Error loading SSD model: {e}")

    def predict(self, imagePath):
        image = cv2.imread(imagePath)
        if image is None:
            raise FileExistsError(f"Could not read image: {imagePath}")

        RGBImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ResizedImage = cv2.resize(RGBImage, (320, 320))
        image = np.array(ResizedImage, dtype=np.uint8)
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

        outputBoundingBoxList = list()
        for i in range(num_detections):
            ymin, xmin, ymax, xmax = bboxes[i]
            xCenter = (xmin + xmax) / 2
            yCenter = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            size = width * height * (800 * 600)
            if size < utils.SIZE_THRESHOLD:
                continue

            classId = classes[i] - 1
            confidenceScore = scores[i]
            label = utils.COCO_LABELS.get(classId, 'unknown')

            boundingBoxInstance = BoundingBox(
                xCenter, yCenter, width, height, classId, label, confidenceScore)
            # label = class_id
            outputBoundingBoxList.append(boundingBoxInstance)
        return outputBoundingBoxList
