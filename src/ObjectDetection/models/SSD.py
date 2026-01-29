import tensorflow_hub as hub
import cv2
import numpy as np
from .AbstractObjectDetector import AbstractObjectDetector
import tensorflow as tf
from ..utils import utils
from boundingBox.boundingBox import DetectionBoundingBox
import torch


class SSDDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # print(f"original image: type={type(image)}, shape={image.shape}")
        imageHeight: int = image.shape[0]
        imageWidth: int = image.shape[1]

        imageTensor = tf.convert_to_tensor(image)
        imageTensor = tf.expand_dims(imageTensor, axis=0)
        # print(
        # f"image Tensor: type={type(imageTensor)}, shape={imageTensor.shape}")

        detections = self.model(imageTensor)
        # - detection_boxes: バウンディングボックスの座標 (ymin, xmin, ymax, xmax)
        # - detection_classes: 検出されたオブジェクトのクラスID
        # - detection_scores: 検出の信頼度スコア
        # - num_detections: 検出されたオブジェクトの数
        classIdList = detections["detection_classes"].numpy().astype(int)
        # 有効なクラスIDのマスクを作成
        classIdMask = classIdList in (0, 2, 9, 11)
        validIndices = torch.where(classIdMask)[0]

        yxyxList = detections["detection_boxes"][0].numpy()[validIndices]
        confidenceScoreList = detections["detection_scores"][0].numpy()[
            validIndices]
        classIdList = classIdList[validIndices]

        outputBoundingBoxList = list()
        for coodinate, confidenceScore, classId in zip(yxyxList, confidenceScoreList, classIdList):
            if confidenceScore < utils.CONF_THRESHOLD:
                continue

            ymin, xmin, ymax, xmax = coodinate
            xCenter = (xmin + xmax) / 2
            yCenter = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            size = width * imageWidth * height * imageHeight
            if size < utils.SIZE_THRESHOLD:
                continue

            classId = utils.COCO_ID_MAPPER.get(classId-1, -1)

            boundingBoxInstance = DetectionBoundingBox(
                xCenter, yCenter, width, height, classId, confidenceScore)
            outputBoundingBoxList.append(boundingBoxInstance)
        return outputBoundingBoxList
