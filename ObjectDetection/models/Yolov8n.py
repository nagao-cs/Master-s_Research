from ultralytics import YOLO
import cv2
from .AbstractObjectDetector import AbstractObjectDetector
from ..utils import utils
from boundingBox.boundingBox import DetectionBoundingBox


class Yolov8nDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = YOLO("yolov8n.pt")
            print(f"YOLOv8n model loaded")

        except Exception as e:
            raise RuntimeError(f"Error loading yolov8n model: {e}")

    def predict(self, imagePath):
        image = cv2.imread(imagePath)
        if image is None:
            raise FileExistsError(f"Could not read image: {imagePath}")

        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        detections = self.model.predict(image, device="cuda")
        rawBoundingBoxList = detections[0].boxes

        outputBoundingBoxList = list()
        for rawBoundingBox in rawBoundingBoxList:
            if rawBoundingBox.conf < utils.CONF_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = rawBoundingBox.xyxy[0].tolist()
            size = (xmax - xmin) * (ymax - ymin)
            if size < utils.SIZE_THRESHOLD:
                continue

            xmin, xmax, ymin, ymax = xmin/imageWidth, xmax / \
                imageWidth, ymin/imageHeight, ymax/imageHeight
            xCenter = (xmin + xmax) / 2
            yCenter = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            classId = int(rawBoundingBox.cls[0])
            confidencescore = rawBoundingBox.conf[0].item()
            label = self.model.names[classId] if classId < len(
                self.model.names) else 'unknown'

            boundingBoxInstance = DetectionBoundingBox(
                xCenter, yCenter, width, height, classId, label, confidencescore)
            outputBoundingBoxList.append(boundingBoxInstance)

        return outputBoundingBoxList
