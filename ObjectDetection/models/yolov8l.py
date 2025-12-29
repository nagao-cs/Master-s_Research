from ultralytics import YOLO
import cv2
from models.AbstractObjectDetector import AbstractObjectDetector
import utils.utils as utils
from boundingbox.boundingBox import BoundingBox


class Yolov8lDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = YOLO("yolov8l.pt")
            print(f"YOLOv8l model loaded")

        except Exception as e:
            raise RuntimeError(f"Error loading yolov8l model: {e}")

    def predict(self, imagePath):
        image = cv2.imread(imagePath)
        if image is None:
            raise FileExistsError(f"Could not read image: {imagePath}")

        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        detections = self.model.predict(image, device="cuda")
        boundingBoxList = list()
        boundingBoxList = detections[0].boxes

        outputBoundingBoxList = list()
        for boundingBox in boundingBoxList:
            if boundingBox.conf > utils.CONF_THRESHOLD:
                xmin, ymin, xmax, ymax = boundingBox.xyxy[0].tolist()
                size = (xmax - xmin) * (ymax - ymin)
                if size < utils.SIZE_THRESHOLD:
                    continue
                xmin, xmax, ymin, ymax = xmin/imageWidth, xmax / \
                    imageWidth, ymin/imageHeight, ymax/imageHeight
                xCenter = (xmin + xmax) / 2
                yCenter = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                classId = int(boundingBox.cls[0])
                confidencescore = boundingBox.conf[0].item()
                label = self.model.names[classId] if classId < len(
                    self.model.names) else 'unknown'

                boundingBoxInstance = BoundingBox(
                    xCenter, yCenter, width, height, classId, label, confidencescore)
                outputBoundingBoxList.append(boundingBoxInstance)

        return outputBoundingBoxList
