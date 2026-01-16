from abc import ABC
from enum import Enum
from typing import Optional
import cv2


class BoundingBox(ABC):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int):
        self.xCenter: float = xCenter
        self.yCenter: float = yCenter
        self.width: float = width
        self.height: float = height
        self.classId: int = classId

    def computeArea(self) -> float:
        """
        @param self: 面積を計算するbbox
        @return 面積
        """
        area = self.width * self.height
        return area

    def computeIoU(self, boundingBox: 'BoundingBox') -> float:
        if self.classId != boundingBox.classId:
            return 0.0

        thisBoxXMin = self.xCenter - self.width / 2
        thisBoxXMax = self.xCenter + self.width / 2
        thisBoxYMin = self.yCenter - self.height / 2
        thisBoxYMax = self.yCenter + self.height / 2

        targetBoxXMin = boundingBox.xCenter - boundingBox.width / 2
        targetBoxXMax = boundingBox.xCenter + boundingBox.width / 2
        targetBoxYMin = boundingBox.yCenter - boundingBox.height / 2
        targetBoxYMax = boundingBox.yCenter + boundingBox.height / 2

        interXMin = max(thisBoxXMin, targetBoxXMin)
        interYMin = max(thisBoxYMin, targetBoxYMin)
        interXMax = min(thisBoxXMax, targetBoxXMax)
        interYMax = min(thisBoxYMax, targetBoxYMax)

        if (interXMin >= interXMax) or (interYMin >= interYMax):
            return 0.0

        intersectionArea = (interXMax - interXMin) * \
            (interYMax - interYMin)
        unionArea = self.computeArea() + boundingBox.computeArea() - intersectionArea

        iou = intersectionArea / unionArea

        return iou

    def drawBoundingBoxOnImage(self, image):
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        absoluteXCenter: float = self.xCenter * imageWidth
        absoluteYCenter: float = self.yCenter * imageHeight
        absoluteWidth: float = self.width * imageWidth
        absoluteHeight: float = self.height * imageHeight

        absoluteXMin: int = int(absoluteXCenter - absoluteWidth / 2)
        absoluteXMax: int = int(absoluteXCenter + absoluteWidth / 2)
        absoluteYMin: int = int(absoluteYCenter - absoluteHeight / 2)
        absoluteYMax: int = int(absoluteYCenter + absoluteHeight / 2)

        confidenceScore: float = self.confidenceScore
        classId: int = self.classId

        cv2.rectangle(image, (absoluteXMin, absoluteYMin),
                      (absoluteXMax, absoluteYMax), (0, 255, 0), 2)
        text = f"{classId} {confidenceScore:.2f}"
        cv2.putText(image, text, (absoluteXMin, absoluteYMin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image


class DetectionBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, confidenceScore: float):
        super().__init__(xCenter, yCenter, width, height, classId)
        self.confidenceScore = confidenceScore


class GroundTruthBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int):
        super().__init__(xCenter, yCenter, width, height, classId)


class ClassifyCategory(Enum):
    TP = 1
    FP = 2
    FN = 3


class ClassifiedBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, confidenceScore: Optional[float], classifyCategory: ClassifyCategory):
        super().__init__(xCenter, yCenter, width, height, classId)
        if confidenceScore == None:
            self.confidenceScore = 0.0
        else:
            self.confidenceScore = confidenceScore
        self.classifyCategory = classifyCategory
