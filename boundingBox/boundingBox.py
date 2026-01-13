from abc import ABC
from enum import Enum
from typing import Optional


class BoundingBox(ABC):
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


class DetectionBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, confidenceScore: float):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height
        self.classId = classId
        self.confidenceScore = confidenceScore


class GroundTruthBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int):
        self.xCenter: float = xCenter
        self.yCenter: float = yCenter
        self.width: float = width
        self.height: float = height
        self.classId: int = classId


class ClassifyCategory(Enum):
    TP = 1
    FP = 2
    FN = 3


class ClassifiedBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, confidenceScore: Optional[float], classifyCategory: ClassifyCategory):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height
        self.classId = classId
        if confidenceScore == None:
            self.confidenceScore = 0.0
        else:
            self.confidenceScore = confidenceScore
        self.classifyCategory = classifyCategory
