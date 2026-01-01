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

        aXMin = self.xCenter - self.width / 2
        aXMax = self.xCenter + self.width / 2
        aYMin = self.yCenter - self.height / 2
        aYMax = self.yCenter + self.height / 2

        bXMin = boundingBox.xCenter - boundingBox.width / 2
        bXMax = boundingBox.xCenter + boundingBox.width / 2
        bYMin = boundingBox.yCenter - boundingBox.height / 2
        bYMax = boundingBox.yCenter + boundingBox.height / 2

        interXMin = max(aXMin, bXMin)
        interYMin = max(aYMin, bYMin)
        interXMax = min(aXMax, bXMax)
        interYMax = min(aYMax, bYMax)

        if (interXMin >= interXMax) or (interYMin >= interYMax):
            return 0.0

        intersectionArea = (interXMax - interXMin) * \
            (interYMax - interYMin)
        unionArea = self.computeArea() + boundingBox.computeArea() - intersectionArea

        iou = intersectionArea / unionArea

        return iou


class DetectionBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, label: str, confidenceScore: float):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height
        self.classId = classId
        self.label = label
        self.confidenceScore = confidenceScore


class GroundTruthBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, label: str):
        self.xCenter: float = xCenter
        self.yCenter: float = yCenter
        self.width: float = width
        self.height: float = height
        self.classId: int = classId
        self.label: str = label


class ClassifyCategory(Enum):
    TP = 1
    FP = 2
    FN = 3


class ClassifiedBoundingBox(BoundingBox):
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, label: str, confidenceScore: Optional[float], classifyCategory: ClassifyCategory):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height
        self.classId = classId
        self.label = label
        if confidenceScore == None:
            self.confidenceScore = 0.0
        else:
            self.confidenceScore = confidenceScore
        self.classifyCategory = classifyCategory
