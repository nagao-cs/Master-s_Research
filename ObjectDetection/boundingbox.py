from typing import Optional


class BoundingBox:
    def __init__(self, xCenter: float, yCenter: float, width: float, height: float, classId: int, label: str, confidenceScore: Optional[float] = None):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height
        self.classId = classId
        self.label = label
        self.confidenceScore = confidenceScore

    def computeArea(self) -> float:
        """
        @param self: 面積を計算するbbox
        @return 面積
        """
        area = self.width * self.height
        return area

    def computeIou(self, boundingbox: 'BoundingBox') -> float:
        """
        @param boundingbox: このbboxとのiouを計算するbbox
        @return: iou
        """
        if self.classId != boundingbox.classId:
            return 0.0

        aXMin = self.xCenter - self.width
        aXMax = self.xCenter + self.width
        aYMin = self.yCenter - self.height
        aYMax = self.yCenter + self.height

        bXMin = boundingbox.xCenter - boundingbox.width
        bXMax = boundingbox.xCenter + boundingbox.width
        bYMin = boundingbox.yCenter - boundingbox.height
        bYMax = boundingbox.yCenter + boundingbox.height

        interXMin = max(aXMin, bXMin)
        interYMin = max(aYMin, bYMin)
        interXMax = min(aXMax, bXMax)
        interYMax = min(aYMax, bYMax)

        if (interXMin >= interXMax) or (interYMin >= interYMax):
            return 0.0

        intersection_area = (interXMax - interXMin) * \
            (interYMax - interYMin)
        union_area = self.compute_area() + boundingbox.compute_area() - intersection_area

        iou = intersection_area / union_area

        return iou
