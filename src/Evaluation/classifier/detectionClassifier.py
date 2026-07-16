from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox, ClassifiedBoundingBox, ClassifyCategory


class DetectionClassifier:
    def __init__(self, iouThreshold: float):
        self.iouThreshold = iouThreshold

    def classify(self, groundTruthBoundingBoxList: list[GroundTruthBoundingBox], detectionBoundingBoxList: list[DetectionBoundingBox]) -> list[ClassifiedBoundingBox]:
        classifiedBoundingBoxList: list[ClassifiedBoundingBox] = list()

        iouList: list[tuple[int, int, float]] = list()

        for groundTruthIndex in range(len(groundTruthBoundingBoxList)):
            for detectionIndex in range(len(detectionBoundingBoxList)):
                iou = groundTruthBoundingBoxList[groundTruthIndex].computeIoU(
                    detectionBoundingBoxList[detectionIndex])
                iouList.append((groundTruthIndex, detectionIndex, iou))

        iouList.sort(key=lambda x: x[2], reverse=True)

        groundTruthIsProcessedList: list[int] = [
            False] * len(groundTruthBoundingBoxList)
        detectionIsProcessedList: list[int] = [
            False] * len(detectionBoundingBoxList)

        for groundTruthIndex, detectionIndex, iou in iouList:
            if iou < self.iouThreshold:
                break
            if groundTruthIsProcessedList[groundTruthIndex] == True:
                continue
            if detectionIsProcessedList[detectionIndex] == True:
                continue

            groundTruthIsProcessedList[groundTruthIndex] = True
            detectionIsProcessedList[detectionIndex] = True

            detectionBoundingBox = detectionBoundingBoxList[detectionIndex]

            truePositiveBoundingBox = ClassifiedBoundingBox(xCenter=detectionBoundingBox.xCenter, yCenter=detectionBoundingBox.yCenter, width=detectionBoundingBox.width, height=detectionBoundingBox.height,
                                                            classId=detectionBoundingBox.classId, confidenceScore=detectionBoundingBox.confidenceScore, classifyCategory=ClassifyCategory.TP)
            classifiedBoundingBoxList.append(truePositiveBoundingBox)

        for groundTruthIndex, groundTruthBoundingBox in enumerate(groundTruthBoundingBoxList):
            if groundTruthIsProcessedList[groundTruthIndex] == True:
                continue

            falseNegativeBoundingBox = ClassifiedBoundingBox(groundTruthBoundingBox.xCenter, groundTruthBoundingBox.yCenter, groundTruthBoundingBox.width,
                                                             groundTruthBoundingBox.height, groundTruthBoundingBox.classId, confidenceScore=None, classifyCategory=ClassifyCategory.FN)
            classifiedBoundingBoxList.append(falseNegativeBoundingBox)

        for detectionIndex, detectionBoundingBox in enumerate(detectionBoundingBoxList):
            if detectionIsProcessedList[detectionIndex] == True:
                continue

            falsePositiveBoundingBox = ClassifiedBoundingBox(detectionBoundingBox.xCenter, detectionBoundingBox.yCenter, detectionBoundingBox.width,
                                                             detectionBoundingBox.height, detectionBoundingBox.classId, detectionBoundingBox.confidenceScore, ClassifyCategory.FP)
            classifiedBoundingBoxList.append(falsePositiveBoundingBox)

        return classifiedBoundingBoxList
