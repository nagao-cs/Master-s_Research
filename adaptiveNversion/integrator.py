from typing import Dict, List, Any, Tuple, Optional
from math import ceil
from boundingBox.boundingBox import DetectionBoundingBox


class MajorityIntegrator:
    def __init__(self, iouThreshold: float, maxVersion: int):
        self.iouThreshold = iouThreshold
        self.majorityThreshold = ceil(maxVersion/2)

    def _findBestMatchBoundingBoxIndex(self, targetBoundingBox: DetectionBoundingBox, subjectBoundingBoxList: list[DetectionBoundingBox], subjectBoundingBoxIsProcessedList: list[bool]) -> Optional[int]:
        """
        _findBestMatchBoundingBoxIndex の Docstring

        :param self: 説明
        :param targetBoundingBox: 説明
        :type targetBoundingBox: BoundingBox
        :param subjectBoundingBoxList: 説明
        :type subjectBoundingBoxList: list[BoundingBox]
        :param subjectBoundingBoxIsProcessedList: 説明
        :type subjectBoundingBoxIsProcessedList: list[bool]
        :return: 説明
        :rtype: int | None
        """
        bestIou = 0.0
        bestMatchBoundingBox: Optional[DetectionBoundingBox] = None
        bestMatchBoundingBoxIndex: int = -1
        for index, subjectBoundingBox in enumerate(subjectBoundingBoxList):
            if subjectBoundingBoxIsProcessedList[index] == True:
                continue
            # classIdが違うならスキップ
            if targetBoundingBox.classId != subjectBoundingBox.classId:
                continue

            currentIou = targetBoundingBox.computeIoU(subjectBoundingBox)
            if (currentIou > self.iouThreshold) and (currentIou > bestIou):
                bestIou = currentIou
                bestMatchBoundingBox = targetBoundingBox
                bestMatchBoundingBoxIndex = index

        return bestMatchBoundingBoxIndex

    def groupingDetections(self, detectionModelDict: dict[object, list[DetectionBoundingBox]]) -> list[list[DetectionBoundingBox]]:
        """
        groupingDetections の Docstring

        :param self: 説明
        :param detectionModelDict: 説明
        :type detectionModelDict: dict[object, list[BoundingBox]]
        :return: 説明
        :rtype: list[list[BoundingBox]]
        """
        groupedBoundingBoxList = list()
        NO_MATCH_INDEX = -1
        isProcessedListDict: dict[list[bool]] = {detector: [
            False] * len(detections) for detector, detections in detectionModelDict.items()}

        for targetDetector, targetBoundingBoxList in detectionModelDict.items():
            for targetBoundingBoxIndex, targetBoundingBox in enumerate(targetBoundingBoxList):
                if isProcessedListDict[targetDetector][targetBoundingBoxIndex] == True:
                    continue

                groupedBoundingBox: list[DetectionBoundingBox] = [
                    targetBoundingBox]
                isProcessedListDict[targetDetector][targetBoundingBoxIndex] = True
                for subjectDetector, subjectBoundingBoxList in detectionModelDict.items():
                    if targetDetector == subjectDetector:
                        continue

                    bestMatchBoundingBoxIndex = self._findBestMatchBoundingBoxIndex(
                        targetBoundingBox, subjectBoundingBoxList, isProcessedListDict[subjectDetector])
                    if bestMatchBoundingBoxIndex == NO_MATCH_INDEX:
                        continue

                    isProcessedListDict[subjectDetector][bestMatchBoundingBoxIndex] = True
                    matchedBoundingBox = subjectBoundingBoxList[bestMatchBoundingBoxIndex]
                    groupedBoundingBox.append(matchedBoundingBox)

                groupedBoundingBoxList.append(groupedBoundingBox)
        return groupedBoundingBoxList

    def _averageBoundingBox(self, boungingBoxList: list[DetectionBoundingBox]) -> DetectionBoundingBox:
        """
        _averageBoundingBox の Docstring

        :param self: 説明
        :param boungingBoxList: 説明
        :type boungingBoxList: list[BoundingBox]
        :return: 説明
        :rtype: BoundingBox
        """
        numMatchedBoundingBox = len(boungingBoxList)

        sumXCenter = 0.0
        sumYCenter = 0.0
        sumWidth = 0.0
        sumHeight = 0.0
        sumConfidenceScore = 0.0

        for boundingBox in boungingBoxList:
            sumXCenter += boundingBox.xCenter
            sumYCenter += boundingBox.yCenter
            sumWidth += boundingBox.width
            sumHeight += boundingBox.height
            sumConfidenceScore += boundingBox.confidenceScore

        averageXCenter = sumXCenter / numMatchedBoundingBox
        averageYCenter = sumYCenter / numMatchedBoundingBox
        averageWidth = sumWidth / numMatchedBoundingBox
        averageHeight = sumHeight / numMatchedBoundingBox
        averageConfidenceScore = sumConfidenceScore / numMatchedBoundingBox
        classId = boungingBoxList[0].classId
        label = boungingBoxList[0].label

        averagedBoundingBox = DetectionBoundingBox(
            averageXCenter, averageYCenter, averageWidth, averageHeight, classId, label, averageConfidenceScore)

        return averagedBoundingBox

    def integrate(self, groupedBoundingBoxList) -> list[DetectionBoundingBox]:
        """
        integrate の Docstring

        :param self: 説明
        :param groupedBoundingBoxList: 説明
        :return: 説明
        :rtype: list[BoundingBox]
        """
        integratedBoundingBoxList: list[DetectionBoundingBox] = list()

        for groupedBoundingBox in groupedBoundingBoxList:
            if len(groupedBoundingBox) < self.majorityThreshold:
                continue

            averagedBoundingBox: DetectionBoundingBox = self._averageBoundingBox(
                groupedBoundingBox)
            integratedBoundingBoxList.append(averagedBoundingBox)

        return integratedBoundingBoxList

    def __call__(self, detectionModelDict: dict[object, list[DetectionBoundingBox]]) -> tuple[list[DetectionBoundingBox], list[list[DetectionBoundingBox]]]:
        # groupingDetections() で前処理してから integrate() を呼ぶ
        groupedBoundingBoxList: list[list[DetectionBoundingBox]
                                     ] = self.groupingDetections(detectionModelDict)
        integratedBoundingBoxList: list[DetectionBoundingBox] = self.integrate(
            groupedBoundingBoxList)
        return integratedBoundingBoxList, groupedBoundingBoxList
