from typing import Optional
from math import ceil

from .integrator import Integrator
from src.boundingBox.boundingBox import DetectionBoundingBox


class MajorityIntegrator(Integrator):
    def __init__(self, iouThreshold: float, maxVersion: int):
        self.iouThreshold = iouThreshold
        self.majorityThreshold = ceil(maxVersion/2)

    def _findBestMatchBoundingBoxIndex(self, targetBoundingBox: DetectionBoundingBox, subjectBoundingBoxList: list[DetectionBoundingBox], subjectBoundingBoxIsProcessedList: list[bool]) -> Optional[int]:
        """
        _findBestMatchBoundingBoxIndex の Docstring

        :param targetBoundingBox: 説明
        :param subjectBoundingBoxList: 説明
        :param subjectBoundingBoxIsProcessedList: 説明
        :return: 説明
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

    def _buildIouList(self, boundingBoxList1: list[DetectionBoundingBox], boundingBoxList2: list[DetectionBoundingBox]) -> list[tuple[int, int, float]]:
        iouList: list[tuple[int, int, float]] = list()
        for targetBoundingBoxIndex, targetBoundingBox in enumerate(boundingBoxList1):
            for subjectBoundingBoxIndex, subjectBoundingBox in enumerate(boundingBoxList2):
                iou = targetBoundingBox.computeIoU(subjectBoundingBox)
                if iou < self.iouThreshold:
                    continue
                matchTuple: tuple[int, int, float] = (
                    targetBoundingBoxIndex, subjectBoundingBoxIndex, iou)
                iouList.append(matchTuple)

        iouList.sort(key=lambda matchTuple: matchTuple[2], reverse=True)

        return iouList

    def groupingDetections(
        self,
        detectionModelDict: dict[object, list[DetectionBoundingBox]]
    ) -> list[list[DetectionBoundingBox]]:
        """
        detector 制約付き Union-Find によるグルーピング

        制約：
        ・IoU >= threshold
        ・同一 detector は同一グループに最大 1 box
        """

        # --- Step 1: box を ID 化 ---
        id_to_box: dict[int, tuple[object, int]] = {}
        box_to_id: dict[tuple[object, int], int] = {}

        box_id = 0
        for detector, boxes in detectionModelDict.items():
            for idx in range(len(boxes)):
                id_to_box[box_id] = (detector, idx)
                box_to_id[(detector, idx)] = box_id
                box_id += 1

        total_boxes = box_id

        # --- Step 2: Union-Find 初期化 ---
        parent = list(range(total_boxes))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int):
            rx = find(x)
            ry = find(y)
            if rx != ry:
                parent[rx] = ry

        # --- Step 3: 全ペア IoU 計算 ---
        matches: list[tuple[int, int, float]] = []

        detectors = list(detectionModelDict.keys())
        for i, d1 in enumerate(detectors):
            for d2 in detectors[i + 1:]:
                boxes1 = detectionModelDict[d1]
                boxes2 = detectionModelDict[d2]

                for idx1, b1 in enumerate(boxes1):
                    for idx2, b2 in enumerate(boxes2):
                        iou = b1.computeIoU(b2)
                        if iou >= self.iouThreshold:
                            id1 = box_to_id[(d1, idx1)]
                            id2 = box_to_id[(d2, idx2)]
                            matches.append((id1, id2, iou))

        # IoU 降順
        matches.sort(key=lambda x: x[2], reverse=True)

        # --- Step 4: detector 制約付き union ---
        # 各 root が含む detector 集合を管理
        root_detectors: dict[int, set[object]] = {
            i: {id_to_box[i][0]} for i in range(total_boxes)
        }

        for id1, id2, _ in matches:
            r1 = find(id1)
            r2 = find(id2)

            if r1 == r2:
                continue

            # detector が衝突するならスキップ
            if root_detectors[r1] & root_detectors[r2]:
                continue

            # union 実行
            union(r1, r2)
            new_root = find(r1)

            # detector 集合を更新
            root_detectors[new_root] = (
                root_detectors[r1] | root_detectors[r2]
            )

        # --- Step 5: グループ生成 ---
        groups: dict[int, list[DetectionBoundingBox]] = {}

        for box_id in range(total_boxes):
            root = find(box_id)
            if root not in groups:
                groups[root] = []

            detector, idx = id_to_box[box_id]
            groups[root].append(detectionModelDict[detector][idx])

        return list(groups.values())

    # def groupingDetections(self, detectionModelDict: dict[object, list[DetectionBoundingBox]]) -> list[list[DetectionBoundingBox]]:
    #     """
    #     複数モデルの検出結果をグルーピング（全体最適マッチング）

    #     処理流程：
    #     1. 全モデルペア間でIoU値を計算
    #     2. IoU値が高い順に処理
    #     3. 既にグループに含まれたボックスはスキップ
    #     4. マッチしたボックスを同じグループに統合
    #     """
    #     groupedBoundingBoxList: list[list[DetectionBoundingBox]] = list()

    #     isProcessedListDict: dict[object, list[bool]] = {detector: [
    #         False] * len(detections) for detector, detections in detectionModelDict.items()}

    #     for targetDetector, targetBoundingBoxList in detectionModelDict.items():
    #         for subjcetDetector, subjectBoundingBoxList in detectionModelDict.items():
    #             if targetDetector == subjcetDetector:
    #                 continue

    #             iouList: list[tuple[int, int, float]] = self._buildIouList(targetBoundingBoxList, subjectBoundingBoxList)
    #             for targetBoundingBoxIndex, subjcetBoundingBoxIndex, iouValue in iouList:
    #                 if isProcessedListDict[targetDetector][targetBoundingBoxIndex] == True:
    #                     continue
    #                 if isProcessedListDict[subjcetDetector][subjcetBoundingBoxIndex] == True:
    #                     continue

    # def groupingDetections(self, detectionModelDict: dict[object, list[DetectionBoundingBox]]) -> list[list[DetectionBoundingBox]]:
    #     """
    #     groupingDetections の Docstring

    #     :param self: 説明
    #     :param detectionModelDict: 説明
    #     :type detectionModelDict: dict[object, list[BoundingBox]]
    #     :return: 説明
    #     :rtype: list[list[BoundingBox]]
    #     """
    #     groupedBoundingBoxList = list()
    #     NO_MATCH_INDEX = -1
    #     isProcessedListDict: dict[list[bool]] = {detector: [
    #         False] * len(detections) for detector, detections in detectionModelDict.items()}

    #     for targetDetector, targetBoundingBoxList in detectionModelDict.items():
    #         for targetBoundingBoxIndex, targetBoundingBox in enumerate(targetBoundingBoxList):
    #             if isProcessedListDict[targetDetector][targetBoundingBoxIndex]:
    #                 continue

    #             groupedBoundingBox: list[DetectionBoundingBox] = [
    #                 targetBoundingBox]
    #             isProcessedListDict[targetDetector][targetBoundingBoxIndex] = True
    #             for subjectDetector, subjectBoundingBoxList in detectionModelDict.items():
    #                 if targetDetector == subjectDetector:
    #                     continue

    #                 bestMatchBoundingBoxIndex = self._findBestMatchBoundingBoxIndex(
    #                     targetBoundingBox, subjectBoundingBoxList, isProcessedListDict[subjectDetector])
    #                 if bestMatchBoundingBoxIndex == NO_MATCH_INDEX:
    #                     continue

    #                 isProcessedListDict[subjectDetector][bestMatchBoundingBoxIndex] = True
    #                 matchedBoundingBox = subjectBoundingBoxList[bestMatchBoundingBoxIndex]
    #                 groupedBoundingBox.append(matchedBoundingBox)

    #             groupedBoundingBoxList.append(groupedBoundingBox)
    #     return groupedBoundingBoxList

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

        averagedBoundingBox = DetectionBoundingBox(
            averageXCenter, averageYCenter, averageWidth, averageHeight, classId, averageConfidenceScore)

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
