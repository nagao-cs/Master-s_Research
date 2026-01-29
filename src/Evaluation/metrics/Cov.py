import os
from pathlib import Path
from collections.abc import Generator

from ..dataset import fileReader
from src.boundingBox.boundingBox import GroundTruthBoundingBox, DetectionBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.boundingBox.averagingBondingBox import averageBoundingBox

# -----------
# 型エイリアス
# ----------
FrameData = tuple[list[GroundTruthBoundingBox],
                  list[list[DetectionBoundingBox]]]


def frameDataGenerator(gtDatasetDirPath: Path, detDatasetDirPathList: list[Path]) -> Generator[FrameData]:
    for frameFile in os.listdir(gtDatasetDirPath):
        detFrameData: list[list[DetectionBoundingBox]] = []
        gtFilePath: Path = gtDatasetDirPath / frameFile
        gtBBoxList: list[GroundTruthBoundingBox] = fileReader.convertGroundTruthFileToBoundingBoxList(
            gtFilePath)

        for detDirPath in detDatasetDirPathList:
            detFilePath: Path = detDirPath / frameFile
            detBBoxList: list[DetectionBoundingBox] = fileReader.convertDetectionFileToBoundingBoxList(
                detFilePath)
            detFrameData.append(detBBoxList)

        yield gtBBoxList, detFrameData


def computeUnionFalsePositive(gtBBoxList: list[GroundTruthBoundingBox], BBoxGroupList: list[list[DetectionBoundingBox]], iouThreshold: float) -> list[list[DetectionBoundingBox]]:
    unionFalsePositiveGroupList: list[list[DetectionBoundingBox]] = []

    for BBoxGroup in BBoxGroupList:
        averagedBBox: DetectionBoundingBox = averageBoundingBox(BBoxGroup)

        for gtBBox in gtBBoxList:
            iou = averagedBBox.computeIoU(gtBBox)
            if iou > iouThreshold:
                break
        else:
            unionFalsePositiveGroupList.append(BBoxGroup)

    return unionFalsePositiveGroupList


def computeIntersectionFalsePositive(gtBBoxList: list[GroundTruthBoundingBox], BBoxGroupList: list[list[DetectionBoundingBox]], numVersion: int, iouThreshold: float) -> list[list[DetectionBoundingBox]]:
    unanimousFalsePositiveGroupList: list[list[DetectionBoundingBox]] = []
    # ----------
    # 全会一致の検出を抽出
    # ----------
    unanimousDetectionList: list[list[DetectionBoundingBox]] = list(
        filter(lambda BBoxGroup: len(BBoxGroup) == numVersion, BBoxGroupList))

    for BBoxGroup in unanimousDetectionList:
        averagedBBox: DetectionBoundingBox = averageBoundingBox(BBoxGroup)

        for gtBBox in gtBBoxList:
            iou = averagedBBox.computeIoU(gtBBox)
            if iou > iouThreshold:
                break
        else:
            # ----------
            # どのgtともマッチしなかったらFP
            # ----------
            unanimousFalsePositiveGroupList.append(BBoxGroup)

    return unanimousFalsePositiveGroupList


def computeIntersectionFalseNegative(gtBBoxList: list[GroundTruthBoundingBox], BBoxGroupList: list[list[DetectionBoundingBox]], iouThreshold: float) -> list[GroundTruthBoundingBox]:
    unanimousFalseNegativeList: list[GroundTruthBoundingBox] = []

    # ----------
    # 全グループの平均化
    # ----------
    averagedBBoxList: list[DetectionBoundingBox] = list(
        map(averageBoundingBox, BBoxGroupList))

    # ----------
    # (gtIdx, detIdx, Iou)のリスト作成
    # ----------
    IOU_VALUE_INDEX = 2
    iouPairList: list[tuple[int, int, float]] = []
    for gtIdx, gtBBox in enumerate(gtBBoxList):
        for detIdx, detBBox in enumerate(averagedBBoxList):
            iou = gtBBox.computeIoU(detBBox)
            if iou < iouThreshold:
                continue
            iouPair: tuple[int, int, float] = (gtIdx, detIdx, iou)
            iouPairList.append(iouPair)
    iouPairList.sort(
        key=lambda iouPair: iouPair[IOU_VALUE_INDEX], reverse=True)

    # ----------
    # マッチングの処理
    # ----------
    matchedGtIdx: set[int] = set()
    matchedDetIdx: set[int] = set()
    for gtIdx, detIdx, iou in iouPairList:
        if (gtIdx in matchedGtIdx) or (detIdx in matchedDetIdx):
            continue

        matchedGtIdx.add(gtIdx)
        matchedDetIdx.add(detIdx)

    unanimousFalseNegativeList = [
        gtBBoxList[idx]
        for idx in range(len(gtBBoxList))
        if idx not in matchedGtIdx
    ]

    return unanimousFalseNegativeList


def computeCov(gtDatasetDirPath: Path, detDatasetDirPathList: list[Path], iouThreshold: float) -> float:
    """
    Covの計算
    (メモリ効率のため、こっちでファイル展開しながらgeneratorで処理)
    """
    cov: float = 0.0
    numImage: int = 0
    numVersion: int = len(detDatasetDirPathList)

    # ----------
    # データセットの展開（リスト）
    # ----------
    for gtBBoxList, detFrameData in frameDataGenerator(gtDatasetDirPath, detDatasetDirPathList):
        # ----------
        # 検出結果のグルーピング
        # ----------
        detModelDict: dict[int, list[DetectionBoundingBox]] = {}
        for index, detBBoxList in enumerate(detFrameData):
            detModelDict[index] = detBBoxList
        BBoxGroupList: list[list[DetectionBoundingBox]
                            ] = groupingBoundingBox(detModelDict, iouThreshold)

        # ----------
        # 共通エラーを抽出
        # ----------
        unanimousFalsePositiveList: list[list[DetectionBoundingBox]] = computeIntersectionFalsePositive(
            gtBBoxList, BBoxGroupList, numVersion, iouThreshold)
        unanimousFalseNegativeList: list[GroundTruthBoundingBox] = computeIntersectionFalseNegative(
            gtBBoxList, BBoxGroupList, iouThreshold)

        # ----------
        # 総インスタンス数の計算
        # ----------
        unionFalsePositiveList: list[list[DetectionBoundingBox]] = computeUnionFalsePositive(
            gtBBoxList, BBoxGroupList, iouThreshold)
        totalBBoxInstance: int = len(gtBBoxList) + len(unionFalsePositiveList)

        cov += (len(unanimousFalsePositiveList) +
                len(unanimousFalseNegativeList)) / totalBBoxInstance
        numImage += 1

    cov = 1 - (cov / numImage)

    return cov
