from boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


def drawPrecisionRecallCurve(targetClassId: int, precisionList: list[float], recallList: list[float], figureSaveDirPath: Path) -> None:
    os.makedirs(figureSaveDirPath, exist_ok=True)
    plt.figure(figsize=(10, 8))

    plt.plot(recallList, precisionList, marker='o',
             markersize=6, label='PR Curve', color='#1f77b4')
    plt.grid(True)

    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title(label=f"{targetClassId} PR-Curve")

    figureSavePath = figureSaveDirPath / f"{targetClassId}_prCurve.png"
    plt.savefig(figureSavePath)

    plt.close()


def _buildTargetClassIdBoundingBoxList(targetClassId: int, boundingBoxList: list[ClassifiedBoundingBox]) -> list[ClassifiedBoundingBox]:
    targetClassIdBoundingBoxList: list[ClassifiedBoundingBox] = list(filter(
        lambda boundingBox: boundingBox.classId == targetClassId, boundingBoxList))

    return targetClassIdBoundingBoxList


def computeAP(targetClassId: int, targetBoundingBoxList: list[ClassifiedBoundingBox]) -> float:
    numTruePositive: int = 0
    numFalsePositive: int = 0
    numGroundTruthBoundingBox: int = 0

    numTruePositiveList: list[int] = list()
    numFalsePositiveList: list[int] = list()

    for boundingBox in targetBoundingBoxList:
        if (boundingBox.classifyCategory == ClassifyCategory.TP) or (boundingBox.classifyCategory == ClassifyCategory.FN):
            numGroundTruthBoundingBox += 1

    for bouningBox in targetBoundingBoxList:
        if bouningBox.classifyCategory == ClassifyCategory.TP:
            numTruePositive += 1
        if bouningBox.classifyCategory == ClassifyCategory.FP:
            numFalsePositive += 1

        numTruePositiveList.append(numTruePositive)
        numFalsePositiveList.append(numFalsePositive)

    numTruePositiveList = np.array(numTruePositiveList, dtype=np.float32)
    numFalsePositiveList = np.array(numFalsePositiveList, dtype=np.float32)

    precisionList: np.array[np.float32] = numTruePositiveList / \
        (numTruePositiveList + numFalsePositiveList)
    recallList: np.array[np.float32] = numTruePositiveList / \
        numGroundTruthBoundingBox

    # 0 で始まる Recall と Precision を追加（11-point補間の基点）
    recallList = np.concatenate([np.array([0.0]), recallList])
    precisionList = np.concatenate([np.array([0.0]), precisionList])

    for i in range(len(precisionList) - 1, 0, -1):
        precisionList[i-1] = max(precisionList[i-1], precisionList[i])

    ap: float = 0.0

    NUM_POINT = 11
    for recallLevel in np.arange(0, 1.1, 0.1):
        if np.sum(recallList > recallLevel) == 0:
            precision = 0
        else:
            precision = np.max(precisionList[recallList >= recallLevel])

        ap += precision

    ap /= NUM_POINT

    cwd: Path = Path(__file__).parent
    figureSaveDirPath: Path = cwd.parent / "prCurve"
    drawPrecisionRecallCurve(targetClassId=targetClassId, precisionList=precisionList,
                             recallList=recallList, figureSaveDirPath=figureSaveDirPath)

    return float(ap)


def computeMeanAP(classifiedBoundingBoxList: list[ClassifiedBoundingBox], targetClassIdList: list[int]) -> tuple[float, dict[int, float]]:
    classIdApDict = {classId: list() for classId in targetClassIdList}

    for targetClassId in targetClassIdList:
        targetClassIdBoundingBoxList: list[ClassifiedBoundingBox] = _buildTargetClassIdBoundingBoxList(
            targetClassId, classifiedBoundingBoxList)

        targetClassIdAp: float = computeAP(
            targetClassId, targetClassIdBoundingBoxList)
        classIdApDict[targetClassId] = targetClassIdAp

    mAP: float = sum(classIdApDict.values()) / len(targetClassIdList)

    return mAP, classIdApDict
