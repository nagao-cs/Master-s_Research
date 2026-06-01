import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory


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

    if not np.any(numGroundTruthBoundingBox > 0):
        return -1.0
    
    numTruePositiveList = np.array(numTruePositiveList, dtype=np.float32)
    numFalsePositiveList = np.array(numFalsePositiveList, dtype=np.float32)

    total_detection_list = numTruePositiveList + numFalsePositiveList
    if np.any(total_detection_list > 0):
        precisionList = numTruePositiveList / total_detection_list
    else:
        precisionList = np.zeros(shape=len(total_detection_list))
    recallList: np.array[np.float32] = numTruePositiveList / \
        numGroundTruthBoundingBox

    # point補間の基点
    recallList = np.concatenate([np.array([0.0]), recallList])
    precisionList = np.concatenate([np.array([1.0]), precisionList])

    for i in range(len(precisionList) - 1, 0, -1):
        precisionList[i-1] = max(precisionList[i-1], precisionList[i])

    ap: float = 0.0

    NUM_POINT = 101
    recallLevelList: list[float] = [
        i * (1 / (NUM_POINT-1)) for i in range(NUM_POINT)]
    elevenPointPrecisionList: list[float] = [0.0] * NUM_POINT
    for i, recallLevel in enumerate(recallLevelList):
        if np.sum(recallList > recallLevel) == 0:
            precision = 0
        else:
            precision = np.max(precisionList[recallList > recallLevel])
        elevenPointPrecisionList[i] = precision

    elevenPointPrecisionList: np.array = np.array(elevenPointPrecisionList)

    cwd: Path = Path(__file__).parent
    figureSaveDirPath: Path = cwd.parent / "prCurve"
    drawPrecisionRecallCurve(targetClassId=targetClassId, precisionList=elevenPointPrecisionList,
                             recallList=recallLevelList, figureSaveDirPath=figureSaveDirPath)

    ap = np.sum(elevenPointPrecisionList) / NUM_POINT

    return float(ap)


def computeMeanAP(classifiedBoundingBoxList: list[ClassifiedBoundingBox], targetClassIdList: list[int]) -> tuple[float, dict[int, float]]:
    classIdApDict = {classId: list() for classId in targetClassIdList}

    for targetClassId in targetClassIdList:
        targetClassIdBoundingBoxList: list[ClassifiedBoundingBox] = _buildTargetClassIdBoundingBoxList(
            targetClassId, classifiedBoundingBoxList)

        targetClassIdAp: float = computeAP(
            targetClassId, targetClassIdBoundingBoxList)
        classIdApDict[targetClassId] = targetClassIdAp
    
    valid_ap_values = list(ap for ap in classIdApDict.values() if ap > 0 )

    mAP: float = sum(valid_ap_values) / len(valid_ap_values)

    return mAP, classIdApDict
