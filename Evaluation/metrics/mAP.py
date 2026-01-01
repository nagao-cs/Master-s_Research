from boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory
import numpy as np


def computeAP(targetBoundingBoxList: list[ClassifiedBoundingBox]) -> float:
    numTruePositive = 0
    numFalsePositive = 0
    numGroundTruthBoundingBox = 0

    numTruePositiveList: list[int] = list()
    numFalsePositiveList: list[int] = list()

    for boundingBox in targetBoundingBoxList:
        if (boundingBox.classifyCategory == ClassifyCategory.TP) or (boundingBox.classifyCategory == ClassifyCategory.FN):
            numGroundTruthBoundingBox += 1

    recallLevelIndex = 1
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

    return float(ap)


def computeMeanAP(classifiedBoundingBoxList: list[ClassifiedBoundingBox], targetClassIdList: list[int]) -> tuple[float, dict[int, float]]:
    classIdApDict = {classId: list() for classId in targetClassIdList}

    for targetClassId in targetClassIdList:
        targetClassIdBoundingBoxList: list[ClassifiedBoundingBox] = list(filter(
            lambda boundingBox: boundingBox.classId == targetClassId, classifiedBoundingBoxList))

        targetClassIdAp = computeAP(targetClassIdBoundingBoxList)
        classIdApDict[targetClassId] = targetClassIdAp

    mAP: float = sum(classIdApDict.values()) / len(targetClassIdList)

    return mAP, classIdApDict
