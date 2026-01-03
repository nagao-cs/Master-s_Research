from boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory


def computeF1Score(classifiedBoundingBoxList: list[ClassifiedBoundingBox]) -> float:
    numTruePositive: int = 0
    numFalsePositive: int = 0
    numFalseNegative: int = 0

    for classifiedBoundingBox in classifiedBoundingBoxList:
        if classifiedBoundingBox.classifyCategory == ClassifyCategory.TP:
            numTruePositive += 1
        elif classifiedBoundingBox.classifyCategory == ClassifyCategory.FP:
            numFalsePositive += 1
        elif classifiedBoundingBox.classifyCategory == ClassifyCategory.FN:
            numFalseNegative += 1

    f1 = numTruePositive / \
        (numTruePositive + ((numFalsePositive + numFalseNegative) / 2))

    return f1
