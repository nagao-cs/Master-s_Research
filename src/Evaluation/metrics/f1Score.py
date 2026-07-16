from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory


def computeF1Score(classifiedBoundingBoxList: list[ClassifiedBoundingBox]) -> tuple[float, float, float]:
    numTruePositive: int = 0
    numFalsePositive: int = 0
    numFalseNegative: int = 0

    VALID_CLASS_ID = (0, 2, 9, 11)
    # VALID_CLASS_ID = (0, 2)

    for classifiedBoundingBox in classifiedBoundingBoxList:
        if classifiedBoundingBox.classId not in VALID_CLASS_ID:
            continue
        if classifiedBoundingBox.classifyCategory == ClassifyCategory.TP:
            numTruePositive += 1
        elif classifiedBoundingBox.classifyCategory == ClassifyCategory.FP:
            numFalsePositive += 1
        elif classifiedBoundingBox.classifyCategory == ClassifyCategory.FN:
            numFalseNegative += 1

    precision: float = numTruePositive / (numTruePositive + numFalsePositive)
    recall: float = numTruePositive / (numTruePositive + numFalseNegative)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1, precision, recall
