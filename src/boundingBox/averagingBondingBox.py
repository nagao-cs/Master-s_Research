from .boundingBox import DetectionBoundingBox


def averageBoundingBox(boungingBoxList: list[DetectionBoundingBox]) -> DetectionBoundingBox:
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
