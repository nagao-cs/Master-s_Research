from src.boundingBox.boundingBox import GroundTruthBoundingBox, DetectionBoundingBox
from src.Evaluation.utils import utils


def convertGroundTruthFileToBoundingBoxList(groundTruthFilePath: str) -> list[GroundTruthBoundingBox]:
    groundTruthBoundingBoxList: list[GroundTruthBoundingBox] = list()

    with open(groundTruthFilePath, mode='r') as groundTruthFile:
        lineList = groundTruthFile.readlines()
        for line in lineList:
            if not line.strip():
                continue

            boundingBoxComponentList = line.strip().split(' ')
            classId = int(boundingBoxComponentList[0])
            # classId = utils.class_Map.get(
            #     (int(boundingBoxComponentList[0])), -1)  # -1（無視するクラス）
            # if classId == -1:
            #     continue
            
            xCenter = float(boundingBoxComponentList[1])
            yCenter = float(boundingBoxComponentList[2])
            width = float(boundingBoxComponentList[3])
            height = float(boundingBoxComponentList[4])
            size = width * height * utils.IM_WIDTH * utils.IM_HEIGHT
            # if size < utils.SIZE_THRESHOLD:
            #     continue
            # if ((width * utils.IM_WIDTH) < utils.SIZE_THRESHOLD) or ((height * utils.IM_HEIGHT) < utils.SIZE_THRESHOLD):
                # continue

            groundTruthBoundingBox: GroundTruthBoundingBox = GroundTruthBoundingBox(
                xCenter, yCenter, width, height, classId)

            # print(xCenter, yCenter, width, height)
            groundTruthBoundingBoxList.append(groundTruthBoundingBox)

    return groundTruthBoundingBoxList


def convertDetectionFileToBoundingBoxList(detectionFilePath: str) -> list[DetectionBoundingBox]:
    detectionBoundingBoxList: list[GroundTruthBoundingBox] = list()

    with open(detectionFilePath, mode='r') as detectionFile:
        lineList = detectionFile.readlines()
        for line in lineList:
            if not line.strip():
                continue

            boundingBoxComponentList = line.strip().split(' ')
            classId = int(boundingBoxComponentList[0])
            # classId = utils.class_Map.get(
                # (int(boundingBoxComponentList[0])), -1)  # -1（無視するクラス）
            # if classId == -1:
                # continue

            xCenter = float(boundingBoxComponentList[1])
            yCenter = float(boundingBoxComponentList[2])
            width = float(boundingBoxComponentList[3])
            height = float(boundingBoxComponentList[4])
            size = width * height * utils.IM_WIDTH * utils.IM_HEIGHT
            # if size < utils.SIZE_THRESHOLD:
            #     continue
            # if ((width * utils.IM_WIDTH) < utils.SIZE_THRESHOLD) or ((height * utils.IM_HEIGHT) < utils.SIZE_THRESHOLD):
            #     continue

            confidenceScore = float(boundingBoxComponentList[5])

            detecionBoundingBox: DetectionBoundingBox = DetectionBoundingBox(
                xCenter, yCenter, width, height, classId, confidenceScore)
            # print(xCenter, yCenter, width, height)

            detectionBoundingBoxList.append(detecionBoundingBox)

    return detectionBoundingBoxList
