import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse

from adaptiveNversion.NversionExecutor import NversionExecutor
from adaptiveNversion.versionController.VersionController import VersionController, VersionState
from adaptiveNversion.integrator import MajorityIntegrator
from adaptiveNversion.statsRecorder import StatsRecorder

from boundingBox.boundingBox import DetectionBoundingBox

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Adaptive Object Detection"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        choices=["yolov8n", "yolov5n", "yolov11n",
                 "rtdetr", "ssd", "yolov8l", "fastrcnn"],
        nargs='+',
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )

    args = argparser.parse_args()
    print(args)

    modelNameList: list[str] = args.models
    numModel: int = len(modelNameList)
    mapName: str = args.map

    modelList: list[object] = list()

    for modelName in modelNameList:
        if modelName == "yolov8n":
            from ObjectDetection.models.Yolov8n import Yolov8nDetector
            model = Yolov8nDetector()
        elif modelName == "yolov11n":
            from ObjectDetection.models.Yolov11n import Yolov11nDetector
            model = Yolov11nDetector()
        elif modelName == "yolov5n":
            from ObjectDetection.models.Yolov5n import Yolov5nDetector
            model = Yolov5nDetector()
        elif modelName == "rtdetr":
            from ObjectDetection.models.rtDETR import RTDETRDetector
            model = RTDETRDetector()
        elif modelName == 'yolov8l':
            from ObjectDetection.models.yolov8l import Yolov8lDetector
            model = Yolov8lDetector()
        elif modelName == "ssd":
            from ObjectDetection.models.SSD import SSDDetector
            model = SSDDetector()
        elif modelName == "fastrcnn":
            from ObjectDetection.models.FastRCNN import FastRCNNDetector
            model = FastRCNNDetector()
        else:
            raise ValueError(
                f"モデル '{modelName}' はサポートされていません。\n"
            )
        modelList.append(model)

    cwd: Path = Path(__file__).parent

    inputImageDir: Path = cwd / "output" / "image" / \
        f"{mapName}" / "original" / "front"

    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    print("map: ", mapName)
    print("models: ", modelList)

    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(inputImageDir):
        raise FileNotFoundError(
            f"Input directory does not exist: {inputImageDir},\n execution file is {Path(__file__)}")

    inputFileList: list[str] = os.listdir(inputImageDir)

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    AGREEMENT_THRESHOLD = 0.5

    numVersion: int = len(modelList)

    detectionIntegrator: MajorityIntegrator = MajorityIntegrator(
        iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)
    detectionExecutor: NversionExecutor = NversionExecutor(
        modelList, detectionIntegrator)
    versionController: VersionController = VersionController(
        confidenceScoreThreshold=CONF_THRESHOLD, agreementScoreThreshold=AGREEMENT_THRESHOLD, maxVersion=numVersion)
    statesRecorder = StatsRecorder(modelNameList=modelNameList)

    # 計測開始
    start: float = time.time()
    for inputFile in tqdm(inputFileList):
        inputImagePath: Path = inputImageDir / inputFile
        if not os.path.exists(inputImagePath):
            raise FileNotFoundError(f"{inputImagePath} does not exist")

        if versionController.state == VersionState.ONE:
            baseDetection = detectionExecutor.executeOneVersionDetection(
                inputImagePath)
            versionController.updateState(detections=baseDetection)

            if versionController.state == VersionState.N:
                integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.executeNMinusOneVersionDetection(
                    inputImagePath, baseDetection)
                finalDetections = integratedBoundingBoxList
            elif versionController.state == VersionState.ONE:
                finalDetections = baseDetection
            executeState = versionController.state
        elif versionController.state == VersionState.N:
            executeState = versionController.state
            integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.executeNVersionDetection(
                inputImagePath)
            versionController.updateState(
                groupedBoundingBoxList=groupedBoundingBoxList)
            finalDetections = integratedBoundingBoxList

        statesRecorder.update(executeState)

        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    statesRecorder.registerExecutionTime(executionTime)
    print(f"total object detection time: {end - start:.2f} seconds")

    outputLabelDir: Path = cwd / "adaptiveDetectionResult" / "labels" / \
        f"{mapName}" / f"{'_'.join(modelNameList)}"
    os.makedirs(outputLabelDir, exist_ok=True)

    index: int = 0
    for outputLabelList in outputDetectionList:
        outputLabelPath = outputLabelDir / f"{index:06}.txt"
        with open(outputLabelPath, 'w') as outputFile:
            for boundingBox in outputLabelList:
                xCenter = boundingBox.xCenter
                yCenter = boundingBox.yCenter
                width = boundingBox.width
                height = boundingBox.height
                classId = boundingBox.classId
                confidenceScore = boundingBox.confidenceScore
                outputFile.write(
                    f"{classId} {xCenter} {yCenter} {width} {height} {confidenceScore}\n")
        index += 1

    outputStatsFilePath: Path = cwd / "adaptiveDetectionResult" / "resultStats.csv"
    statesRecorder.writeStatsToCsvFile(
        statsWriteCsvFilePath=outputStatsFilePath)
