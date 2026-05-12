import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse

from .NversionExecutor import NversionExecutor
from .versionController.prevFrameDependVersionController import PrevFrameDependVersionController, VersionState
from ..boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from .stats.statsRecorder import StatsRecorder

from src.boundingBox.boundingBox import DetectionBoundingBox

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Adaptive Object Detection"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
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
            from src.ObjectDetection.models.Yolov8n import Yolov8nDetector
            model = Yolov8nDetector()
        elif modelName == "yolov11n":
            from src.ObjectDetection.models.Yolov11n import Yolov11nDetector
            model = Yolov11nDetector()
        elif modelName == "yolov5n":
            from src.ObjectDetection.models.Yolov5n import Yolov5nDetector
            model = Yolov5nDetector()
        elif modelName == "rtdetr":
            from src.ObjectDetection.models.rtDETR import RTDETRDetector
            model = RTDETRDetector()
        elif modelName == 'yolov8l':
            from src.ObjectDetection.models.yolov8l import Yolov8lDetector
            model = Yolov8lDetector()
        elif modelName == "ssd":
            from src.ObjectDetection.models.SSD_torch import SSDDetector
            model = SSDDetector()
        elif modelName == "fastrcnn":
            from src.ObjectDetection.models.FastRCNN import FasterRCNNDetector
            model = FasterRCNNDetector()
        elif modelName == "fcos":
            from src.ObjectDetection.models.FCOS import FcosDetector
            model = FcosDetector()
        else:
            raise ValueError(
                f"モデル '{modelName}' はサポートされていません。\n"
            )
        modelList.append(model)

    cwd: Path = Path(__file__).parent
    baseDir: Path = cwd.parent.parent

    inputImageDir: Path = baseDir / "output" / "image" / \
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

    NUM_DETCTION_DIFF_THRESHOLD: int = 2
    IOU_THRESHOLD: float = 0.5

    numVersion: int = len(modelList)

    detectionIntegrator: MajorityIntegrator = MajorityIntegrator(
        iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)
    detectionExecutor: NversionExecutor = NversionExecutor(
        modelList, detectionIntegrator)
    versionController: PrevFrameDependVersionController = PrevFrameDependVersionController(
        numDetDiffThreshold=NUM_DETCTION_DIFF_THRESHOLD, maxVersion=numVersion)
    statesRecorder = StatsRecorder(modelNameList=modelNameList)

    # 計測開始
    start: float = time.time()
    for inputFile in tqdm(inputFileList):
        inputImagePath: Path = inputImageDir / inputFile
        if not os.path.exists(inputImagePath):
            raise FileNotFoundError(f"{inputImagePath} does not exist")

        # ----------
        # いったん毎フレーム1バージョンでNにするか判定
        # ----------
        versionController.state = VersionState.ONE
        if versionController.state == VersionState.ONE:
            baseDetection = detectionExecutor.executeOneVersionDetection(
                inputImagePath)
            versionController.updateState(BBoxList=baseDetection)

            if versionController.state == VersionState.N:
                integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.executeNMinusOneVersionDetection(
                    inputImagePath, baseDetection)
                finalDetections = integratedBoundingBoxList
            elif versionController.state == VersionState.ONE:
                finalDetections = baseDetection
            executeState = versionController.state

        statesRecorder.update(executeState)
        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    statesRecorder.registerExecutionTime(executionTime)
    print(f"total object detection time: {end - start:.2f} seconds")

    outputLabelDir: Path = baseDir / "adaptiveDetectionResult" / "prevFrame" / "labels" / \
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

    outputStatsFilePath: Path = baseDir / \
        "adaptiveDetectionResult" / "prevFrame" / "resultStats.csv"
    statesRecorder.writeStatsToCsvFile(
        statsWriteCsvFilePath=outputStatsFilePath)

    stateTransitionCsvFilePath: Path = baseDir / \
        "adaptiveDetectionResult" / "prevFrame" / \
        f"{mapName}_stateTransition.csv"
    statesRecorder.saveStateTransition(stateTransitionCsvFilePath)
