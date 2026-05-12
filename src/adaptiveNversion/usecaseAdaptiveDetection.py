import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse

from .usecaseDetExecutor import UseCaseNversionExecutor
from .versionController.usecaseVersionController import UseCaseVersionController, UseCaseVersionState
from ..boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from ..boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from .stats.statsRecorder import StatsRecorder

from src.boundingBox.boundingBox import DetectionBoundingBox

if __name__ == "__main__":
    # ----------
    # 引数の受付
    # ----------
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
    argparser.add_argument(
        "--cov",
        type=str,
        required=True,
        nargs="+"
    )
    argparser.add_argument(
        "--cer",
        type=str,
        required=True,
        nargs="+"
    )

    # -----------
    # 引数の整理
    # -----------
    args = argparser.parse_args()
    print(args)
    mapName: str = args.map
    modelNameList: list[str] = args.models

    # covComb = modelNameList
    # cerComb = modelNameList
    covComb: list[str] = args.cov
    covDetectors: list[object] = []
    cerComb: list[str] = args.cer
    cerDetectors: list[object] = []
    numVersion: int = len(modelNameList)

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
        elif modelName == "retinanet":
            from src.ObjectDetection.models.retinanet import RetinanetDetector
            model = RetinanetDetector()
        else:
            raise ValueError(
                f"モデル '{modelName}' はサポートされていません。\n"
            )
        modelList.append(model)

    for modelName in covComb:
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
        elif modelName == "retinanet":
            from src.ObjectDetection.models.retinanet import RetinanetDetector
            model = RetinanetDetector()
        else:
            raise ValueError(
                f"モデル '{modelName}' はサポートされていません。\n"
            )
        covDetectors.append(model)

    for modelName in cerComb:
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
        elif modelName == "retinanet":
            from src.ObjectDetection.models.retinanet import RetinanetDetector
            model = RetinanetDetector()
        else:
            raise ValueError(
                f"モデル '{modelName}' はサポートされていません。\n"
            )
        cerDetectors.append(model)

    # -----------
    # 入力ファイルの整理
    # -----------
    cwd: Path = Path(__file__).parent
    baseDir: Path = cwd.parent.parent
    inputImageDir: Path = baseDir / "output" / "image" / \
        f"{mapName}" / "original" / "front"
    if not os.path.exists(inputImageDir):
        raise FileNotFoundError(
            f"Input directory does not exist: {inputImageDir},\n execution file is {Path(__file__)}")
    inputImagePathList: list[Path] = []
    for inputImageName in os.listdir(inputImageDir):
        inputImagePath = inputImageDir / inputImageName
        if not os.path.exists(inputImagePath):
            raise FileNotFoundError(
                f"Input File does not exist: {inputImagePath}"
            )
        inputImagePathList.append(inputImagePath)

    # -----------
    # 出力ディレクトリの整理
    # -----------
    outputLabelDir: Path = baseDir / "adaptiveDetectionResult" / "usecase" / "labels" / \
        f"{mapName}" / f"{'_'.join(modelNameList)}"
    os.makedirs(outputLabelDir, exist_ok=True)
    outputStatsFilePath: Path = baseDir / \
        "adaptiveDetectionResult" / "resultStats.csv"
    stateTransitionCsvFilePath: Path = baseDir / \
        "adaptiveDetectionResult" / f"{mapName}_stateTransition.csv"

    # -----------
    # 最終出力の一時保存リスト
    # -----------
    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    # ----------
    # 余計なcli出力を消す
    # ----------
    logger = getLogger('ultralytics')
    logger.disabled = True

    # ----------
    # 定数の管理（要改善）
    # ----------
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    NUM_OBJ_THRESHOLD = 5
    INTEGRATE_CONF_THRESHOLD = 0.4
    print(
        f"cer遷移の信頼度:{CONF_THRESHOLD}, cer遷移の物体数:{NUM_OBJ_THRESHOLD}, 採用の信頼度:{INTEGRATE_CONF_THRESHOLD}")

    # -----------
    # Nバージョン検出の実行や管理のためのクラスを管理
    # -----------
    detectionIntegrator: ConfidenceBaseIntegrator = ConfidenceBaseIntegrator(
        iouThreshold=IOU_THRESHOLD, confidenceThreshold=INTEGRATE_CONF_THRESHOLD)
    # detectionIntegrator = MajorityIntegrator(
    # iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)
    detectionExecutor: UseCaseNversionExecutor = UseCaseNversionExecutor(
        detectors=modelList, detectionIntegrator=detectionIntegrator, covComb=covDetectors, cerCovb=cerDetectors)
    versionController: UseCaseVersionController = UseCaseVersionController(
        confidenceScoreThreshold=CONF_THRESHOLD, numObjThreshold=NUM_OBJ_THRESHOLD, maxVersion=numVersion)
    statesRecorder = StatsRecorder(modelNameList=modelNameList)
    exeVersionRecorder: list[UseCaseVersionState] = []

    # ----------
    # 計測開始
    # ----------
    start: float = time.time()
    for inputImagePath in tqdm(inputImagePathList):
        # ----------
        # 1バージョンの場合
        # ----------
        if versionController.state == UseCaseVersionState.ONE:
            # ----------
            # まず検出をしてそれをもとにバージョンを変えるか決める
            # ----------
            baseDetection = detectionExecutor.executeOneVersionDetection(
                inputImagePath)
            versionController.updateState(BBoxList=baseDetection)
            finalDetections = baseDetection
            # ----------
            # 状態が変わったなら追加の検出
            # ----------
            if versionController.state == UseCaseVersionState.COV_STATE:
                integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.exeCovDetection(
                    imagePath=inputImagePath)
                finalDetections = integratedBoundingBoxList
                # ----------
                # 最終的な検出バージョンを記録
                # ----------
                executeState = versionController.state
            elif versionController.state == UseCaseVersionState.CER_STATE:
                integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.exeCerDetection(
                    imagePath=inputImagePath)
                finalDetections = integratedBoundingBoxList
                # ----------
                # 最終的な検出バージョンを記録
                # ----------
                executeState = versionController.state
                # ----------
                # FPが次のフレームにもあるとは限らないため1バージョンにもどす
                # ----------
                versionController.state = UseCaseVersionState.ONE
            else:
                # ----------
                # 最終的な検出バージョンを記録
                # ----------
                executeState = versionController.state
        # ----------
        # Covの場合
        # ----------
        elif versionController.state == UseCaseVersionState.COV_STATE:
            executeState = versionController.state
            integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.exeCovDetection(
                inputImagePath)
            # ----------
            # 最終的な検出バージョンを記録
            # ----------
            executeState = versionController.state
            versionController.updateState(integratedBoundingBoxList)
            finalDetections = integratedBoundingBoxList
        # ----------
        # Cervの場合
        # ----------
        elif versionController.state == UseCaseVersionState.CER_STATE:
            executeState = versionController.state
            integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.exeCerDetection(
                inputImagePath)
            # ----------
            # 最終的な検出バージョンを記録
            # ----------
            executeState = versionController.state
            versionController.updateState(integratedBoundingBoxList)
            finalDetections = integratedBoundingBoxList

        # ----------
        # バージョンの記録
        # ----------
        statesRecorder.update(executeState)
        exeVersionRecorder.append(executeState)
        # -----------

        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    statesRecorder.registerExecutionTime(executionTime)
    print(f"total object detection time: {end - start:.2f} seconds")

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

    from collections import Counter
    exeVersionCounter = Counter(exeVersionRecorder)
    print(exeVersionCounter)

    # statesRecorder.writeStatsToCsvFile(
    # statsWriteCsvFilePath=outputStatsFilePath)

    statesRecorder.saveStateTransition(stateTransitionCsvFilePath)
