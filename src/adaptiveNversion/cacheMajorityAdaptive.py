# src/adaptiveNversion/usecaseAdaptiveDetectionLoadCache.py
import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse

from .versionController.VersionController import VersionController, VersionState
from ..boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from ..boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from .stats.statsRecorder import StatsRecorder

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.Evaluation.dataset import fileReader

if __name__ == "__main__":
    # ----------
    # 引数の受付
    # ----------
    argparser = argparse.ArgumentParser(
        description="Adaptive Object Detection using Cached Detection Results"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        nargs='+',
        help="Models to use (e.g., yolov8n yolov11n)"
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )
    # argparser.add_argument(
    #     "--cov",
    #     type=str,
    #     required=True,
    #     nargs="+",
    #     help="Models for COV state (e.g., yolov5n yolov11n)"
    # )
    # argparser.add_argument(
    #     "--cer",
    #     type=str,
    #     required=True,
    #     nargs="+",
    #     help="Models for CER state (e.g., rtdetr ssd)"
    # )

    # -----------
    # 引数の整理
    # -----------
    args = argparser.parse_args()
    print(args)
    mapName: str = args.map
    modelNameList: list[str] = args.models
    numVersion: int = len(modelNameList)

    # -----------
    # 既存の検出結果ディレクトリから読み込み
    # -----------
    cwd: Path = Path(__file__).parent
    baseDir: Path = cwd.parent.parent
    detectionBaseDir: Path = baseDir / "oneVersionDetectionResult" / "labels" / mapName

    print("Loading detection results from cached files...")
    oneVersionDetectionCache: dict[str, list[list[DetectionBoundingBox]]] = {}

    # 各モデルの検出結果を読み込む
    for modelName in modelNameList:
        modelDetectionDir = detectionBaseDir / modelName
        if not os.path.exists(modelDetectionDir):
            raise FileNotFoundError(
                f"Detection directory does not exist: {modelDetectionDir}\n"
                f"Available models: {os.listdir(detectionBaseDir)}"
            )

        # ファイルを番号順でソート
        detectionFiles = sorted([
            f for f in os.listdir(modelDetectionDir)
            if f.endswith('.txt')
        ])

        modelDetections: list[list[DetectionBoundingBox]] = []
        for detectionFile in detectionFiles:
            detectionFilePath = modelDetectionDir / detectionFile
            detectionBBoxList = fileReader.convertDetectionFileToBoundingBoxList(
                str(detectionFilePath)
            )
            modelDetections.append(detectionBBoxList)

        oneVersionDetectionCache[modelName] = modelDetections
        print(f"  Loaded {len(modelDetections)} frames for {modelName}")

    # フレーム数の確認
    numFrames = len(oneVersionDetectionCache[modelNameList[0]])
    print(f"Total frames: {numFrames}")

    # -----------
    # 出力ディレクトリの整理
    # -----------
    outputLabelDir: Path = baseDir / "adaptiveDetectionResult" / "majority" / "labels" / \
        f"{mapName}" / f"{'_'.join(modelNameList)}"
    os.makedirs(outputLabelDir, exist_ok=True)
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
    CONF_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.5

    print(
        f"信頼度の閾値:{CONF_THRESHOLD}")

    # -----------
    # 統合と状態管理のためのクラスを管理
    # -----------
    detectionIntegrator = MajorityIntegrator(
        iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)
    versionController: VersionController = VersionController(
        confidenceScoreThreshold=CONF_THRESHOLD, agreementScoreThreshold=0, maxVersion=numVersion)
    statesRecorder = StatsRecorder(modelNameList=modelNameList)
    exeVersionRecorder: list[VersionState] = []

    # ----------
    # 計測開始
    # ----------
    print("\nCombining detection results with adaptive strategy...")
    start: float = time.time()

    for frameIdx in tqdm(range(numFrames)):
        versionController.state = VersionState.ONE

        baseDetection = oneVersionDetectionCache[modelNameList[0]][frameIdx]
        versionController.updateState(BBoxList=baseDetection)
        finalDetections = baseDetection

        if versionController.state == VersionState.N:
            Detections = {
                modelName: oneVersionDetectionCache[modelName][frameIdx] for modelName in modelNameList
            }
            BBoxGroupList = groupingBoundingBox(
                Detections, iouThreshold=IOU_THRESHOLD)
            finalDetections = detectionIntegrator.integrate(BBoxGroupList)

        executeState = versionController.state

        # バージョン記録
        statesRecorder.update(executeState)
        exeVersionRecorder.append(executeState)
        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    statesRecorder.registerExecutionTime(executionTime)
    print(f"\nTotal processing time: {executionTime:.2f} seconds")

    # 結果の保存
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
    print(f"State distribution: {exeVersionCounter}")
    # statesRecorder.saveStateTransition(stateTransitionCsvFilePath)
    print(f"Results saved to: {outputLabelDir}")
