# src/adaptiveNversion/usecaseAdaptiveDetectionLoadCache.py
import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse

from .versionController.usecaseVersionController import UseCaseVersionController, UseCaseVersionState
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
    covComb: list[str] = args.models
    cerComb: list[str] = args.models
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
    outputLabelDir: Path = baseDir / "adaptiveDetectionResult" / "usecase" / "labels" / \
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
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    NUM_OBJ_THRESHOLD = 5
    INTEGRATE_CONF_THRESHOLD = 0.5

    print(
        f"cer遷移の信頼度:{CONF_THRESHOLD}, cer遷移の物体数:{NUM_OBJ_THRESHOLD}, 採用の信頼度:{INTEGRATE_CONF_THRESHOLD}")

    # -----------
    # 統合と状態管理のためのクラスを管理
    # -----------
    detectionIntegrator: ConfidenceBaseIntegrator = ConfidenceBaseIntegrator(
        iouThreshold=IOU_THRESHOLD, confidenceThreshold=INTEGRATE_CONF_THRESHOLD)
    # detectionIntegrator: MajorityIntegrator = MajorityIntegrator(
    # iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)
    versionController: UseCaseVersionController = UseCaseVersionController(
        confidenceScoreThreshold=CONF_THRESHOLD, numObjThreshold=NUM_OBJ_THRESHOLD, maxVersion=numVersion)
    statesRecorder = StatsRecorder(modelNameList=modelNameList)
    exeVersionRecorder: list[UseCaseVersionState] = []

    # ----------
    # 計測開始
    # ----------
    print("\nCombining detection results with adaptive strategy...")
    start: float = time.time()

    for frameIdx in tqdm(range(numFrames)):
        # 1バージョンの検出結果を取得
        baseDetection = oneVersionDetectionCache[modelNameList[0]][frameIdx]
        versionController.updateState(BBoxList=baseDetection)
        finalDetections = baseDetection
        executeState = versionController.state

        # 状態遷移に応じた統合処理
        if versionController.state == UseCaseVersionState.COV_STATE:
            # COV検出結果の統合
            covDetections = {
                modelName: oneVersionDetectionCache[modelName][frameIdx] for modelName in covComb
            }
            BBoxGroupList = groupingBoundingBox(
                covDetections, iouThreshold=IOU_THRESHOLD)
            finalDetections = detectionIntegrator.integrate(BBoxGroupList)
            executeState = versionController.state
        elif versionController.state == UseCaseVersionState.CER_STATE:
            # CER検出結果の統合
            cerDetections = {
                modelName: oneVersionDetectionCache[modelName][frameIdx] for modelName in cerComb
            }
            BBoxGroupList = groupingBoundingBox(
                cerDetections, iouThreshold=IOU_THRESHOLD)
            finalDetections = detectionIntegrator.integrate(BBoxGroupList)
            executeState = versionController.state
            versionController.state = UseCaseVersionState.ONE

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
