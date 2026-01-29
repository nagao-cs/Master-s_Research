from pathlib import Path
from argparse import ArgumentParser
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.boundingBox.boundingBox import BoundingBox, GroundTruthBoundingBox, DetectionBoundingBox, ClassifiedBoundingBox, ClassifyCategory
from src.Evaluation.dataset import fileReader
from src.Evaluation.utils.utils import IM_HEIGHT, IM_WIDTH


def calculateBoundingBoxSize(boundingBox: BoundingBox) -> float:
    return boundingBox.width * IM_WIDTH * boundingBox.height * IM_HEIGHT


if __name__ == "__main__":
    # -----------
    # コマンドライン引数の整理
    # -----------
    argparser = ArgumentParser(
        description="False Negative Analysis"
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
        "--iou",
        type=float,
        required=False,
        default=0.5
    )

    args = argparser.parse_args()
    print(args)

    modelNameList: list[str] = args.models
    numVersion: int = len(modelNameList)
    modelCombinationName: str = "_".join(modelNameList)
    mapName: str = args.map
    iouThreshold = args.iou

    # ------------
    # 入出力のファイル整理
    # ------------
    baseDir: Path = Path(__file__).parent.parent.parent

    groundTruthDatasetDirPath: Path = baseDir / \
        "output" / "label" / f"{mapName}" / "front"
    if numVersion > 1:
        # detectionResultDirPath: Path = baseDir / \
        #     "adaptiveDetectionResult" / "labels" / \
        #     f"{mapName}" / f"{modelCombinationName}"
        detectionResultDirPath = baseDir / "adaptiveDetectionResult" / \
            "labels" / "affirmative" / f"{mapName}" / f"{modelCombinationName}"
        outputFigSaveDir: Path = baseDir / \
            "adaptiveDetectionResult" / "figure" / "histgram"

    if numVersion == 1:
        detectionResultDirPath: Path = baseDir / \
            "oneVersionDetectionResult" / "labels" / \
            f"{mapName}" / f"{modelCombinationName}"
        outputFigSaveDir: Path = baseDir / \
            "oneVersionDetectionResult" / "figure" / "histgram"

    if not os.path.exists(groundTruthDatasetDirPath):
        raise FileNotFoundError(
            f"groundTruth directory does not exist: {groundTruthDatasetDirPath},\n execution file is {Path(__file__)}")
    if not os.path.exists(detectionResultDirPath):
        raise FileNotFoundError(
            f"detection Result CSV File does not exist: {detectionResultDirPath}")
    os.makedirs(outputFigSaveDir, exist_ok=True)

    # -----------
    # ディレクトリの展開
    # -----------
    groundTruthFilePathList: list[Path] = [groundTruthDatasetDirPath /
                                           groundTruthFilePath for groundTruthFilePath in os.listdir(groundTruthDatasetDirPath)]
    detectionResultFilePathList: list[Path] = [detectionResultDirPath /
                                               detectionResultFilePath for detectionResultFilePath in os.listdir(detectionResultDirPath)]

    # -----------
    # FNの収集
    # -----------
    detectionClassifier = DetectionClassifier(iouThreshold=iouThreshold)
    falseNegativeBoundingBoxList: list[list[ClassifiedBoundingBox]] = []
    truePositiveBoundingBoxList: list[list[ClassifiedBoundingBox]] = []

    for groundTruthFilePath, detectionResultFilePath in zip(groundTruthFilePathList, detectionResultFilePathList):
        groundTruthBoundingBoxList: list[GroundTruthBoundingBox] = fileReader.convertGroundTruthFileToBoundingBoxList(
            groundTruthFilePath)
        detectionBoundingBoxList: list[DetectionBoundingBox] = fileReader.convertDetectionFileToBoundingBoxList(
            detectionResultFilePath)
        classifiedBoundingBoxList: list[ClassifiedBoundingBox] = detectionClassifier.classify(
            groundTruthBoundingBoxList, detectionBoundingBoxList)

        # ----------
        # FNのみを抽出
        # ----------
        falseNegativeBoundingBoxList.append(list(filter(
            lambda boundingBox: boundingBox.classifyCategory == ClassifyCategory.FN, classifiedBoundingBoxList)))

        # ----------
        # TPのみを抽出
        # ----------
        truePositiveBoundingBoxList.append(list(filter(
            lambda boundingBox: boundingBox.classifyCategory == ClassifyCategory.TP, classifiedBoundingBoxList)))

    # -----------
    # FNの傾向や特徴を分析
    # -----------
    # 発生頻度
    appearanceCountDict: defaultdict[int, int] = defaultdict(int)
    fnCountDict: defaultdict[int, int] = defaultdict(int)
    # 大きさ
    tpSizeList: list[float] = []
    fnSizeList: list[float] = []

    for frameTpBoundingBoxList, frameFnBoundingBoxList in zip(truePositiveBoundingBoxList, falseNegativeBoundingBoxList):
        for tpBoundingBox in frameTpBoundingBoxList:
            appearanceCountDict[tpBoundingBox.classId] += 1
            boundingBoxSize: float = calculateBoundingBoxSize(tpBoundingBox)
            tpSizeList.append(boundingBoxSize)

        for fnBoundingBox in frameFnBoundingBoxList:
            appearanceCountDict[fnBoundingBox.classId] += 1
            fnCountDict[fnBoundingBox.classId] += 1
            boundingBoxSize: float = calculateBoundingBoxSize(fnBoundingBox)
            fnSizeList.append(boundingBoxSize)
    tpSizeList = np.array(tpSizeList)
    fnSizeList = np.array(fnSizeList)

    # ----------
    # 結果の表示
    # ----------
    print(f"FN : {fnCountDict}")
    print(f"total: {appearanceCountDict}")

    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.histplot(tpSizeList, label="True Positive",
    #              color="skyblue", kde=True, alpha=0.6)
    sns.histplot(fnSizeList, label='False Negative',
                 color='orange', kde=True, ax=ax, alpha=0.6)

    ax.set_title("Comparison of BoundingBox size", fontsize=16)
    ax.set_xlabel("Size (pixels)", fontsize=12)
    ax.set_xbound(lower=0, upper=1000)
    ax.set_ylabel("Freqency", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    histgramSavePath: Path = outputFigSaveDir / \
        f"{mapName}_{modelCombinationName}.png"
    fig.savefig(histgramSavePath, bbox_inches='tight', dpi=300)
