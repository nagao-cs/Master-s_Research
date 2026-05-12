import argparse
from pathlib import Path
import os

from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox, ClassifiedBoundingBox, ClassifyCategory
from .classifier.detectionClassifier import DetectionClassifier
from .dataset import fileReader
from .metrics import f1Score

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="compute mAP")
    argparser.add_argument(
        "--iou_th",
        type=float,
        default=0.5,
    )
    argparser.add_argument(
        "--map",
        type=str,
        default="Town02",
    )
    argparser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True
    )
    argparser.add_argument(
        "--is_adaptive",
        type=str,
        required=True
    )
    args = argparser.parse_args()
    print(args)

    mapName = args.map
    modelNameList = args.models
    iouThreshold = args.iou_th

    targetModelCombination = f"{'_'.join(modelNameList)}"

    cwd = Path(__file__).parent
    baseDir = cwd.parent.parent

    groundTruthDatasetDir = baseDir / "output" / "label" / \
        f"{mapName}" / "front"
    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(f"{groundTruthDatasetDir} does not exist")
    if args.is_adaptive == 'True':  # 適応的処理の場合
        detectionDatasetDir = baseDir / "adaptiveDetectionResult" / "escalation" / "labels" / \
            f"{mapName}" / f"{targetModelCombination}"
    elif len(modelNameList) > 1:  # 常にNバージョンの場合
        detectionDatasetDir = baseDir / "NversionDetectionResult" / "labels" / \
            f"{mapName}" / f"{targetModelCombination}"
    else:  # 常に1バージョンの場合
        detectionDatasetDir = baseDir / "oneVersionDetectionResult" / \
            "labels" / f"{mapName}" / f"{targetModelCombination}"
    if not os.path.exists(detectionDatasetDir):
        raise FileNotFoundError(f"{detectionDatasetDir} does not exist")
    classifiedBoundingBoxList: list[ClassifiedBoundingBox] = list()
    detectionClassifier: DetectionClassifier = DetectionClassifier(
        iouThreshold=iouThreshold)

    groundTruthFileList: list[str] = os.listdir(groundTruthDatasetDir)
    detectionFileList: list[str] = os.listdir(detectionDatasetDir)

    for groudTruthFile, detectionFile in zip(groundTruthFileList, detectionFileList):
        groudTruthFilePath = os.path.join(
            groundTruthDatasetDir, groudTruthFile)
        detectionFilePath = os.path.join(detectionDatasetDir, detectionFile)

        if not os.path.exists(groudTruthFilePath):
            raise FileNotFoundError(f"{groudTruthFilePath} does not exist")
        if not os.path.exists(detectionFilePath):
            raise FileNotFoundError(f"{detectionFilePath} does not exist")

        groundTruthBoundingBoxList: list[GroundTruthBoundingBox] = fileReader.convertGroundTruthFileToBoundingBoxList(
            groudTruthFilePath)
        detectionBoundingBoxList: list[DetectionBoundingBox] = fileReader.convertDetectionFileToBoundingBoxList(
            detectionFilePath)

        classifiedBoundingBoxListPerFrame: list[ClassifiedBoundingBox] = detectionClassifier.classify(
            groundTruthBoundingBoxList, detectionBoundingBoxList)

        classifiedBoundingBoxList.extend(classifiedBoundingBoxListPerFrame)

    # f1Scoreを計算する
    f1, precision, recall = f1Score.computeF1Score(classifiedBoundingBoxList)

    print(f"precision = {precision:.3f}")
    print(f"recall = {recall:.3f}")
    print(
        f"f1Score = {f1:.3f}")
