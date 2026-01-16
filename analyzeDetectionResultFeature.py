import argparse
from pathlib import Path
import os
from boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox, ClassifiedBoundingBox, ClassifyCategory
from Evaluation.classifier.detectionClassifier import DetectionClassifier
from Evaluation.dataset import fileReader
from Evaluation.metrics.confidenceScoreAnalysis import ConfidenceScoreAnalyzer

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
    args = argparser.parse_args()
    print(args)

    mapName = args.map
    modelNameList = args.models
    iouThreshold = args.iou_th

    targetModelCombination = f"{'_'.join(modelNameList)}"

    cwd = Path(__file__).parent

    groundTruthDatasetDir = cwd / "output" / "label" / \
        f"{mapName}" / "front"
    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(f"{groundTruthDatasetDir} does not exist")

    # detectionDatasetDir = cwd / "ObjectDetection" / "output" / \
        # f"{mapName}" / "labels" / f"{targetModelCombination}" / "front"
    detectionDatasetDir = cwd / "adaptiveDetectionResult" / \
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

    analyzer = ConfidenceScoreAnalyzer()

    # クラスごと・全体の統計を出力
    analyzer.print_analysis(
        classifiedBoundingBoxList,
        by_class=True,
        overall=True
    )

    # プログラムで使用するために統計情報を取得
    by_class_stats = analyzer.analyze_by_class(classifiedBoundingBoxList)
    overall_stats = analyzer.analyze_overall(classifiedBoundingBoxList)
