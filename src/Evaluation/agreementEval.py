import argparse
from tqdm import tqdm
import os
from pathlib import Path

from .metrics.CovCer import frameDataGenerator
from src.boundingBox.boundingBox import GroundTruthBoundingBox, DetectionBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.boundingBox.averagingBondingBox import averageBoundingBox


def computeAgreementPrecision(gtDatasetDirPath: Path, detDatasetDirPathList: list[Path], iouThreshold: float) -> tuple[float, float]:
    """
    複数モデルで一致している検出がTPである割合と、単一モデルによる検出がTPである割合を計算

    Returns:
        (agreementPrecision, singleModelPrecision)
        - agreementPrecision: 複数モデルで一致している検出がTPである割合
        - singleModelPrecision: 単一モデルによる検出がTPである割合
    """
    agreementTP: int = 0  # 複数モデルで一致してTP
    agreementDetections: int = 0  # 複数モデルで一致している検出総数

    singleModelTP: int = 0  # 単一モデルのみTP
    singleModelDetections: int = 0  # 単一モデルのみの検出総数

    numVersion: int = len(detDatasetDirPathList)

    # ----------
    # データセットの展開（リスト）
    # ----------
    for gtBBoxList, detFrameData in tqdm(frameDataGenerator(gtDatasetDirPath, detDatasetDirPathList), desc="[AgreementPrecision]", total=len(os.listdir(gtDatasetDirPath))):
        # ----------
        # 検出結果のグルーピング
        # ----------
        detModelDict: dict[int, list[DetectionBoundingBox]] = {}
        for index, detBBoxList in enumerate(detFrameData):
            detModelDict[index] = detBBoxList
        BBoxGroupList: list[list[DetectionBoundingBox]
                            ] = groupingBoundingBox(detModelDict, iouThreshold)

        # ----------
        # 全グループの平均化
        # ----------
        averagedBBoxList: list[DetectionBoundingBox] = list(
            map(averageBoundingBox, BBoxGroupList))

        # ----------
        # (gtIdx, detIdx, Iou)のリスト作成
        # ----------
        IOU_VALUE_INDEX = 2
        iouPairList: list[tuple[int, int, float]] = []
        for gtIdx, gtBBox in enumerate(gtBBoxList):
            for detIdx, detBBox in enumerate(averagedBBoxList):
                iou = gtBBox.computeIoU(detBBox)
                if iou < iouThreshold:
                    continue
                iouPair: tuple[int, int, float] = (gtIdx, detIdx, iou)
                iouPairList.append(iouPair)
        iouPairList.sort(
            key=lambda iouPair: iouPair[IOU_VALUE_INDEX], reverse=True)

        # ----------
        # マッチングの処理（iouの高い順にgtとdetを対応させていく）
        # ----------
        matchedGtIdx: set[int] = set()
        matchedDetIdx: set[int] = set()
        matchedDict: dict[int, int] = {}
        for gtIdx, detIdx, iou in iouPairList:
            if (gtIdx in matchedGtIdx) or (detIdx in matchedDetIdx):
                continue

            matchedGtIdx.add(gtIdx)
            matchedDetIdx.add(detIdx)
            matchedDict[gtIdx] = detIdx
            # ----------
            # マッチング処理
            # ----------
            matchedGtIdx: set[int] = set()
            matchedDetIdx: set[int] = set()
            for gtIdx, detIdx, iou in iouPairList:
                if (gtIdx in matchedGtIdx) or (detIdx in matchedDetIdx):
                    continue
                matchedGtIdx.add(gtIdx)
                matchedDetIdx.add(detIdx)

        # ----------
        # 複数モデル一致 vs 単一モデルの判定
        # ----------
        for detIdx in range(len(BBoxGroupList)):
            targetBBoxGroup = BBoxGroupList[detIdx]
            # ----------
            # マッチしていればTP
            # ----------
            if detIdx in matchedDetIdx:
                if len(targetBBoxGroup) == numVersion:
                    # 複数モデルで一致
                    agreementDetections += 1
                    agreementTP += 1
                else:
                    # 単一モデルのみ
                    singleModelDetections += 1
                    singleModelTP += 1

            # ----------
            # マッチしなかった検出（FP）を処理
            # ----------
            else:
                if len(targetBBoxGroup) == numVersion:
                    agreementDetections += 1
                else:
                    singleModelDetections += 1

    # ----------
    # 精度の計算
    # ----------
    agreementPrecision = agreementTP / \
        agreementDetections if agreementDetections > 0 else 0.0
    singleModelPrecision = singleModelTP / \
        singleModelDetections if singleModelDetections > 0 else 0.0

    print(
        f"一致率: {agreementDetections / (agreementDetections+singleModelDetections)}")

    return agreementPrecision, singleModelPrecision


def computeAgreementRecall(gtDatasetDirPath: Path, detDatasetDirPathList: list[Path], iouThreshold: float) -> tuple[float, float]:
    """
    複数モデルで一致している検出がTPである割合と、単一モデルによる検出がTPである割合を計算

    Returns:
        (agreementPrecision, singleModelPrecision)
        - agreementPrecision: 複数モデルで一致している検出がTPである割合
        - singleModelPrecision: 単一モデルによる検出がTPである割合
    """
    agreementTP: int = 0  # 複数モデルで一致してTP

    singleModelTP: int = 0  # 単一モデルのみTP

    numGTInstance: int = 0

    numVersion: int = len(detDatasetDirPathList)

    # ----------
    # データセットの展開（リスト）
    # ----------
    for gtBBoxList, detFrameData in tqdm(frameDataGenerator(gtDatasetDirPath, detDatasetDirPathList), desc="[AgreementRecall]", total=len(os.listdir(gtDatasetDirPath))):
        # ----------
        # 検出結果のグルーピング
        # ----------
        detModelDict: dict[int, list[DetectionBoundingBox]] = {}
        for index, detBBoxList in enumerate(detFrameData):
            detModelDict[index] = detBBoxList
        BBoxGroupList: list[list[DetectionBoundingBox]
                            ] = groupingBoundingBox(detModelDict, iouThreshold)

        # ----------
        # 全グループの平均化
        # ----------
        averagedBBoxList: list[DetectionBoundingBox] = list(
            map(averageBoundingBox, BBoxGroupList))

        # ----------
        # (gtIdx, detIdx, Iou)のリスト作成
        # ----------
        IOU_VALUE_INDEX = 2
        iouPairList: list[tuple[int, int, float]] = []
        for gtIdx, gtBBox in enumerate(gtBBoxList):
            for detIdx, detBBox in enumerate(averagedBBoxList):
                iou = gtBBox.computeIoU(detBBox)
                if iou < iouThreshold:
                    continue
                iouPair: tuple[int, int, float] = (gtIdx, detIdx, iou)
                iouPairList.append(iouPair)
        iouPairList.sort(
            key=lambda iouPair: iouPair[IOU_VALUE_INDEX], reverse=True)

        # ----------
        # マッチングの処理（iouの高い順にgtとdetを対応させていく）
        # ----------
        matchedGtIdx: set[int] = set()
        matchedDetIdx: set[int] = set()
        matchedDict: dict[int, int] = {}
        for gtIdx, detIdx, iou in iouPairList:
            if (gtIdx in matchedGtIdx) or (detIdx in matchedDetIdx):
                continue

            matchedGtIdx.add(gtIdx)
            matchedDetIdx.add(detIdx)
            matchedDict[gtIdx] = detIdx
            # ----------
            # マッチング処理
            # ----------
            matchedGtIdx: set[int] = set()
            matchedDetIdx: set[int] = set()
            for gtIdx, detIdx, iou in iouPairList:
                if (gtIdx in matchedGtIdx) or (detIdx in matchedDetIdx):
                    continue
                matchedGtIdx.add(gtIdx)
                matchedDetIdx.add(detIdx)

        # ----------
        # 複数モデル一致 vs 単一モデルの判定
        # ----------
        for detIdx in range(len(BBoxGroupList)):
            targetBBoxGroup = BBoxGroupList[detIdx]
            # ----------
            # マッチしていればTP
            # ----------
            if detIdx in matchedDetIdx:
                if len(targetBBoxGroup) == numVersion:
                    # 複数モデルで一致
                    agreementTP += 1
                else:
                    # 単一モデルのみ
                    singleModelTP += 1
        # ----------
        # GTを数える
        # ----------
        numGTInstance += len(gtBBoxList)

    # ----------
    # 再現率の計算
    # ----------
    agreementRecall: float = agreementTP / \
        numGTInstance if numGTInstance > 0 else 0.0
    singleModelRecall: float = singleModelTP / \
        numGTInstance if numGTInstance > 0 else 0.0

    return agreementRecall, singleModelRecall


if __name__ == '__main__':
    # ----------
    # 引数の処理
    # ----------
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

    mapName: str = args.map
    modelNameList: list[str] = args.models
    iouThreshold: float = args.iou_th

    # ----------
    # 入出力のファイル処理
    # ----------
    cwd: Path = Path(__file__).parent
    baseDir: Path = cwd.parent.parent

    GTDatasetDir: Path = baseDir / "output" / "label" / \
        f"{mapName}" / "front"
    if not os.path.exists(GTDatasetDir):
        raise FileNotFoundError(f"{GTDatasetDir} does not exist")
    detectionDatasetDirList: list[Path] = []
    for modelName in modelNameList:
        detectionDatasetDir: Path = baseDir / "oneVersionDetectionResult" / \
            "labels" / f"{mapName}" / f"{modelName}"
        if not os.path.exists(detectionDatasetDir):
            raise FileNotFoundError(f"{detectionDatasetDir} does not exist")

        detectionDatasetDirList.append(detectionDatasetDir)

    # -----------
    # それぞれの精度計算
    # -----------
    agreementPrecision, singleModelPrecision = computeAgreementPrecision(gtDatasetDirPath=GTDatasetDir,
                                                                         detDatasetDirPathList=detectionDatasetDirList, iouThreshold=iouThreshold)

    print(f"一致した場合の精度: {agreementPrecision}")
    print(f"単一モデルの場合の精度: {singleModelPrecision}")

    agreementRecall, singleModelRecall = computeAgreementRecall(gtDatasetDirPath=GTDatasetDir,
                                                                detDatasetDirPathList=detectionDatasetDirList, iouThreshold=iouThreshold)

    print(f"一致した場合の再現率: {agreementRecall}")
    print(f"単一モデルの場合の再現率: {singleModelRecall}")
