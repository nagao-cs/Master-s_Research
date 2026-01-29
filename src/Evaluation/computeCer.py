import argparse
from pathlib import Path
import os

from .metrics.CovCer import computeCer, computeFpCer, computeFnCer

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

    groundTruthDatasetDir: Path = baseDir / "output" / "label" / \
        f"{mapName}" / "front"
    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(f"{groundTruthDatasetDir} does not exist")
    detectionDatasetDirList: list[Path] = []
    for modelName in modelNameList:
        detectionDatasetDir: Path = baseDir / "oneVersionDetectionResult" / \
            "labels" / f"{mapName}" / f"{modelName}"
        if not os.path.exists(detectionDatasetDir):
            raise FileNotFoundError(f"{detectionDatasetDir} does not exist")

        detectionDatasetDirList.append(detectionDatasetDir)

    # -----------
    # cov計算
    # -----------
    fpCer: float = computeFpCer(
        groundTruthDatasetDir, detectionDatasetDirList, iouThreshold)
    fnCer: float = computeFnCer(
        groundTruthDatasetDir, detectionDatasetDirList, iouThreshold)
    cer: float = computeCer(groundTruthDatasetDir,
                            detectionDatasetDirList, iouThreshold)

    print(f"cer: {cer:.3f}")
    print(f"fpCer: {fpCer:.3f}, fnCer: {fnCer:.3f}")
