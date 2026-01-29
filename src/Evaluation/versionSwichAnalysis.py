from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import csv
import numpy as np


if __name__ == "__main__":
    # -----------
    # cla の整理
    # -----------
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
    modelCombinationName: str = "_".join(modelNameList)
    mapName: str = args.map

    # ------------
    # 入出力のファイル整理
    # ------------
    baseDir: Path = Path(__file__).parent.parent

    groundTruthDatasetDir: Path = baseDir / \
        "output" / "label" / f"{mapName}" / "front"
    detectionResultCsvFilePath: Path = baseDir / \
        "adaptiveDetectionResult" / f"{mapName}_stateTransition.csv"

    figureSaveDir: Path = baseDir / "adaptiveDetectionResult" / "figure"
    os.makedirs(figureSaveDir, exist_ok=True)
    figureSavePath: Path = figureSaveDir / \
        f"{mapName}_{modelCombinationName}.png"

    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(
            f"groundTruth directory does not exist: {groundTruthDatasetDir},\n execution file is {Path(__file__)}")
    if not os.path.exists(detectionResultCsvFilePath):
        raise FileNotFoundError(
            f"detection Result CSV File does not exist: {detectionResultCsvFilePath}")

    # -----------
    # gt データの読み込み
    # -----------
    numGroundTruthObjectList: list[int] = []
    hasTrafficLightList: list[bool] = []

    gtFilePathList: list[Path] = [groundTruthDatasetDir /
                                  gtFile for gtFile in os.listdir(groundTruthDatasetDir)]

    for gtFilePath in gtFilePathList:
        numObject: int = 0
        hasTrafficLight: bool = False
        with open(gtFilePath, "r") as gtFile:
            gtObjectList: list[str] = gtFile.readlines()
            for gtObjectTxt in gtObjectList:
                if not gtObjectTxt.strip():
                    continue

                numObject += 1

                gtObjectInformation: list[str] = gtObjectTxt.strip().split(
                    sep=" ")
                gtClassId: str = gtObjectInformation[0]
                if gtClassId == "9":
                    hasTrafficLight == True

        numGroundTruthObjectList.append(numObject)
        hasTrafficLightList.append(hasTrafficLight)

    numGroundTruthObjectList = np.array(numGroundTruthObjectList)
    hasTrafficLightList = np.array(hasTrafficLightList)

    # ------------
    # 検出バージョンの切り替わりを読み込む
    # ------------
    modelCombinationName: str = "_".join(modelNameList)
    numVersionList: list[int] = []

    with open(detectionResultCsvFilePath, mode="r") as detectionResultCsvFile:
        reader = csv.DictReader(detectionResultCsvFile)
        if modelCombinationName not in reader.fieldnames:
            raise ValueError(
                f"{modelNameList} combination does not exist in {detectionResultCsvFilePath}")

        for row in reader:
            numVersionList.append(int(row[modelCombinationName]))

    numVersionList = np.array(numVersionList)

    numFrame = min(len(numGroundTruthObjectList), len(numVersionList))
    frameIdList = np.arange(start=1, stop=numFrame+1, step=1)

    # -----------
    # プロット
    # -----------
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左軸：GT物体数
    ax1.plot(
        frameIdList,
        numGroundTruthObjectList,
        label="Number of GT Objects",
    )
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("Number of GT Objects")
    ax1.grid(True)

    # 右軸：使用バージョン数
    ax2 = ax1.twinx()
    ax2.step(
        frameIdList,
        numVersionList,
        where="post",
        linestyle="--",
        label="Number of Versions",
    )
    ax2.set_ylabel("Number of Versions")

    # 信号機ありフレームを背景で強調
    for i in range(len(frameIdList)):
        if hasTrafficLightList[i]:
            ax1.axvspan(frameIdList[i] - 0.5, frameIdList[i] + 0.5, alpha=0.1)

    # 凡例まとめ
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(
        f"Models: {modelCombinationName}, Map: {mapName}"
    )

    plt.tight_layout()
    plt.savefig(figureSavePath)
