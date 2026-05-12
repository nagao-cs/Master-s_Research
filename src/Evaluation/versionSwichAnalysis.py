from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib.patches import Patch


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
    baseDir: Path = Path(__file__).parent.parent.parent

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
    # バージョン数ごとの色を定義
    # -----------
    version_colors = {
        1: '#FFE6E6',  # 1バージョン：薄い赤
        2: '#FFF4E6',  # 2バージョン：薄いオレンジ
        3: '#E6F4FF',  # 3バージョン：薄い青
        4: '#E6FFE6',  # 4バージョン：薄い緑
        5: '#F0E6FF',  # 5バージョン：薄い紫
    }

    # -----------
    # プロット（2段グラフ）
    # -----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [1, 1]})

    # ==================
    # 上部：GT物体数（背景にバージョン色）
    # ==================
    ax1.plot(frameIdList, numGroundTruthObjectList,
             label="Number of GT Objects", color='steelblue', linewidth=2.5, zorder=3)

    # バージョンの背景色
    i = 0
    while i < len(frameIdList):
        current_version = numVersionList[i]
        j = i
        while j < len(frameIdList) and numVersionList[j] == current_version:
            j += 1
        color = version_colors.get(current_version, '#FFFFFF')
        ax1.axvspan(frameIdList[i] - 0.5, frameIdList[j - 1] + 0.5,
                    alpha=0.2, color=color, zorder=0)
        i = j

    ax1.set_ylabel("Number of GT Objects", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.set_xlim(frameIdList[0] - 1, frameIdList[-1] + 1)

    # ==================
    # 下部：バージョン数を段差グラフで強調表示
    # ==================
    ax2.step(frameIdList, numVersionList, where='post',
             linewidth=3, color='darkred', label='Number of Versions', zorder=3)
    ax2.fill_between(frameIdList, 0, numVersionList, step='post',
                     alpha=0.3, color='darkred', zorder=2)

    # バージョンの背景色（下部も同じ）
    i = 0
    while i < len(frameIdList):
        current_version = numVersionList[i]
        j = i
        while j < len(frameIdList) and numVersionList[j] == current_version:
            j += 1
        color = version_colors.get(current_version, '#FFFFFF')
        ax2.axvspan(frameIdList[i] - 0.5, frameIdList[j - 1] + 0.5,
                    alpha=0.15, color=color, zorder=0)
        i = j

    ax2.set_ylabel("Version(s)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Frame ID", fontsize=11, fontweight='bold')
    ax2.set_ylim(1, max(numVersionList) + 1)
    ax2.set_yticks(range(1, int(max(numVersionList)) + 1))
    ax2.grid(True, alpha=0.3, axis='y', zorder=1)
    ax2.set_xlim(frameIdList[0] - 1, frameIdList[-1] + 1)

    # ==================
    # 凡例と背景色凡例
    # ==================
    legend_elements = [
        ax1.get_lines()[0],  # GT Objects
    ]

    # 使用されたバージョン数の凡例を追加
    used_versions = sorted(set(numVersionList))
    for version in used_versions:
        color = version_colors.get(version, '#FFFFFF')
        legend_elements.append(
            Patch(facecolor=color, alpha=0.6, label=f'{version} Version(s)')
        )

    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10)

    # ==================
    # タイトルと保存
    # ==================
    plt.suptitle(f"Models: {modelCombinationName}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figureSavePath, dpi=100)
    print(f"Figure saved to {figureSavePath}")
