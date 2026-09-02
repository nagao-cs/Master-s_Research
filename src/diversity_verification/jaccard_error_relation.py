"""
1. 検出結果のファイルの読み込み
2. 各フレームごとにjaccardを計算する
3. そのフレームでのエラー（FP, FN, error rate）の最大値と最小値を計算する
4. 結果を保存
"""
"""
1. 検出結果のファイルの読み込み
2. 各フレームごとにjaccardを計算する
3. そのフレームでのエラー（FP, FN, error rate）の最大値と最小値を計算する
4. 結果を保存

前提: 2モデル(N=2)の比較を対象とする(jaccard.calc_jaccardの定義に合わせる)

エラー最大値/最小値の定義:
    共通エラー   = 両モデルが同じ誤りをした場合
                   ・共通FP: 両モデルが同じ非存在物体を検出した(groupingで一致し、GTと不一致)
                   ・共通FN: 両モデルとも見逃した物体(どちらの検出ともマッチしないGT)
    片方のみのFN = 片方のモデルだけが物体を見逃した(もう片方は検出しGTと一致)
    片方のみのFP = 片方のモデルだけの誤検出(グループサイズ1でGTと不一致)

    最大値 = 共通エラー + 片方のみのFN + 片方のみのFP
             (2モデルのエラーの和集合。単独のモデルを信頼した場合に起こりうる最悪のエラー数)
    最小値 = 共通エラー
             (2モデルのエラーの積集合。両モデルが同時に間違えた場合のみをエラーとみなす)
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.Evaluation.metrics.jaccard import calc_jaccard
from src.diversity_verification.agree_error_relation import (
    loadGroundTruthFile,
    getImageIdList,
    buildDetectionModelDict,
    computeRepresentativeBox,
)
from src.config import RESULT_DIR, IOU_THRESHOLD, DATASET_DIR

# ------------------------------------------------------------
# フレーム単位のエラー分類
# ------------------------------------------------------------
 
@dataclass
class FrameErrorCounts:
    imageId: str
    jaccard: float
    totalGtCount: int
    commonFpCount: int = 0
    commonFnCount: int = 0
    oneModelFnCount: int = 0
    oneModelFpCount: int = 0
    commonTpCount: int = 0
    oneModelTpCount: int = 0
 
    @property
    def commonErrorCount(self) -> int:
        """両モデルが同時に間違えた(共通FP + 共通FN)数"""
        return self.commonFpCount + self.commonFnCount
 
    @property
    def maxErrorCount(self) -> int:
        """2モデルのエラーの和集合"""
        return self.commonErrorCount + self.oneModelFnCount + self.oneModelFpCount
 
    @property
    def minErrorCount(self) -> int:
        """2モデルのエラーの積集合(共通エラーのみ)"""
        return self.commonErrorCount
 
    @property
    def totalUnitCount(self) -> int:
        """
        GTオブジェクト数 + 誤検出として現れた物体数(共通/片方のみ問わず)
        エラー率の分母として使う「判定対象となった物体の総数」
        """
        return self.totalGtCount + self.commonFpCount + self.oneModelFpCount
 
    @property
    def maxErrorRate(self) -> float:
        total = self.totalUnitCount
        return self.maxErrorCount / total if total > 0 else float("nan")
 
    @property
    def minErrorRate(self) -> float:
        total = self.totalUnitCount
        return self.minErrorCount / total if total > 0 else float("nan")
 
 
def classifyFrame(
    imageId: str,
    detectionModelDict: dict[str, list[DetectionBoundingBox]],
    groundTruthBoxes: list[GroundTruthBoundingBox],
    groupingIouThreshold: float,
    matchIouThreshold: float,
) -> FrameErrorCounts:
    """
    1フレーム分の検出結果(2モデル)をgroupingし、GTと比較して
    共通エラー/片方のみのFN/片方のみのFPを分類する
    """
    groups = groupingBoundingBox(detectionModelDict, groupingIouThreshold)
    jaccard = calc_jaccard(groups)
 
    counts = FrameErrorCounts(
        imageId=imageId,
        jaccard=jaccard,
        totalGtCount=len(groundTruthBoxes),
    )
 
    representativeBoxes = [computeRepresentativeBox(group) for group in groups]
 
    # confidence降順に貪欲マッチング
    order = sorted(
        range(len(groups)),
        key=lambda i: representativeBoxes[i].confidenceScore,
        reverse=True,
    )
 
    matchedGtIndices: set[int] = set()
 
    for i in order:
        group = groups[i]
        repBox = representativeBoxes[i]
 
        bestIou = 0.0
        bestGtIdx = -1
        for gtIdx, gtBox in enumerate(groundTruthBoxes):
            if gtIdx in matchedGtIndices:
                continue
            iou = repBox.computeIoU(gtBox)
            if iou >= matchIouThreshold and iou > bestIou:
                bestIou = iou
                bestGtIdx = gtIdx
 
        if bestGtIdx >= 0:
            matchedGtIndices.add(bestGtIdx)
            if len(group) == 1:
                # 片方のモデルだけが検出しGTと一致 -> もう片方はこの物体を見逃した
                counts.oneModelFnCount += 1
                counts.oneModelTpCount += 1
            # len==2なら両モデルとも正しく検出(TP) -> エラーではない
            elif len(group) == 2:
                counts.commonTpCount += 1
        else:
            if len(group) == 2:
                # 両モデルが同じ非存在物体を検出 -> 共通FP
                counts.commonFpCount += 1
            else:
                # 片方のモデルだけの誤検出 -> 片方のみのFP
                counts.oneModelFpCount += 1
 
    # どちらの検出ともマッチしなかったGT -> 共通FN(両モデルが見逃した)
    counts.commonFnCount = len(groundTruthBoxes) - len(matchedGtIndices)
 
    return counts
 
 
# ------------------------------------------------------------
# データセット全体の集計
# ------------------------------------------------------------
 
def run(
    dataset: str,
    mapName: str,
    modelNames: list[str],
    dataset_root: Path,
    groupingIouThreshold: float,
    matchIouThreshold: float,
    gtRoot: str,
) -> list[FrameErrorCounts]:
    if len(modelNames) != 2:
        raise ValueError("jaccard_error_relationは2モデルの比較")
 
    labelsDirForIdList = Path(dataset_root) / "single_model_detection" /dataset / mapName / modelNames[0] / "labels"
    imageIds = getImageIdList(str(labelsDirForIdList))
 
    frameRecords: list[FrameErrorCounts] = []
    for imageId in imageIds:
        detectionModelDict = buildDetectionModelDict(
            dataset, mapName, modelNames, imageId, dataset_root
        )
    
        gtFilePath = Path(gtRoot) / dataset / "tracking" / mapName / "labels" / f"{imageId}.txt"
        groundTruthBoxes = loadGroundTruthFile(str(gtFilePath))
        counts = classifyFrame(
            imageId=imageId,
            detectionModelDict=detectionModelDict,
            groundTruthBoxes=groundTruthBoxes,
            groupingIouThreshold=groupingIouThreshold,
            matchIouThreshold=matchIouThreshold,
        )
        frameRecords.append(counts)
 
    return frameRecords
 
 
def saveFrameRecordsToCsv(frameRecords: list[FrameErrorCounts], outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imageId", "jaccard",
            "totalGtCount", "commonFpCount", "commonFnCount",
            "oneModelFnCount", "oneModelFpCount",
            "commonTpCount",
            "maxErrorCount", "minErrorCount",
            "maxErrorRate", "minErrorRate",
        ])
        for r in frameRecords:
            writer.writerow([
                r.imageId, f"{r.jaccard:.6f}",
                r.totalGtCount, r.commonFpCount, r.commonFnCount,
                r.oneModelFnCount, r.oneModelFpCount,
                r.commonTpCount,
                r.maxErrorCount, r.minErrorCount,
                f"{r.maxErrorRate:.6f}", f"{r.minErrorRate:.6f}",
            ])
 
 
# ------------------------------------------------------------
# 可視化(平均化せず、フレームごとの値をそのままjaccardに対してプロットする)
# ------------------------------------------------------------
 
# プロット対象の項目名 -> グラフのy軸ラベル
PLOT_METRICS: dict[str, str] = {
    "maxErrorRate": "max error rate",
    "minErrorRate": "min error rate",
    "commonFpCount": "common FP count",
    "commonFnCount": "common FN count",
    "oneModelFnCount": "one-model FN count",
    "oneModelFpCount": "one-model FP count",
}
 
 
def binJaccard(jaccard: float, binWidth: float = 0.1) -> float:
    """
    jaccardをbinWidth刻みのbinに丸め、bin下端の値を返す
    例: binWidth=0.1のとき 0.34 -> 0.3, 1.0 -> 0.9 (最後のbinは[0.9, 1.0]を含む)
    """
    numBins = max(round(1.0 / binWidth), 1)
    binIdx = min(int(jaccard / binWidth), numBins - 1)
    return round(binIdx * binWidth, 10)
 
 
def plotJaccardVsMetrics(
    frameRecords: list[FrameErrorCounts],
    outputDir: Path,
    metricNames: list[str] | None = None,
    binWidth: float = 0.1,
) -> None:
    """
    jaccardを横軸、各項目を縦軸にしたグラフを保存する
    各項目につき
      ・散布図(binごとにx方向へ少しジッターさせた全フレームの生データ)
      ・binごとにグループ化した箱ひげ図
    を1枚にまとめて保存する
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random as _random
 
    if not frameRecords:
        print("frameRecordsが空のためプロットをスキップします")
        return
 
    if metricNames is None:
        metricNames = list(PLOT_METRICS.keys())
 
    outputDir.mkdir(parents=True, exist_ok=True)
 
    jaccards = [r.jaccard for r in frameRecords]
    binnedJaccards = [binJaccard(j, binWidth) for j in jaccards]
 
    rng = _random.Random(0)
 
    for metricName in metricNames:
        label = PLOT_METRICS.get(metricName, metricName)
        values = [getattr(r, metricName) for r in frameRecords]
 
        # NaN(GT・検出がともに無いフレームなど)は除外する
        pairs = [
            (b, v) for b, v in zip(binnedJaccards, values) if v == v
        ]
        if not pairs:
            continue
 
        uniqueBins = sorted(set(b for b, _ in pairs))
        groupedValues = [
            [v for b, v in pairs if b == ub] for ub in uniqueBins
        ]
 
        binLabels = [f"{ub:.1f}-{ub + binWidth:.1f}" for ub in uniqueBins]
 
        # 散布図はbin中心を基準に少しジッターさせて重なりを見やすくする
        jitterWidth = binWidth * 0.3
        scatterX = [
            ub + binWidth / 2 + rng.uniform(-jitterWidth, jitterWidth)
            for b, v in pairs
            for ub in [b]
        ]
        scatterY = [v for _, v in pairs]
 
        fig, (axScatter, axBox) = plt.subplots(1, 2, figsize=(11, 4.5))
 
        axScatter.scatter(scatterX, scatterY, alpha=0.35, s=15)
        axScatter.set_xlabel(f"jaccard (bin width={binWidth})")
        axScatter.set_ylabel(label)
        axScatter.set_title(f"{label} vs jaccard (all frames)")
        axScatter.set_xticks([ub + binWidth / 2 for ub in uniqueBins])
        axScatter.set_xticklabels(binLabels, rotation=45, ha="right")
        axScatter.set_xlim(-binWidth * 0.2, 1.0 + binWidth * 0.2)
 
        axBox.boxplot(
            groupedValues,
            positions=[ub + binWidth / 2 for ub in uniqueBins],
            widths=binWidth * 0.7,
        )
        axBox.set_xlabel(f"jaccard (bin width={binWidth})")
        axBox.set_ylabel(label)
        axBox.set_title(f"{label} by jaccard bin (boxplot)")
        axBox.set_xticks([ub + binWidth / 2 for ub in uniqueBins])
        axBox.set_xticklabels(binLabels, rotation=45, ha="right")
        axBox.set_xlim(-binWidth * 0.2, 1.0 + binWidth * 0.2)
 
        fig.tight_layout()
        savePath = outputDir / f"{metricName}_vs_jaccard.png"
        fig.savefig(savePath, dpi=150)
        plt.close(fig)
        print(f"プロットを保存しました: {savePath}")
 
 
def loadFrameRecordsFromCsv(csvPath: Path) -> list[FrameErrorCounts]:
    """
    saveFrameRecordsToCsvで保存したCSVからFrameErrorCountsを復元する
    (再計算済みの結果からプロットだけをやり直したい場合に使う)
    """
    frameRecords: list[FrameErrorCounts] = []
    with open(csvPath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frameRecords.append(FrameErrorCounts(
                imageId=row["imageId"],
                jaccard=float(row["jaccard"]),
                totalGtCount=int(row["totalGtCount"]),
                commonFpCount=int(row["commonFpCount"]),
                commonFnCount=int(row["commonFnCount"]),
                oneModelFnCount=int(row["oneModelFnCount"]),
                oneModelFpCount=int(row["oneModelFpCount"]),
            ))
    return frameRecords
 
 
def summarize(frameRecords: list[FrameErrorCounts]) -> dict:
 
    """データセット全体で件数を合算し、集計版のmax/min error rateを求める"""
    totalGt = sum(r.totalGtCount for r in frameRecords)
    commonFp = sum(r.commonFpCount for r in frameRecords)
    commonFn = sum(r.commonFnCount for r in frameRecords)
    oneModelFn = sum(r.oneModelFnCount for r in frameRecords)
    oneModelFp = sum(r.oneModelFpCount for r in frameRecords)
    commonTpCount = sum(r.commonTpCount for r in frameRecords)
 
    maxErrorCount = commonFp + commonFn + oneModelFn + oneModelFp
    minErrorCount = commonFp + commonFn
    totalUnit = totalGt + commonFp + oneModelFp
 
    meanJaccard = (
        sum(r.jaccard for r in frameRecords) / len(frameRecords)
        if frameRecords else float("nan")
    )
 
    return {
        "meanJaccard": meanJaccard,
        "totalGtCount": totalGt,
        "commonFpCount": commonFp,
        "commonFnCount": commonFn,
        "oneModelFnCount": oneModelFn,
        "oneModelFpCount": oneModelFp,
        "maxErrorCount": maxErrorCount,
        "minErrorCount": minErrorCount,
        "commonTpCount": commonTpCount,
        "maxErrorRate": maxErrorCount / totalUnit if totalUnit > 0 else float("nan"),
        "minErrorRate": minErrorCount / totalUnit if totalUnit > 0 else float("nan"),
    }
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jaccard係数とエラー率(最大値/最小値)の関係を分析")
    parser.add_argument("--dataset", required=True, help="例: kitti")
    parser.add_argument("--map", required=True, dest="map", help="マップ名")
    parser.add_argument(
        "--models", required=True, nargs=2,
        help="比較する2モデルの名前 (例: --models yolov11n fasterrcnn)",
    )
    parser.add_argument("--gt-root", default="/mnt/d", dest="gtRoot")
 
    args = parser.parse_args()
 
    print(f"models: {args.models}, dataset: {args.dataset}, map: {args.map}")
 
    frameRecords = run(
        dataset=args.dataset,
        mapName=args.map,
        modelNames=args.models,
        dataset_root=DATASET_DIR,
        groupingIouThreshold=IOU_THRESHOLD,
        matchIouThreshold=IOU_THRESHOLD,
        gtRoot=args.gtRoot,
    )
 
    summary = summarize(frameRecords)
    print(
        f"mean jaccard={summary['meanJaccard']:.4f}  "
        f"max_error_rate={summary['maxErrorRate']:.4f}  "
        f"min_error_rate={summary['minErrorRate']:.4f}"
    )
 
    model_key = "_".join(sorted(args.models))
    out_dir = RESULT_DIR / "jaccard_error_relation"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    per_frame_csv = out_dir / f"{model_key}_{args.dataset}_{args.map}.csv"
    saveFrameRecordsToCsv(frameRecords, per_frame_csv)
 
    plot_dir = out_dir / "plots" / f"{model_key}_{args.dataset}_{args.map}"
    plotJaccardVsMetrics(frameRecords, plot_dir)
 
    summary_path = out_dir / "summary.csv"
 
    summary_fieldnames = [
        "timestamp", "models", "dataset", "map",
        "mean_jaccard",
        "common_fp", "common_fn", "one_model_fn", "one_model_fp",
        "max_error_count", "min_error_count",
        "max_error_rate", "min_error_rate",
        "common_tp"
    ]
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=summary_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": model_key,
            "dataset": args.dataset,
            "map": args.map,
            "mean_jaccard": summary["meanJaccard"],
            "common_fp": summary["commonFpCount"],
            "common_fn": summary["commonFnCount"],
            "one_model_fn": summary["oneModelFnCount"],
            "one_model_fp": summary["oneModelFpCount"],
            "max_error_count": summary["maxErrorCount"],
            "min_error_count": summary["minErrorCount"],
            "max_error_rate": summary["maxErrorRate"],
            "min_error_rate": summary["minErrorRate"],
            "common_tp": summary["commonTpCount"]
        })
 
    print(f"Per-frame records saved: {per_frame_csv}")
    print(f"Summary appended: {summary_path}")