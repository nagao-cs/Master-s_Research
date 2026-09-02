"""
1. 2モデル(A, B)の検出結果を読み込みgroupingし、
   片方のみのFN/FP(=A,Bの間で意見が割れた"解消候補")を洗い出す
2. 3モデル目(C)の検出結果を読み込み、各解消候補についてCがそれを解消するかどうかを判定する
3. フレームごとに 解消率(Recovery Rate) = 解消できた件数 / 解消候補の総数 を計算する
4. jaccard(A,B)とRecovery Rateの関係を(平均化せず)可視化し、
   相関係数や区間ごとの集計から「Jaccardが低いほど解消率が高いか」を確認する

前提:
    "片方のみのFN/FP" の定義はjaccard_error_relation.pyのoneModelFn/oneModelFpと同じ
    (2モデルのgroupingで size==1 のグループのうち、GTと一致すればFN由来、
     不一致であればFP由来として扱う)

解消(Resolved)の定義:
    FN由来(A,Bのどちらか一方だけがGTと一致する物体を検出できていた場合):
        3モデル目Cが同じ物体(同classId, IoU>=閾値)を検出していれば解消
        (3モデルのうち2モデルが検出でき、多数決で正しく拾える)
    FP由来(A,Bのどちらか一方だけの誤検出で、GTと一致しなかった場合):
        3モデル目Cがその誤検出に追従していなければ解消
        (誤検出が少数派のままなら、多数決要求で正しく棄却できる)
        逆にCも同じ誤検出をした場合は共通誤検出に格上げされるため未解消とする
"""

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.boundingBox.boundingBox import BoundingBox, DetectionBoundingBox, GroundTruthBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.Evaluation.metrics.jaccard import calc_jaccard
from src.diversity_verification.agree_error_relation import (
    loadDetectionFile,
    loadGroundTruthFile,
    getImageIdList,
    buildDetectionModelDict,
    computeRepresentativeBox,
)
from src.diversity_verification.jaccard_error_relation import binJaccard
from src.config import DATASET_DIR, IOU_THRESHOLD, KITTI_ROOT, RESULT_DIR


# ------------------------------------------------------------
# A,Bペアのgroupingから"片方のみのFN/FP"候補を洗い出す
# ------------------------------------------------------------

@dataclass
class UnresolvedInstance:
    kind: str  # "FN" or "FP"
    referenceBox: BoundingBox  # 3モデル目の検出とマッチングする際の基準bbox
    classId: int


def classifyFramePairwise(
    detectionModelDictAB: dict[str, list[DetectionBoundingBox]],
    groundTruthBoxes: list[GroundTruthBoundingBox],
    groupingIouThreshold: float,
    matchIouThreshold: float,
) -> tuple[float, list[UnresolvedInstance]]:
    """
    2モデル(A,B)の検出結果をgroupingし、jaccardと
    片方のみのFN/FP(解消候補)の一覧を返す
    (jaccard_error_relation.classifyFrameと同じマッチング方針)
    """
    groups = groupingBoundingBox(detectionModelDictAB, groupingIouThreshold)
    jaccard = calc_jaccard(groups)

    representativeBoxes = [computeRepresentativeBox(group) for group in groups]
    order = sorted(
        range(len(groups)),
        key=lambda i: representativeBoxes[i].confidenceScore,
        reverse=True,
    )

    matchedGtIndices: set[int] = set()
    unresolved: list[UnresolvedInstance] = []

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
                # 片方だけがGTと一致する物体を検出 -> もう片方のFN(解消候補)
                gtBox = groundTruthBoxes[bestGtIdx]
                unresolved.append(UnresolvedInstance(
                    kind="FN", referenceBox=gtBox, classId=gtBox.classId
                ))
            # len==2は両モデルとも正しく検出(TP) -> 解消候補ではない
        else:
            if len(group) == 1:
                # 片方だけの誤検出(GTと不一致) -> 解消候補
                unresolved.append(UnresolvedInstance(
                    kind="FP", referenceBox=group[0], classId=group[0].classId
                ))
            # len==2で不一致は共通FP
            elif len(group) == 2:
                unresolved.append(UnresolvedInstance(
                    kind="both FP", referenceBox=repBox, classId=group[0].classId
                ))
        # both FN
        for gtIdx, gtBox in enumerate(groundTruthBoxes):
            if gtIdx in matchedGtIndices:
                continue
            unresolved.append(UnresolvedInstance(
                kind="both FN", referenceBox=gtBox, classId=gtBox.classId
            ))

    return jaccard, unresolved


def isResolvedByThirdModel(
    instance: UnresolvedInstance,
    thirdModelBoxes: list[DetectionBoundingBox],
    matchIouThreshold: float,
) -> bool:
    """
    3モデル目の検出結果でinstanceが解消されるかどうかを判定する
    FN: 3モデル目が同じ物体を検出していれば解消(3モデル中2モデルが正しく認識できる)
    FP: 3モデル目もその誤検出に追従していなければ解消(誤検出が多数派にならず棄却できる)
    """
    matchedByThirdModel = any(
        box.classId == instance.classId
        and box.computeIoU(instance.referenceBox) >= matchIouThreshold
        for box in thirdModelBoxes
    )

    if instance.kind == "both FN":
        return matchedByThirdModel
    elif instance.kind == "both FP":  # "FP"
        return not matchedByThirdModel
    else :
        raise ValueError(f"instance: {instance}")


# ------------------------------------------------------------
# フレーム単位の解消率
# ------------------------------------------------------------

@dataclass
class FrameRecoveryRecord:
    imageId: str
    jaccard: float
    totalUnresolved: int
    resolvedCount: int
    totalUnresolvedFn: int = 0
    resolvedFn: int = 0
    totalUnresolvedFp: int = 0
    resolvedFp: int = 0

    @property
    def recoveryRate(self) -> float:
        return (
            self.resolvedCount / self.totalUnresolved
            if self.totalUnresolved > 0
            else float("nan")
        )

    @property
    def recoveryRateFn(self) -> float:
        return (
            self.resolvedFn / self.totalUnresolvedFn
            if self.totalUnresolvedFn > 0
            else float("nan")
        )

    @property
    def recoveryRateFp(self) -> float:
        return (
            self.resolvedFp / self.totalUnresolvedFp
            if self.totalUnresolvedFp > 0
            else float("nan")
        )


def processFrame(
    dataset: str,
    mapName: str,
    modelPair: list[str],
    thirdModel: str,
    imageId: str,
    groupingIouThreshold: float,
    matchIouThreshold: float,
    dataset_root: Path,
    gtRoot: str,
) -> FrameRecoveryRecord | None:
    detectionModelDictAB = buildDetectionModelDict(
        dataset=dataset, mapName=mapName, modelNames=modelPair, imageId=imageId, resultRoot=Path(dataset_root)
    )

    gtFilePath = Path(gtRoot) / "tracking" / mapName / "labels" / f"{imageId}.txt"
    groundTruthBoxes = loadGroundTruthFile(str(gtFilePath))

    jaccard, unresolvedInstances = classifyFramePairwise(
        detectionModelDictAB, groundTruthBoxes, groupingIouThreshold, matchIouThreshold
    )

    if not unresolvedInstances:
        return FrameRecoveryRecord(
            imageId=imageId, jaccard=jaccard, totalUnresolved=0, resolvedCount=0
        )

    thirdModelFilePath = Path(dataset_root) / "single_model_detection" / dataset / mapName / thirdModel / "labels" / f"{imageId}.txt"
    thirdModelBoxes = loadDetectionFile(str(thirdModelFilePath))

    resolvedCount = 0
    totalUnresolvedFn = 0
    resolvedFn = 0
    totalUnresolvedFp = 0
    resolvedFp = 0

    for inst in unresolvedInstances:
        resolved = isResolvedByThirdModel(inst, thirdModelBoxes, matchIouThreshold)
        if resolved:
            resolvedCount += 1

        if inst.kind == "FN":
            totalUnresolvedFn += 1
            if resolved:
                resolvedFn += 1
        else:  # "FP"
            totalUnresolvedFp += 1
            if resolved:
                resolvedFp += 1

    return FrameRecoveryRecord(
        imageId=imageId,
        jaccard=jaccard,
        totalUnresolved=len(unresolvedInstances),
        resolvedCount=resolvedCount,
        totalUnresolvedFn=totalUnresolvedFn,
        resolvedFn=resolvedFn,
        totalUnresolvedFp=totalUnresolvedFp,
        resolvedFp=resolvedFp,
    )


def run(
    dataset: str,
    mapName: str,
    modelPair: list[str],
    thirdModel: str,
    dataset_root: Path,
    groupingIouThreshold: float,
    matchIouThreshold: float,
    gtRoot: str,
) -> list[FrameRecoveryRecord]:
    if len(modelPair) != 2:
        raise ValueError("modelPairは2モデル(A,B)を指定してください")
    if thirdModel in modelPair:
        raise ValueError("thirdModelはmodelPairに含まれない別のモデルを指定してください")

    labelsDirForIdList = Path(dataset_root) /"single_model_detection" / dataset / mapName / modelPair[0] / "labels"
    imageIds = getImageIdList(str(labelsDirForIdList))

    frameRecords: list[FrameRecoveryRecord] = []
    for imageId in imageIds:
        record = processFrame(
            dataset=dataset,
            mapName=mapName,
            modelPair=modelPair,
            thirdModel=thirdModel,
            imageId=imageId,
            groupingIouThreshold=groupingIouThreshold,
            matchIouThreshold=matchIouThreshold,
            dataset_root=dataset_root,
            gtRoot=gtRoot,
        )
        if record is not None:
            frameRecords.append(record)

    return frameRecords


# ------------------------------------------------------------
# 保存 / 集計 / 可視化
# ------------------------------------------------------------

def saveFrameRecordsToCsv(frameRecords: list[FrameRecoveryRecord], outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imageId", "jaccard", "totalUnresolved", "resolvedCount", "recoveryRate",
            "totalUnresolvedFn", "resolvedFn", "recoveryRateFn",
            "totalUnresolvedFp", "resolvedFp", "recoveryRateFp",
        ])
        for r in frameRecords:
            writer.writerow([
                r.imageId, f"{r.jaccard:.6f}", r.totalUnresolved, r.resolvedCount,
                f"{r.recoveryRate:.6f}",
                r.totalUnresolvedFn, r.resolvedFn, f"{r.recoveryRateFn:.6f}",
                r.totalUnresolvedFp, r.resolvedFp, f"{r.recoveryRateFp:.6f}",
            ])


def pearsonCorrelation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")

    meanX = sum(xs) / n
    meanY = sum(ys) / n

    covXY = sum((x - meanX) * (y - meanY) for x, y in zip(xs, ys))
    varX = sum((x - meanX) ** 2 for x in xs)
    varY = sum((y - meanY) ** 2 for y in ys)

    denom = math.sqrt(varX * varY)
    return covXY / denom if denom > 0 else float("nan")


def summarizeByBin(frameRecords: list[FrameRecoveryRecord], binWidth: float) -> list[dict]:
    """
    jaccardをbinWidth刻みでまとめ、bin内の総解消候補数/解消数から集計版recoveryRateを求める
    (フレームごとのrecoveryRateを単純平均するのではなく、件数を合算してから比率を取る)
    """
    bins: dict[float, dict] = {}
    for r in frameRecords:
        if r.totalUnresolved == 0:
            continue
        b = binJaccard(r.jaccard, binWidth)
        entry = bins.setdefault(b, {"totalUnresolved": 0, "resolvedCount": 0, "frameCount": 0})
        entry["totalUnresolved"] += r.totalUnresolved
        entry["resolvedCount"] += r.resolvedCount
        entry["frameCount"] += 1

    rows = []
    for b in sorted(bins.keys()):
        entry = bins[b]
        recoveryRate = (
            entry["resolvedCount"] / entry["totalUnresolved"]
            if entry["totalUnresolved"] > 0
            else float("nan")
        )
        rows.append({
            "jaccardBin": f"{b:.1f}-{b + binWidth:.1f}",
            "binLower": b,
            "frameCount": entry["frameCount"],
            "totalUnresolved": entry["totalUnresolved"],
            "resolvedCount": entry["resolvedCount"],
            "recoveryRate": recoveryRate,
        })
    return rows


def saveBinSummaryToCsv(rows: list[dict], outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "jaccardBin", "binLower", "frameCount", "totalUnresolved", "resolvedCount", "recoveryRate",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plotJaccardVsRecoveryRate(
    frameRecords: list[FrameRecoveryRecord],
    binRows: list[dict],
    outputDir: Path,
    binWidth: float,
) -> None:
    """
    jaccardを横軸、Recovery Rateを縦軸にして
      ・散布図(フレームごとの生データ、binごとに少しジッター)
      ・binごとの箱ひげ図(フレームごとのrecoveryRate分布)
      ・binごとの集計recoveryRate(件数合算)の折れ線
    を1枚にまとめて保存する
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random as _random

    outputDir.mkdir(parents=True, exist_ok=True)
    rng = _random.Random(0)

    validRecords = [r for r in frameRecords if r.totalUnresolved > 0]
    if not validRecords:
        print("解消候補(片方のみのFN/FP)を含むフレームが無いためプロットをスキップします")
        return

    binnedJaccards = [binJaccard(r.jaccard, binWidth) for r in validRecords]
    uniqueBins = sorted(set(binnedJaccards))
    binLabels = [f"{b:.1f}-{b + binWidth:.1f}" for b in uniqueBins]
    binCenters = [b + binWidth / 2 for b in uniqueBins]

    groupedRates = [
        [r.recoveryRate for r, b in zip(validRecords, binnedJaccards) if b == ub]
        for ub in uniqueBins
    ]

    jitterWidth = binWidth * 0.3
    scatterX = [
        b + binWidth / 2 + rng.uniform(-jitterWidth, jitterWidth) for b in binnedJaccards
    ]
    scatterY = [r.recoveryRate for r in validRecords]

    binRateByLower = {row["binLower"]: row["recoveryRate"] for row in binRows}
    aggregateRates = [binRateByLower.get(b, float("nan")) for b in uniqueBins]

    fig, (axScatter, axBox) = plt.subplots(1, 2, figsize=(12, 4.8))

    axScatter.scatter(scatterX, scatterY, alpha=0.3, s=15, label="per-frame recovery rate")
    axScatter.plot(binCenters, aggregateRates, color="tab:red", marker="o", label="bin-aggregated recovery rate")
    axScatter.set_xlabel(f"jaccard (bin width={binWidth})")
    axScatter.set_ylabel("recovery rate")
    axScatter.set_title("recovery rate vs jaccard (all frames)")
    axScatter.set_xticks(binCenters)
    axScatter.set_xticklabels(binLabels, rotation=45, ha="right")
    axScatter.set_ylim(-0.05, 1.05)
    axScatter.legend(fontsize=8)

    axBox.boxplot(groupedRates, positions=binCenters, widths=binWidth * 0.7)
    axBox.set_xlabel(f"jaccard (bin width={binWidth})")
    axBox.set_ylabel("recovery rate")
    axBox.set_title("recovery rate by jaccard bin (boxplot)")
    axBox.set_xticks(binCenters)
    axBox.set_xticklabels(binLabels, rotation=45, ha="right")
    axBox.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    savePath = outputDir / "recoveryRate_vs_jaccard.png"
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"プロットを保存しました: {savePath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3モデル目追加によるFN/FP解消率とjaccardの関係を分析"
    )
    parser.add_argument("--dataset", required=True, help="例: kitti")
    parser.add_argument("--map", required=True, dest="map", help="マップ名")
    parser.add_argument(
        "--pair", required=True, nargs=2, dest="pair",
        help="jaccardを計算する2モデル(A,B) (例: --pair yolov11n fasterrcnn)",
    )
    parser.add_argument(
        "--third", required=True, dest="third",
        help="解消率を評価する3モデル目 (例: --third retinanet)",
    )
    parser.add_argument("--gt-root", default=str(KITTI_ROOT), dest="gtRoot")
    parser.add_argument("--bin-width", type=float, default=0.1, dest="binWidth")

    args = parser.parse_args()

    print(f"pair: {args.pair}, third: {args.third}, dataset: {args.dataset}, map: {args.map}")
    from src.config import DATASET_DIR
    dataset_root = DATASET_DIR

    frameRecords = run(
        dataset=args.dataset,
        mapName=args.map,
        modelPair=args.pair,
        thirdModel=args.third,
        dataset_root=dataset_root,
        groupingIouThreshold=IOU_THRESHOLD,
        matchIouThreshold=IOU_THRESHOLD,
        gtRoot=args.gtRoot,
    )

    pair_key = "_".join(sorted(args.pair))
    out_dir = RESULT_DIR / "recovery_rate_relation"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_key = f"{pair_key}_plus_{args.third}_{args.dataset}_{args.map}"

    per_frame_csv = out_dir / f"{run_key}.csv"
    saveFrameRecordsToCsv(frameRecords, per_frame_csv)

    binRows = summarizeByBin(frameRecords, args.binWidth)
    bin_csv = out_dir / f"{run_key}_by_bin.csv"
    saveBinSummaryToCsv(binRows, bin_csv)

    plot_dir = out_dir / "plots" / run_key
    plotJaccardVsRecoveryRate(frameRecords, binRows, plot_dir, args.binWidth)

    validRecords = [record for record in frameRecords if record.totalUnresolved > 0]
    corr = pearsonCorrelation(
        [record.jaccard for record in validRecords], [record.recoveryRate for record in validRecords]
    )

    totalUnresolved = sum(record.totalUnresolved for record in frameRecords)
    totalResolved = sum(record.resolvedCount for record in frameRecords)
    overallRecoveryRate = totalResolved / totalUnresolved if totalUnresolved > 0 else float("nan")

    totalUnresolvedFn = sum(record.totalUnresolvedFn for record in frameRecords)
    totalResolvedFn = sum(record.resolvedFn for record in frameRecords)
    overallRecoveryRateFn = totalResolvedFn / totalUnresolvedFn if totalUnresolvedFn > 0 else float("nan")

    totalUnresolvedFp = sum(record.totalUnresolvedFp for record in frameRecords)
    totalResolvedFp = sum(record.resolvedFp for record in frameRecords)
    overallRecoveryRateFp = totalResolvedFp / totalUnresolvedFp if totalUnresolvedFp > 0 else float("nan")

    print(f"解消候補の総数(片方のみのFN/FP) = {totalUnresolved}")
    print(f"3モデル目により解消した件数     = {totalResolved}")
    print(f"全体のRecovery Rate            = {overallRecoveryRate:.4f}")
    print(f"  FN由来: unresolved={totalUnresolvedFn}, resolved={totalResolvedFn}, rate={overallRecoveryRateFn:.4f}")
    print(f"  FP由来: unresolved={totalUnresolvedFp}, resolved={totalResolvedFp}, rate={overallRecoveryRateFp:.4f}")
    print(f"jaccardとrecovery rateの相関係数 = {corr:.4f}")
    if corr == corr:  # not NaN
        if corr < -0.1:
            print("-> jaccardが低いほどrecovery rateが高い傾向(負の相関)を確認")
        elif corr > 0.1:
            print("-> jaccardが低いほどrecovery rateが高い、という傾向とは逆(正の相関)")
        else:
            print("-> 明確な相関は見られない")

    summary_path = out_dir / "summary.csv"
    summary_fieldnames = [
        "timestamp", "pair", "third", "dataset", "map",
        "total_unresolved", "total_resolved", "overall_recovery_rate",
        "total_unresolved_fn", "total_resolved_fn", "overall_recovery_rate_fn",
        "total_unresolved_fp", "total_resolved_fp", "overall_recovery_rate_fp",
        "jaccard_recovery_correlation",
    ]
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=summary_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pair": pair_key,
            "third": args.third,
            "dataset": args.dataset,
            "map": args.map,
            "total_unresolved": totalUnresolved,
            "total_resolved": totalResolved,
            "overall_recovery_rate": overallRecoveryRate,
            "total_unresolved_fn": totalUnresolvedFn,
            "total_resolved_fn": totalResolvedFn,
            "overall_recovery_rate_fn": overallRecoveryRateFn,
            "total_unresolved_fp": totalUnresolvedFp,
            "total_resolved_fp": totalResolvedFp,
            "overall_recovery_rate_fp": overallRecoveryRateFp,
            "jaccard_recovery_correlation": corr,
        })

    print(f"Per-frame records saved: {per_frame_csv}")
    print(f"Bin summary saved: {bin_csv}")
    print(f"Summary appended: {summary_path}")