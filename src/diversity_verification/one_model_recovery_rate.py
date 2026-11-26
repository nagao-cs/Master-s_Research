"""
M1に対するM2のFP/FN改善率を評価する。

概要:
1. M1の検出結果とGTを比較し、M1のFP/FNを特定する
2. M1の各FP/FNについて、M2がそのエラーを解消するか判定する
3. フレームごとにFP/FNのRecovery Rateを計算する
4. 全フレームを合算して、
      - FP Recovery Rate
      - FN Recovery Rate
   を算出する
5. 必要に応じてJaccardとRecovery Rateの関係も分析する

定義:
------------------------------------------------------------
M1 FN:
    M1が検出できなかったGT物体について、
    M2が同じ物体(class一致, IoU>=threshold)を検出
    -> M2によってFNが解消

M1 FP:
    M1の検出のうちGTと一致しなかったものについて、
    M2が同じ誤検出(class一致, IoU>=threshold)を検出しない
    -> M2によってFPが解消
------------------------------------------------------------

注意:
- 1フレーム内のM1 detectionとGTの対応付けは1対1
- M2による解消判定ではclass一致 + IoU閾値を使用
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from src.boundingBox.boundingBox import (
    BoundingBox,
    DetectionBoundingBox,
    GroundTruthBoundingBox,
)
from src.diversity_verification.agree_error_relation import (
    loadDetectionFile,
    loadGroundTruthFile,
    getImageIdList,
)
from src.config import DATASET_DIR, IOU_THRESHOLD, KITTI_ROOT, RESULT_DIR


# ============================================================
# M1のエラー
# ============================================================

@dataclass
class M1ErrorInstance:
    """
    M1で発生した1つのエラー。

    kind:
        "FN" = M1が見逃したGT
        "FP" = M1の誤検出

    referenceBox:
        M2との対応付けに使用する基準bbox
    """

    kind: str
    referenceBox: BoundingBox
    classId: int


# ============================================================
# M1のFP/FNを抽出
# ============================================================

def classifyM1Errors(
    m1Boxes: list[DetectionBoundingBox],
    groundTruthBoxes: list[GroundTruthBoundingBox],
    matchIouThreshold: float,
) -> list[M1ErrorInstance]:
    """
    M1の検出結果とGTを比較し、M1のFP/FNを抽出する。

    対応付け:
        - 同一class
        - IoU >= matchIouThreshold
        - confidenceの高い検出から1対1でGTに対応

    Returns:
        M1ErrorInstanceのリスト
    """

    # confidenceの高い順に処理
    orderedDetectionIndices = sorted(
        range(len(m1Boxes)),
        key=lambda i: m1Boxes[i].confidenceScore,
        reverse=True,
    )

    matchedGtIndices: set[int] = set()
    matchedDetectionIndices: set[int] = set()

    for detIdx in orderedDetectionIndices:
        detection = m1Boxes[detIdx]

        bestIou = 0.0
        bestGtIdx = -1

        for gtIdx, gtBox in enumerate(groundTruthBoxes):
            if gtIdx in matchedGtIndices:
                continue

            # classが異なるものは対応させない
            if detection.classId != gtBox.classId:
                continue

            iou = detection.computeIoU(gtBox)

            if (
                iou >= matchIouThreshold
                and iou > bestIou
            ):
                bestIou = iou
                bestGtIdx = gtIdx

        if bestGtIdx >= 0:
            matchedGtIndices.add(bestGtIdx)
            matchedDetectionIndices.add(detIdx)

    errors: list[M1ErrorInstance] = []

    # --------------------------------------------------------
    # M1のFP
    # GTと対応しなかったM1 detection
    # --------------------------------------------------------

    for detIdx, detection in enumerate(m1Boxes):
        if detIdx in matchedDetectionIndices:
            continue

        errors.append(
            M1ErrorInstance(
                kind="FP",
                referenceBox=detection,
                classId=detection.classId,
            )
        )

    # --------------------------------------------------------
    # M1のFN
    # M1が対応できなかったGT
    # --------------------------------------------------------

    for gtIdx, gtBox in enumerate(groundTruthBoxes):
        if gtIdx in matchedGtIndices:
            continue

        errors.append(
            M1ErrorInstance(
                kind="FN",
                referenceBox=gtBox,
                classId=gtBox.classId,
            )
        )

    return errors


# ============================================================
# M2によるエラー解消判定
# ============================================================

def isResolvedByM2(
    instance: M1ErrorInstance,
    m2Boxes: list[DetectionBoundingBox],
    matchIouThreshold: float,
) -> bool:
    """
    M1のエラーをM2が解消できるか判定する。

    FN:
        M2が同じGT物体を検出
        -> 解消

    FP:
        M2が同じ誤検出をしない
        -> 解消

    class一致 + IoU>=threshold
    """

    matchedByM2 = any(
        box.classId == instance.classId
        and box.computeIoU(instance.referenceBox) >= matchIouThreshold
        for box in m2Boxes
    )

    if instance.kind == "FN":
        return matchedByM2

    if instance.kind == "FP":
        return not matchedByM2

    raise ValueError(f"Unknown error kind: {instance.kind}")


# ============================================================
# フレーム単位の記録
# ============================================================

@dataclass
class FrameRecoveryRecord:
    imageId: str

    totalFp: int
    resolvedFp: int

    totalFn: int
    resolvedFn: int

    @property
    def recoveryRateFp(self) -> float:
        if self.totalFp == 0:
            return float("nan")

        return self.resolvedFp / self.totalFp

    @property
    def recoveryRateFn(self) -> float:
        if self.totalFn == 0:
            return float("nan")

        return self.resolvedFn / self.totalFn


# ============================================================
# 1フレーム処理
# ============================================================

def processFrame(
    dataset: str,
    mapName: str,
    model1: str,
    model2: str,
    imageId: str,
    datasetRoot: Path,
    gtRoot: str,
    matchIouThreshold: float,
) -> FrameRecoveryRecord:

    # --------------------------------------------------------
    # M1
    # --------------------------------------------------------

    m1Path = (
        datasetRoot
        / "single_model_detection"
        / dataset
        / mapName
        / model1
        / "labels"
        / f"{imageId}.txt"
    )

    m1Boxes = loadDetectionFile(str(m1Path))

    # --------------------------------------------------------
    # M2
    # --------------------------------------------------------

    m2Path = (
        datasetRoot
        / "single_model_detection"
        / dataset
        / mapName
        / model2
        / "labels"
        / f"{imageId}.txt"
    )

    m2Boxes = loadDetectionFile(str(m2Path))

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gtPath = (
        Path(gtRoot)
        / "tracking"
        / mapName
        / "labels"
        / f"{imageId}.txt"
    )

    groundTruthBoxes = loadGroundTruthFile(str(gtPath))

    # --------------------------------------------------------
    # M1のFP/FNを取得
    # --------------------------------------------------------

    m1Errors = classifyM1Errors(
        m1Boxes=m1Boxes,
        groundTruthBoxes=groundTruthBoxes,
        matchIouThreshold=matchIouThreshold,
    )

    totalFp = 0
    resolvedFp = 0

    totalFn = 0
    resolvedFn = 0

    # --------------------------------------------------------
    # M2による解消判定
    # --------------------------------------------------------

    for error in m1Errors:

        resolved = isResolvedByM2(
            instance=error,
            m2Boxes=m2Boxes,
            matchIouThreshold=matchIouThreshold,
        )

        if error.kind == "FP":

            totalFp += 1

            if resolved:
                resolvedFp += 1

        elif error.kind == "FN":

            totalFn += 1

            if resolved:
                resolvedFn += 1

    return FrameRecoveryRecord(
        imageId=imageId,
        totalFp=totalFp,
        resolvedFp=resolvedFp,
        totalFn=totalFn,
        resolvedFn=resolvedFn,
    )


# ============================================================
# 全フレーム実行
# ============================================================

def run(
    dataset: str,
    mapName: str,
    model1: str,
    model2: str,
    datasetRoot: Path,
    gtRoot: str,
    matchIouThreshold: float,
) -> list[FrameRecoveryRecord]:

    if model1 == model2:
        raise ValueError("model1とmodel2は異なるモデルを指定してください")

    labelsDir = (
        datasetRoot
        / "single_model_detection"
        / dataset
        / mapName
        / model1
        / "labels"
    )

    imageIds = getImageIdList(str(labelsDir))

    frameRecords: list[FrameRecoveryRecord] = []

    for imageId in imageIds:

        record = processFrame(
            dataset=dataset,
            mapName=mapName,
            model1=model1,
            model2=model2,
            imageId=imageId,
            datasetRoot=datasetRoot,
            gtRoot=gtRoot,
            matchIouThreshold=matchIouThreshold,
        )

        frameRecords.append(record)

    return frameRecords


# ============================================================
# CSV保存
# ============================================================

def saveFrameRecordsToCsv(
    frameRecords: list[FrameRecoveryRecord],
    outputPath: Path,
) -> None:

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    with open(outputPath, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "imageId",

            "totalFp",
            "resolvedFp",
            "recoveryRateFp",

            "totalFn",
            "resolvedFn",
            "recoveryRateFn",
        ])

        for record in frameRecords:

            writer.writerow([
                record.imageId,

                record.totalFp,
                record.resolvedFp,
                (
                    f"{record.recoveryRateFp:.6f}"
                    if not math.isnan(record.recoveryRateFp)
                    else ""
                ),

                record.totalFn,
                record.resolvedFn,
                (
                    f"{record.recoveryRateFn:.6f}"
                    if not math.isnan(record.recoveryRateFn)
                    else ""
                ),
            ])


# ============================================================
# 集計
# ============================================================

def calculateOverallRecoveryRate(
    frameRecords: list[FrameRecoveryRecord],
) -> dict:

    totalFp = sum(r.totalFp for r in frameRecords)
    resolvedFp = sum(r.resolvedFp for r in frameRecords)

    totalFn = sum(r.totalFn for r in frameRecords)
    resolvedFn = sum(r.resolvedFn for r in frameRecords)

    recoveryRateFp = (
        resolvedFp / totalFp
        if totalFp > 0
        else float("nan")
    )

    recoveryRateFn = (
        resolvedFn / totalFn
        if totalFn > 0
        else float("nan")
    )

    totalErrors = totalFp + totalFn
    resolvedErrors = resolvedFp + resolvedFn

    recoveryRateOverall = (
        resolvedErrors / totalErrors
        if totalErrors > 0
        else float("nan")
    )

    return {
        "total_fp": totalFp,
        "resolved_fp": resolvedFp,
        "recovery_rate_fp": recoveryRateFp,

        "total_fn": totalFn,
        "resolved_fn": resolvedFn,
        "recovery_rate_fn": recoveryRateFn,

        "total_errors": totalErrors,
        "resolved_errors": resolvedErrors,
        "recovery_rate_overall": recoveryRateOverall,
    }


# ============================================================
# Summary CSV
# ============================================================

def saveSummary(
    summary: dict,
    outputPath: Path,
    model1: str,
    model2: str,
    dataset: str,
    mapName: str,
) -> None:

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model1",
        "model2",
        "dataset",
        "map",

        "total_fp",
        "resolved_fp",
        "recovery_rate_fp",

        "total_fn",
        "resolved_fn",
        "recovery_rate_fn",

        "total_errors",
        "resolved_errors",
        "recovery_rate_overall",
    ]

    fileExists = outputPath.exists()

    with open(outputPath, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not fileExists:
            writer.writeheader()

        writer.writerow({
            "model1": model1,
            "model2": model2,
            "dataset": dataset,
            "map": mapName,

            "total_fp": summary["total_fp"],
            "resolved_fp": summary["resolved_fp"],
            "recovery_rate_fp": summary["recovery_rate_fp"],

            "total_fn": summary["total_fn"],
            "resolved_fn": summary["resolved_fn"],
            "recovery_rate_fn": summary["recovery_rate_fn"],

            "total_errors": summary["total_errors"],
            "resolved_errors": summary["resolved_errors"],
            "recovery_rate_overall": summary["recovery_rate_overall"],
        })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="M1に対するM2のFP/FN改善率を評価"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="例: kitti",
    )

    parser.add_argument(
        "--map",
        required=True,
        dest="map",
        help="マップ名",
    )

    parser.add_argument(
        "--model1",
        required=True,
        help="ベースモデルM1",
    )

    parser.add_argument(
        "--model2",
        required=True,
        help="追加モデルM2",
    )

    parser.add_argument(
        "--gt-root",
        default=str(KITTI_ROOT),
        dest="gtRoot",
    )

    args = parser.parse_args()

    print(
        f"M1: {args.model1}, "
        f"M2: {args.model2}, "
        f"dataset: {args.dataset}, "
        f"map: {args.map}"
    )

    frameRecords = run(
        dataset=args.dataset,
        mapName=args.map,
        model1=args.model1,
        model2=args.model2,
        datasetRoot=DATASET_DIR,
        gtRoot=args.gtRoot,
        matchIouThreshold=IOU_THRESHOLD,
    )

    # --------------------------------------------------------
    # 出力
    # --------------------------------------------------------

    outDir = (
        RESULT_DIR
        / "model_recovery_rate"
    )

    outDir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runKey = (
        f"{args.model1}_plus_{args.model2}_"
        f"{args.dataset}_{args.map}"
    )

    perFramePath = (
        outDir
        / f"{runKey}.csv"
    )

    saveFrameRecordsToCsv(
        frameRecords,
        perFramePath,
    )

    summary = calculateOverallRecoveryRate(
        frameRecords
    )

    summaryPath = (
        outDir
        / "summary.csv"
    )

    saveSummary(
        summary=summary,
        outputPath=summaryPath,
        model1=args.model1,
        model2=args.model2,
        dataset=args.dataset,
        mapName=args.map,
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print("========================================")
    print("M1 -> M2 Error Recovery")
    print("========================================")

    print(
        f"FP: "
        f"{summary['resolved_fp']}/"
        f"{summary['total_fp']} "
        f"= "
        f"{summary['recovery_rate_fp']:.4f}"
    )

    print(
        f"FN: "
        f"{summary['resolved_fn']}/"
        f"{summary['total_fn']} "
        f"= "
        f"{summary['recovery_rate_fn']:.4f}"
    )

    print(
        f"Overall: "
        f"{summary['resolved_errors']}/"
        f"{summary['total_errors']} "
        f"= "
        f"{summary['recovery_rate_overall']:.4f}"
    )

    print()
    print(f"Per-frame records saved: {perFramePath}")
    print(f"Summary appended: {summaryPath}")