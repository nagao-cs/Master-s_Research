'''
1. 検出結果の読み込み
2. groupingBoundingBoxによる一致/不一致グループの生成
3. GTとの比較によるTP/FP判定
4. 一致/不一致それぞれのエラー率(FP率)の計算

    検出結果: result/one_version_detection/{dataset}/{map_name}/{model_name}/labels/00000.txt ...
              各行 "class xcenter ycenter h w conf"
    GT      : /mnt/d/{dataset}/tracking/{map_name}/labels/00000.txt ...
              各行 "class xcenter ycenter h w" (confなし)

"一致(agree)" / "不一致(disagree)" の定義:
    groupingBoundingBox の出力で複数モデルのboxを含むグループ -> 一致
    単一モデルのboxのみのグループ                             -> 不一致

各グループはグループ内boxの平均座標を代表bboxとし、GTとIoUでマッチングして
TP/FPを判定する。一致/不一致それぞれについてFP率(=エラー率)を集計する。
'''

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path

from src.boundingBox import BoundingBox, DetectionBoundingBox, GroundTruthBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox


# ------------------------------------------------------------
# 検出結果 / GT の読み込み
# ------------------------------------------------------------

def loadDetectionFile(filePath: str) -> list[DetectionBoundingBox]:
    """
    検出結果txtを読み込む
    フォーマット: class xcenter ycenter h w conf
    """
    boxes: list[DetectionBoundingBox] = []
    if not os.path.exists(filePath):
        return boxes

    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            classId = int(float(parts[0]))
            xCenter = float(parts[1])
            yCenter = float(parts[2])
            height = float(parts[3])
            width = float(parts[4])
            confidenceScore = float(parts[5])

            boxes.append(DetectionBoundingBox(
                xCenter=xCenter,
                yCenter=yCenter,
                width=width,
                height=height,
                classId=classId,
                confidenceScore=confidenceScore,
            ))
    return boxes


def loadGroundTruthFile(filePath: str) -> list[GroundTruthBoundingBox]:
    """
    GTのtxtを読み込む
    フォーマット: class xcenter ycenter h w (confなし、検出結果と同じ列順)
    """
    boxes: list[GroundTruthBoundingBox] = []
    if not os.path.exists(filePath):
        return boxes

    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            classId = int(float(parts[0]))
            xCenter = float(parts[1])
            yCenter = float(parts[2])
            height = float(parts[3])
            width = float(parts[4])

            boxes.append(GroundTruthBoundingBox(
                xCenter=xCenter,
                yCenter=yCenter,
                width=width,
                height=height,
                classId=classId,
            ))
    return boxes


def getImageIdList(labelsDir: str) -> list[str]:
    """
    labelsディレクトリから画像ID(ファイル名から拡張子を除いたもの)の一覧を取得する
    """
    files = sorted(glob.glob(os.path.join(labelsDir, "*.txt")))
    return [Path(f).stem for f in files]


def buildDetectionModelDict(
    dataset: str,
    mapName: str,
    modelNames: list[str],
    imageId: str,
    resultRoot: Path,
) -> dict[str, list[DetectionBoundingBox]]:
    """
    1画像分の検出結果を {model_name: [DetectionBoundingBox, ...]} の形で構築する
    groupingBoundingBox はdetectorをdictのkeyとしてのみ扱う(hash/等価判定のみ)ため
    ここではmodel名の文字列をそのままkeyとして利用する
    """
    detectionModelDict: dict[str, list[DetectionBoundingBox]] = {}

    for modelName in modelNames:
        filePath = os.path.join(
            resultRoot, dataset, mapName, modelName, "labels", f"{imageId}.txt"
        )
        detectionModelDict[modelName] = loadDetectionFile(filePath)

    return detectionModelDict


# ------------------------------------------------------------
# グループ判定 / GTとの比較
# ------------------------------------------------------------

def isAgreeGroup(group: list[BoundingBox]) -> bool:
    """
    グループ内boxが複数(=複数モデルが一致)であれば一致、単一なら不一致
    """
    return len(group) > 1


def computeRepresentativeBox(group: list[BoundingBox]) -> DetectionBoundingBox:
    """
    グループ内boxの平均座標を代表bboxとして返す

    classIdはgroupingBoundingBox内のcomputeIoUがclassId不一致で0.0を返すため
    IoU>=thresholdでunionされたグループ内では常に共通であることが保証されている
    """
    n = len(group)
    xCenter = sum(b.xCenter for b in group) / n
    yCenter = sum(b.yCenter for b in group) / n
    width = sum(b.width for b in group) / n
    height = sum(b.height for b in group) / n
    classId = group[0].classId

    confidences = [
        b.confidenceScore for b in group if hasattr(b, "confidenceScore")
    ]
    confidenceScore = sum(confidences) / len(confidences) if confidences else 0.0

    return DetectionBoundingBox(
        xCenter=xCenter,
        yCenter=yCenter,
        width=width,
        height=height,
        classId=classId,
        confidenceScore=confidenceScore,
    )


def matchGroupsToGroundTruth(
    representativeBoxes: list[DetectionBoundingBox],
    groundTruthBoxes: list[GroundTruthBoundingBox],
    iouThreshold: float,
) -> list[bool]:
    """
    代表bboxをGTとマッチングしTP/FPを判定する
    confidence降順に貪欲マッチングし、一度マッチしたGTは以後使用しない
    戻り値はrepresentativeBoxesと同じ順番のTP/FPフラグ(True=TP)
    """
    order = sorted(
        range(len(representativeBoxes)),
        key=lambda i: representativeBoxes[i].confidenceScore,
        reverse=True,
    )

    matchedGtIndices: set[int] = set()
    isTp = [False] * len(representativeBoxes)

    for i in order:
        detBox = representativeBoxes[i]
        bestIou = 0.0
        bestGtIdx = -1

        for gtIdx, gtBox in enumerate(groundTruthBoxes):
            if gtIdx in matchedGtIndices:
                continue
            iou = detBox.computeIoU(gtBox)
            if iou >= iouThreshold and iou > bestIou:
                bestIou = iou
                bestGtIdx = gtIdx

        if bestGtIdx >= 0:
            isTp[i] = True
            matchedGtIndices.add(bestGtIdx)

    return isTp


# ------------------------------------------------------------
# 集計
# ------------------------------------------------------------

@dataclass
class GroupRecord:
    imageId: str
    groupSize: int
    agree: bool
    isTp: bool
    classId: int
    confidenceScore: float


@dataclass
class AgreeErrorStats:
    agreeTp: int = 0
    agreeFp: int = 0
    disagreeTp: int = 0
    disagreeFp: int = 0

    def update(self, agree: bool, isTp: bool) -> None:
        if agree:
            if isTp:
                self.agreeTp += 1
            else:
                self.agreeFp += 1
        else:
            if isTp:
                self.disagreeTp += 1
            else:
                self.disagreeFp += 1

    @property
    def agreeErrorRate(self) -> float:
        total = self.agreeTp + self.agreeFp
        return self.agreeFp / total if total > 0 else float("nan")

    @property
    def disagreeErrorRate(self) -> float:
        total = self.disagreeTp + self.disagreeFp
        return self.disagreeFp / total if total > 0 else float("nan")

    def summary(self) -> str:
        lines = [
            f"[一致 (agree)]     TP={self.agreeTp}, FP={self.agreeFp}, "
            f"エラー率={self.agreeErrorRate:.4f}",
            f"[不一致 (disagree)] TP={self.disagreeTp}, FP={self.disagreeFp}, "
            f"エラー率={self.disagreeErrorRate:.4f}",
        ]
        return "\n".join(lines)


def processImage(
    dataset: str,
    mapName: str,
    modelNames: list[str],
    imageId: str,
    groupingIouThreshold: float,
    matchIouThreshold: float,
    resultRoot: Path,
    gtRoot: str,
    stats: AgreeErrorStats,
    records: list[GroupRecord],
) -> None:
    detectionModelDict = buildDetectionModelDict(
        dataset, mapName, modelNames, imageId, resultRoot
    )

    # 全モデルとも検出無しならスキップ
    if all(len(boxes) == 0 for boxes in detectionModelDict.values()):
        return

    groups = groupingBoundingBox(detectionModelDict, groupingIouThreshold)

    gtFilePath = os.path.join(
        gtRoot, dataset, "tracking", mapName, "labels", f"{imageId}.txt"
    )
    groundTruthBoxes = loadGroundTruthFile(gtFilePath)
    print(groundTruthBoxes)

    representativeBoxes = [computeRepresentativeBox(group) for group in groups]
    isTpFlags = matchGroupsToGroundTruth(
        representativeBoxes, groundTruthBoxes, matchIouThreshold
    )

    for group, repBox, isTp in zip(groups, representativeBoxes, isTpFlags):
        agree = isAgreeGroup(group)
        stats.update(agree=agree, isTp=isTp)
        records.append(GroupRecord(
            imageId=imageId,
            groupSize=len(group),
            agree=agree,
            isTp=isTp,
            classId=repBox.classId,
            confidenceScore=repBox.confidenceScore,
        ))


def run(
    dataset: str,
    mapName: str,
    modelNames: list[str],
    resultRoot: Path,
    groupingIouThreshold: float = 0.5,
    matchIouThreshold: float = 0.5,
    gtRoot: str = "/mnt/d",
) -> tuple[AgreeErrorStats, list[GroupRecord]]:
    """
    dataset, mapName配下の全画像について一致/不一致のエラー率を集計する
    """
    labelsDirForIdList = os.path.join(
        resultRoot, dataset, mapName, modelNames[0], "labels"
    )
    imageIds = getImageIdList(labelsDirForIdList)
    stats = AgreeErrorStats()
    records: list[GroupRecord] = []

    for imageId in imageIds:
        processImage(
            dataset=dataset,
            mapName=mapName,
            modelNames=modelNames,
            imageId=imageId,
            groupingIouThreshold=groupingIouThreshold,
            matchIouThreshold=matchIouThreshold,
            resultRoot=resultRoot,
            gtRoot=gtRoot,
            stats=stats,
            records=records,
        )

    return stats, records


def saveRecordsToCsv(records: list[GroupRecord], outputPath: str) -> None:
    os.makedirs(os.path.dirname(outputPath) or ".", exist_ok=True)
    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["imageId", "groupSize", "agree", "isTp", "classId", "confidenceScore"]
        )
        for r in records:
            writer.writerow(
                [r.imageId, r.groupSize, r.agree, r.isTp, r.classId, r.confidenceScore]
            )

from src.config import KITTI_ROOT, DATASET_DIR, IOU_THRESHOLD
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一致/不一致検出のエラー率比較")
    parser.add_argument("--dataset", required=True, help="例: kitti")
    parser.add_argument("--map", required=True, dest="map", help="マップ名")
    parser.add_argument(
        "--models", required=True, nargs="+",
        help="モデル名のリスト (例: --models yolov11n fasterrcnn retinanet)",
    )
    parser.add_argument("--gt-root", default="/mnt/d", dest="gtRoot")

    args = parser.parse_args()
    
    print(f"model: {args.models}, dataset: {args.dataset}, map: {args.map}")
    result_root = Path("/mnt/d/dataset/single_model_detection/")
    stats, records = run(
        dataset=args.dataset,
        mapName=args.map,
        modelNames=args.models,
        groupingIouThreshold=IOU_THRESHOLD,
        matchIouThreshold=IOU_THRESHOLD,
        resultRoot=result_root,
        gtRoot=args.gtRoot,
    )

    print(stats.summary())
    from datetime import datetime
    from pathlib import Path as _Path

    # summary と per-model ファイル名はモデル名をソートして決定
    model_key = "_".join(sorted(args.models))
    summary_dir = DATASET_DIR / "agree_error_relation"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / "summary.csv"
    summary_fieldnames = [
        "timestamp",
        "models", "dataset", "map",
        "agree_tp", "agree_fp", "agree_error_rate",
        "disagree_tp", "disagree_fp", "disagree_error_rate",
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
            "agree_tp": stats.agreeTp,
            "agree_fp": stats.agreeFp,
            "agree_error_rate": stats.agreeErrorRate,
            "disagree_tp": stats.disagreeTp,
            "disagree_fp": stats.disagreeFp,
            "disagree_error_rate": stats.disagreeErrorRate,
        })

    out_dir = summary_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model_csv = out_dir / f"{model_key}_{args.dataset}_{args.map}.csv"
    saveRecordsToCsv(records, str(per_model_csv))

    print(f"Summary appended: {summary_path}")
    print(f"Per-image records saved: {per_model_csv}")