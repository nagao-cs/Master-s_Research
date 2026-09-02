"""
detector(1モデル)とtracker(SORTベース等)の検出結果を比較し、
    detectionだけに現れる検出(detection-only)にFPが多いか
    trackerだけに現れる検出(tracker-only)が、detectorのFNを埋めているか
を確認する。

背景にある仮説:
    detection-only(detectorにはあるがtrackerには無い)が多い
        -> 誤検出(FP)を抑制する方向のモデル追加が効果的なはず
    tracker-only(trackerにはあるがdetectorには無い)が多く、かつそれがGTと一致する(TP)ことが多い
        -> 見逃し(FN)を抑制する方向のモデル追加が効果的なはず
        (tracker-onlyがGTと一致するというのは、時系列情報を使うtrackerが、
         その1フレームだけを見るdetectorが見逃した物体を拾えている、ということ)

detector, trackerをgroupingBoundingBoxで2"モデル"としてgroupingし、
    グループサイズ2(両者一致) -> diffの対象外(GTマッチングの母数としては消費する)
    グループサイズ1(どちらか一方のみ) -> どちらのモデルが出したboxかを識別し、
                                          detection-only/tracker-onlyに分類する
した上で、各disagreeボックスをGTとマッチングしTP/FPを判定する。

前提:
    agree_error_relation.pyのbuildDetectionModelDictが、
    modelNameに"tracker"を含むかどうかでパスを振り分ける仕様に対応済みであることを前提とする
    (例: --tracker yolov8n_tracker のように、tracker用のモデル名を渡す)
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.boundingBox.boundingBox import BoundingBox, DetectionBoundingBox, GroundTruthBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.diversity_verification.agree_error_relation import (
    loadGroundTruthFile,
    getImageIdList,
    computeRepresentativeBox,
    loadDetectionFile
)
from src.config import DATASET_DIR, IOU_THRESHOLD, KITTI_ROOT, RESULT_DIR

def buildDetectionModelDict(
    dataset: str,
    mapName: str,
    modelNames: list[str],
    imageId: str,
    resultRoot: Path, # /mnt/d/dataset
) -> dict[str, list[DetectionBoundingBox]]:
    """
    1画像分の検出結果を {model_name: [DetectionBoundingBox, ...]} の形で構築する
    groupingBoundingBox はdetectorをdictのkeyとしてのみ扱う(hash/等価判定のみ)ため
    ここではmodel名の文字列をそのままkeyとして利用する
    """
    detectionModelDict: dict[str, list[DetectionBoundingBox]] = {}
    for modelName in modelNames:
        filePath = resultRoot / "tracker" / dataset / mapName / modelName / "labels" / f"{imageId}.txt"
        if not filePath.exists():
            raise FileNotFoundError(f"{filePath} does not exist")
        detectionModelDict[modelName] = loadDetectionFile(str(filePath))
    return detectionModelDict

# ------------------------------------------------------------
# detector/trackerのgroupingからdiff(片方だけの検出)を洗い出す
# ------------------------------------------------------------

@dataclass
class DiffInstance:
    source: str  # "detectionOnly" or "trackerOnly"
    classId: int
    confidenceScore: float
    isTp: bool  # GTと一致したか(True=TP, False=FP)


def buildBoxToModelMap(
    detectionModelDict: dict[str, list[DetectionBoundingBox]]
) -> dict[int, str]:
    """
    box(オブジェクトのid)がどちらのモデル("detector"名 or "tracker"名)由来かを引けるマップを作る
    groupingBoundingBoxはbox参照をそのまま保持して返すため、id(box)をkeyとして利用できる
    """
    boxToModel: dict[int, str] = {}
    for modelName, boxes in detectionModelDict.items():
        for box in boxes:
            boxToModel[id(box)] = modelName
    return boxToModel


def classifyFrameDiff(
    detectorName: str,
    trackerName: str,
    detectionModelDict: dict[str, list[DetectionBoundingBox]],
    groundTruthBoxes: list[GroundTruthBoundingBox],
    groupingIouThreshold: float,
    matchIouThreshold: float,
) -> list[DiffInstance]:
    """
    detector, trackerの検出結果をgroupingし、片方だけのdiff boxをGTとマッチングして返す
    (両者一致のグループはdiffの対象外だが、GTマッチングの母数としては先に消費する)
    """
    boxToModel = buildBoxToModelMap(detectionModelDict)
    groups = groupingBoundingBox(detectionModelDict, groupingIouThreshold)

    representativeBoxes = [computeRepresentativeBox(group) for group in groups]
    # confidence降順に処理し、両者一致の高信頼度グループから先にGTを消費させる
    order = sorted(
        range(len(groups)),
        key=lambda i: representativeBoxes[i].confidenceScore,
        reverse=True,
    )

    matchedGtIndices: set[int] = set()
    diffInstances: list[DiffInstance] = []

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

        isTp = bestGtIdx >= 0
        if isTp:
            matchedGtIndices.add(bestGtIdx)

        if len(group) != 1:
            # 両者一致のグループはdiffの対象外(GTの消費だけ行って次へ)
            continue

        box = group[0]
        modelName = boxToModel[id(box)]
        source = "detectionOnly" if modelName == detectorName else "trackerOnly"

        diffInstances.append(DiffInstance(
            source=source,
            classId=box.classId,
            confidenceScore=getattr(box, "confidenceScore", 0.0),
            isTp=isTp,
        ))

    return diffInstances


# ------------------------------------------------------------
# 集計
# ------------------------------------------------------------

@dataclass
class DiffRecord:
    imageId: str
    detectionOnlyTp: int
    detectionOnlyFp: int
    trackerOnlyTp: int
    trackerOnlyFp: int


@dataclass
class DiffStats:
    detectionOnlyTp: int = 0
    detectionOnlyFp: int = 0
    trackerOnlyTp: int = 0
    trackerOnlyFp: int = 0

    def update(self, inst: DiffInstance) -> None:
        if inst.source == "detectionOnly":
            if inst.isTp:
                self.detectionOnlyTp += 1
            else:
                self.detectionOnlyFp += 1
        else:  # "trackerOnly"
            if inst.isTp:
                self.trackerOnlyTp += 1
            else:
                self.trackerOnlyFp += 1

    @property
    def detectionOnlyTotal(self) -> int:
        return self.detectionOnlyTp + self.detectionOnlyFp

    @property
    def trackerOnlyTotal(self) -> int:
        return self.trackerOnlyTp + self.trackerOnlyFp

    @property
    def detectionOnlyFpRate(self) -> float:
        """
        detection-onlyのうちFP(誤検出)である割合
        高いほど「detectorだけが余計に検出している」= FP抑制方向のモデル追加が効きそう
        """
        total = self.detectionOnlyTotal
        return self.detectionOnlyFp / total if total > 0 else float("nan")

    @property
    def trackerOnlyTpRate(self) -> float:
        """
        tracker-onlyのうちTP(実在物体)である割合
        高いほど「trackerだけがdetectorの見逃しを拾えている」= FN抑制方向のモデル追加が効きそう
        """
        total = self.trackerOnlyTotal
        return self.trackerOnlyTp / total if total > 0 else float("nan")

    def summary(self) -> str:
        lines = [
            f"[detection-only] TP={self.detectionOnlyTp}, FP={self.detectionOnlyFp}, "
            f"FP率={self.detectionOnlyFpRate:.4f}  (高いほどFP抑制方向のモデル追加が有効)",
            f"[tracker-only]   TP={self.trackerOnlyTp}, FP={self.trackerOnlyFp}, "
            f"TP率={self.trackerOnlyTpRate:.4f}  (高いほどFN抑制方向のモデル追加が有効)",
        ]
        return "\n".join(lines)


def processImage(
    detectorName: str,
    trackerName: str,
    dataset: str,
    mapName: str,
    imageId: str,
    groupingIouThreshold: float,
    matchIouThreshold: float,
    resultRoot: Path,
    gtRoot: str,
    stats: DiffStats,
    records: list[DiffRecord],
) -> None:
    detectionModelDict = buildDetectionModelDict(
        dataset, mapName, [detectorName, trackerName], imageId, resultRoot
    )

    # 両方とも検出無しならスキップ
    if all(len(boxes) == 0 for boxes in detectionModelDict.values()):
        return

    gtFilePath = Path(gtRoot) / "tracking" / mapName / "labels" / f"{imageId}.txt"
    if not gtFilePath.exists():
        raise FileNotFoundError(f"{gtFilePath} does not exist")
    groundTruthBoxes = loadGroundTruthFile(str(gtFilePath))

    diffInstances = classifyFrameDiff(
        detectorName, trackerName, detectionModelDict, groundTruthBoxes,
        groupingIouThreshold, matchIouThreshold,
    )

    frameDetOnlyTp = 0
    frameDetOnlyFp = 0
    frameTrkOnlyTp = 0
    frameTrkOnlyFp = 0
    for inst in diffInstances:
        stats.update(inst)
        if inst.source == "detectionOnly":
            if inst.isTp:
                frameDetOnlyTp += 1
            else:
                frameDetOnlyFp += 1
        else:
            if inst.isTp:
                frameTrkOnlyTp += 1
            else:
                frameTrkOnlyFp += 1

    records.append(DiffRecord(
        imageId=imageId,
        detectionOnlyTp=frameDetOnlyTp,
        detectionOnlyFp=frameDetOnlyFp,
        trackerOnlyTp=frameTrkOnlyTp,
        trackerOnlyFp=frameTrkOnlyFp,
    ))


def run(
    detectorName: str,
    trackerName: str,
    dataset: str,
    mapName: str,
    resultRoot: Path,
    groupingIouThreshold: float = 0.5,
    matchIouThreshold: float = 0.5,
    gtRoot: str = "/mnt/d",
) -> tuple[DiffStats, list[DiffRecord]]:
    """
    dataset, mapName配下の全画像についてdetector/trackerのdiffを集計する
    """
    labelsDirForIdList = Path(resultRoot) / "tracker" / dataset / mapName / detectorName / "labels"
    if not labelsDirForIdList.exists():
        raise FileNotFoundError(f"{labelsDirForIdList} does not exist")
    imageIds = getImageIdList(str(labelsDirForIdList))

    stats = DiffStats()
    records: list[DiffRecord] = []

    for imageId in imageIds:
        processImage(
            detectorName=detectorName,
            trackerName=trackerName,
            dataset=dataset,
            mapName=mapName,
            imageId=imageId,
            groupingIouThreshold=groupingIouThreshold,
            matchIouThreshold=matchIouThreshold,
            resultRoot=resultRoot,
            gtRoot=gtRoot,
            stats=stats,
            records=records,
        )

    return stats, records


def saveRecordsToCsv(records: list[DiffRecord], outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imageId", "detectionOnlyTp", "detectionOnlyFp", "trackerOnlyTp", "trackerOnlyFp",
        ])
        for r in records:
            writer.writerow([
                r.imageId, r.detectionOnlyTp, r.detectionOnlyFp, r.trackerOnlyTp, r.trackerOnlyFp,
            ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="detector-only/tracker-onlyの検出を比較し、FP抑制/FN抑制どちら向けの"
                     "モデル追加が効果的かを検証する"
    )
    parser.add_argument("--dataset", required=True, help="例: kitti")
    parser.add_argument("--map", required=True, dest="map", help="マップ名")
    parser.add_argument("--detector", required=True, dest="detector", help="単一検出モデル名")

    args = parser.parse_args()
    tracker = f"{args.detector}_tracker_pred"
    print(f"detector: {args.detector}, tracker: {tracker}, dataset: {args.dataset}, map: {args.map}")

    stats, records = run(
        detectorName=args.detector,
        trackerName=tracker,
        dataset=args.dataset,
        mapName=args.map,
        resultRoot=DATASET_DIR,
        groupingIouThreshold=IOU_THRESHOLD,
        matchIouThreshold=IOU_THRESHOLD,
        gtRoot=KITTI_ROOT,
    )

    print(stats.summary())

    out_dir = RESULT_DIR / "diff_tracker"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_key = f"{args.detector}_vs_{tracker}_{args.dataset}_{args.map}"
    per_frame_csv = out_dir / f"{run_key}.csv"
    saveRecordsToCsv(records, per_frame_csv)

    summary_path = out_dir / "summary.csv"
    summary_fieldnames = [
        "timestamp", "detector", "tracker", "dataset", "map",
        "detection_only_tp", "detection_only_fp", "detection_only_fp_rate",
        "tracker_only_tp", "tracker_only_fp", "tracker_only_tp_rate",
    ]
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=summary_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detector": args.detector,
            "tracker": tracker,
            "dataset": args.dataset,
            "map": args.map,
            "detection_only_tp": stats.detectionOnlyTp,
            "detection_only_fp": stats.detectionOnlyFp,
            "detection_only_fp_rate": stats.detectionOnlyFpRate,
            "tracker_only_tp": stats.trackerOnlyTp,
            "tracker_only_fp": stats.trackerOnlyFp,
            "tracker_only_tp_rate": stats.trackerOnlyTpRate,
        })

    print(f"\nPer-frame records saved: {per_frame_csv}")
    print(f"Summary appended: {summary_path}")

    print("\n=== 仮説判定の目安 ===")
    if stats.detectionOnlyTotal > 0 and stats.trackerOnlyTotal > 0:
        if stats.detectionOnlyTotal > stats.trackerOnlyTotal:
            print(f"detection-onlyの件数({stats.detectionOnlyTotal})がtracker-onlyの件数"
                  f"({stats.trackerOnlyTotal})より多い -> FP抑制方向のモデル追加が効果的な可能性")
        else:
            print(f"tracker-onlyの件数({stats.trackerOnlyTotal})がdetection-onlyの件数"
                  f"({stats.detectionOnlyTotal})より多い -> FN抑制方向のモデル追加が効果的な可能性")
        print(f"detection-onlyのFP率: {stats.detectionOnlyFpRate:.4f}")
        print(f"tracker-onlyのTP率(=detectorのFNをtrackerが埋めている割合): {stats.trackerOnlyTpRate:.4f}")