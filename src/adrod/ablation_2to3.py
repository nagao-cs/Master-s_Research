import argparse
import os
from collections import Counter
from logging import getLogger
from pathlib import Path
from tqdm import tqdm
import time

from ..adaptiveNversion.versionController.VersionController import VersionController, VersionState
from ..boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from ..boundingBox.integrator.majorityIntegrator import MajorityIntegrator

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.boundingBox.groupingBoundingBox import groupingBoundingBox
from src.Evaluation.dataset import fileReader
from src.ObjectDetection.models.FLOPsDict import FLOPs_Dict


class VersionState:
    """バージョン状態の定義"""
    PAIR = "PAIR"          # 2バージョン
    ENSEMBLE = "ENSEMBLE"  # 3バージョン


if __name__ == "__main__":
    # ----------
    # 引数の受付
    # ----------
    argparser = argparse.ArgumentParser(
        description="Ablation Study: 2-to-3 Version Adaptive Detection"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        nargs='+',
        help="Models to use (e.g., yolov8n yolov11n rtdetr)"
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        required=True
    )
    argparser.add_argument(
        "--pair_to_ensemble",
        type=float,
        default=0.5,
        help="Agreement score threshold: PAIR to ENSEMBLE"
    )
    argparser.add_argument(
        "--ensemble_to_pair",
        type=float,
        default=0.7,
        help="Agreement score threshold: ENSEMBLE to PAIR"
    )
    argparser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for bounding box grouping"
    )
    argparser.add_argument(
        "--integrate",
        type=str,
        choices=["affirmative", "consensus", "conf_base"],
        required=True
    )

    args = argparser.parse_args()
    print(args)
    mapName: str = args.map
    modelNameList: list[str] = args.models
    numVersion: int = len(modelNameList)
    integrate_way = args.integrate

    mapName: str = args.map
    modelNameList: list[str] = args.models

    # 最低3つのモデルが必要
    if len(modelNameList) < 3:
        raise ValueError("3つ以上のモデルが必要です (基本2つ + エスカレーション用1つ)")

    IOU_THRESHOLD: float = args.iou
    AGREEMENT_LOWER_THRESHOLD: float = args.pair_to_ensemble
    CONF_THRESHOLD = 0.5

    # -----------
    # キャッシュから検出結果を読み込み
    # -----------
    cwd: Path = Path(__file__).parent
    baseDir: Path = cwd.parent.parent
    detectionBaseDir: Path = baseDir / "oneVersionDetectionResult" / "labels" / mapName

    print("Loading detection results from cached files...")
    detection_cache: dict[str, list[list[DetectionBoundingBox]]] = {}

    for modelName in modelNameList:
        modelDetectionDir = detectionBaseDir / modelName
        if not os.path.exists(modelDetectionDir):
            raise FileNotFoundError(
                f"Detection directory does not exist: {modelDetectionDir}\n"
                f"Available models: {os.listdir(detectionBaseDir)}"
            )

        detectionFiles = sorted([
            f for f in os.listdir(modelDetectionDir)
            if f.endswith('.txt')
        ])

        modelDetections: list[list[DetectionBoundingBox]] = []
        for detectionFile in detectionFiles:
            detectionFilePath = modelDetectionDir / detectionFile
            detectionBBoxList = fileReader.convertDetectionFileToBoundingBoxList(
                str(detectionFilePath)
            )
            modelDetections.append(detectionBBoxList)

        detection_cache[modelName] = modelDetections
        print(f"  Loaded {len(modelDetections)} frames for {modelName}")

    numFrames = len(detection_cache[modelNameList[0]])
    print(f"Total frames: {numFrames}\n")

    # -----------
    # 出力ディレクトリ
    # -----------
    outputLabelDir: Path = baseDir / "adaptiveDetectionResult" / "ablation_2to3" / f"{integrate_way}" / "labels" / \
        f"{mapName}" / f"{'_'.join(modelNameList)}"
    os.makedirs(outputLabelDir, exist_ok=True)

    # ----------
    # 初期化
    # ----------
    logger = getLogger('ultralytics')
    logger.disabled = True

    # -----------
    # 統合と状態管理のためのクラスを管理
    # -----------
    if integrate_way == "affirmative":
        detectionIntegrator: ConfidenceBaseIntegrator = ConfidenceBaseIntegrator(
            iouThreshold=IOU_THRESHOLD, confidenceThreshold=0.0)
    elif integrate_way == "conf_base":
        detectionIntegrator: ConfidenceBaseIntegrator = ConfidenceBaseIntegrator(
            iouThreshold=IOU_THRESHOLD, confidenceThreshold=CONF_THRESHOLD)
    elif integrate_way == "consensus":
        detectionIntegrator = MajorityIntegrator(
            iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)

    exeVersionRecorder: list[VersionState] = []
    totalFLOPs: float = 0.0

    baseModel = modelNameList[0]
    second_model = modelNameList[1]
    third_model = modelNameList[2]

    outputDetectionList: list[list[DetectionBoundingBox]] = []
    version_state_record: list[str] = []
    totalFLOPs: float = 0.0

    # ----------
    # 計測開始
    # ----------
    print("Processing with 2-to-3 version adaptive strategy...")
    start: float = time.time()

    current_state = VersionState.PAIR  # 初期状態は2バージョン

    for frameIdx in tqdm(range(numFrames)):
        current_state = VersionState.PAIR  # 初期状態は2バージョン

        base_detections = detection_cache[baseModel][frameIdx]
        second_detections = detection_cache[second_model][frameIdx]
        third_detections = detection_cache[third_model][frameIdx]

        # ----------
        # PAIR（2バージョン）で検出実行
        # ----------
        pair_dict = {
            baseModel: base_detections,
            second_model: second_detections
        }
        pair_detections, pair_bbox_groups = detectionIntegrator(pair_dict)

        # ----------
        # Agreement Score計算
        # ----------
        numTotalBBox = len(pair_bbox_groups)
        if numTotalBBox == 0:
            # 完全一致（同じ検出）
            agreementScore = 1.0
        else:
            # 2モデル両方で検出された割合
            numAgreedBBox = sum(
                1 for bbox_group in pair_bbox_groups if len(bbox_group) == 2
            )
            agreementScore = numAgreedBBox / numTotalBBox

        # ----------
        # 状態遷移判定
        # ----------
        if current_state == VersionState.PAIR:
            if agreementScore <= AGREEMENT_LOWER_THRESHOLD:
                # 一致度が低いため、3バージョンに遷移
                current_state = VersionState.ENSEMBLE
            # else: PAIR状態継続

        # ----------
        # 検出結果を決定
        # ----------
        if current_state == VersionState.PAIR:
            final_detections = pair_detections
            totalFLOPs += (FLOPs_Dict[baseModel] + FLOPs_Dict[second_model])
        else:  # ENSEMBLE
            ensemble_dict = {
                baseModel: base_detections,
                second_model: second_detections,
                third_model: third_detections
            }
            final_detections, _ = detectionIntegrator(ensemble_dict)
            totalFLOPs += (FLOPs_Dict[baseModel] +
                           FLOPs_Dict[second_model] + FLOPs_Dict[third_model])

        # ----------
        # 記録
        # ----------
        version_state_record.append(current_state)
        outputDetectionList.append(final_detections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"\nTotal processing time: {executionTime:.2f} seconds")

    # -----------
    # 結果を保存
    # -----------
    for idx, detections in enumerate(outputDetectionList):
        output_path = outputLabelDir / f"{idx:06d}.txt"
        with open(output_path, 'w') as f:
            for bbox in detections:
                f.write(
                    f"{bbox.classId} {bbox.xCenter} {bbox.yCenter} "
                    f"{bbox.width} {bbox.height} {bbox.confidenceScore}\n"
                )

    # -----------
    # 統計情報の出力
    # -----------
    state_counter = Counter(version_state_record)
    total_frames = len(version_state_record)

    print(f"\nVersion State Distribution:")
    print(
        f"  PAIR (2-version):     {state_counter[VersionState.PAIR]:6d} frames ({100*state_counter[VersionState.PAIR]/total_frames:.1f}%)")
    print(
        f"  ENSEMBLE (3-version): {state_counter[VersionState.ENSEMBLE]:6d} frames ({100*state_counter[VersionState.ENSEMBLE]/total_frames:.1f}%)")
    print(f"\nTotal computational cost: {totalFLOPs:.2f} GFLOPs")
    print(f"Results saved to: {outputLabelDir}")
