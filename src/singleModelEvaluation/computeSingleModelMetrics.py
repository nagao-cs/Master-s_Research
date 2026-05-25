import argparse
from pathlib import Path
import os
import csv
from typing import Dict, Tuple
import pandas as pd

from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox, ClassifiedBoundingBox, ClassifyCategory
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.dataset import fileReader
from src.Evaluation.metrics import mAP, f1Score


def compute_metrics_for_model(
    model_name: str,
    map_name: str,
    ground_truth_dir: Path,
    detection_dir: Path,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    単一モデルのmAPとF1スコアを計算
    
    Args:
        model_name: モデル名
        map_name: マップ名
        ground_truth_dir: グラウンドトゥルースディレクトリ
        detection_dir: 検出結果ディレクトリ
        iou_threshold: IoU閾値
    
    Returns:
        メトリクス辞書（mAP, F1, Precision, Recall）
    """
    
    classifiedBoundingBoxList: list[ClassifiedBoundingBox] = list()
    detectionClassifier: DetectionClassifier = DetectionClassifier(
        iouThreshold=iou_threshold)

    # ファイルリストを取得
    groundTruthFileList: list[str] = sorted(os.listdir(ground_truth_dir))
    detectionFileList: list[str] = sorted(os.listdir(detection_dir))

    # 最小のファイル数でループ（ファイル数が異なる場合に対応）
    min_files = min(len(groundTruthFileList), len(detectionFileList))

    for i in range(min_files):
        groudTruthFile = groundTruthFileList[i]
        detectionFile = detectionFileList[i]

        groudTruthFilePath = os.path.join(ground_truth_dir, groudTruthFile)
        detectionFilePath = os.path.join(detection_dir, detectionFile)

        if not os.path.exists(groudTruthFilePath):
            print(f"Warning: Ground truth file not found: {groudTruthFilePath}")
            continue
        if not os.path.exists(detectionFilePath):
            print(f"Warning: Detection file not found: {detectionFilePath}")
            continue

        groundTruthBoundingBoxList: list[GroundTruthBoundingBox] = fileReader.convertGroundTruthFileToBoundingBoxList(
            groudTruthFilePath)
        detectionBoundingBoxList: list[DetectionBoundingBox] = fileReader.convertDetectionFileToBoundingBoxList(
            detectionFilePath)

        classifiedBoundingBoxListPerFrame: list[ClassifiedBoundingBox] = detectionClassifier.classify(
            groundTruthBoundingBoxList, detectionBoundingBoxList)

        classifiedBoundingBoxList.extend(classifiedBoundingBoxListPerFrame)

    # メトリクス計算
    # 0 = pedestrian, 2 = vehicle, 9 = traffic light, 11 = traffic sign
    targetClassIdList = [0, 2, 9, 11]
    
    # mAP計算
    mAPValue, classIdApDict = mAP.computeMeanAP(
        classifiedBoundingBoxList, targetClassIdList)
    
    # F1スコア計算
    f1Value, precisionValue, recallValue = f1Score.computeF1Score(
        classifiedBoundingBoxList)

    return {
        'model': model_name,
        'map': map_name,
        'mAP': mAPValue,
        'F1': f1Value,
        'Precision': precisionValue,
        'Recall': recallValue,
        'AP_pedestrian': classIdApDict.get(0, 0.0),
        'AP_vehicle': classIdApDict.get(2, 0.0),
        'AP_traffic_light': classIdApDict.get(9, 0.0),
        'AP_traffic_sign': classIdApDict.get(11, 0.0),
    }


def compute_all_single_models(
    map_name: str = "Town02",
    iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    すべてのモデルのメトリクスを計算
    
    Args:
        map_name: マップ名
        iou_threshold: IoU閾値
    
    Returns:
        結果データフレーム
    """
    
    cwd = Path(__file__).parent
    baseDir = cwd.parent.parent

    # グラウンドトゥルースディレクトリ
    groundTruthDatasetDir = baseDir / "output" / "label" / f"{map_name}" / "front"
    
    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(f"{groundTruthDatasetDir} does not exist")

    # 単一モデルの検出結果ディレクトリ
    oneVersionDetectionDir = baseDir / "oneVersionDetectionResult" / "labels" / f"{map_name}"
    
    if not os.path.exists(oneVersionDetectionDir):
        raise FileNotFoundError(f"{oneVersionDetectionDir} does not exist")

    # すべてのモデルディレクトリをリストアップ
    model_dirs = [d for d in os.listdir(oneVersionDetectionDir) 
                  if os.path.isdir(os.path.join(oneVersionDetectionDir, d))]
    model_dirs.sort()

    print(f"Found {len(model_dirs)} models: {model_dirs}")

    results = []
    
    for model_name in model_dirs:
        model_detection_dir = oneVersionDetectionDir / model_name
        
        print(f"\nComputing metrics for model: {model_name}")
        try:
            metrics = compute_metrics_for_model(
                model_name=model_name,
                map_name=map_name,
                ground_truth_dir=groundTruthDatasetDir,
                detection_dir=model_detection_dir,
                iou_threshold=iou_threshold
            )
            results.append(metrics)
            print(f"  mAP: {metrics['mAP']:.4f}")
            print(f"  F1:  {metrics['F1']:.4f}")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue

    return pd.DataFrame(results)


def save_results_to_csv(
    results_df: pd.DataFrame,
    output_dir: Path,
    map_name: str
) -> None:
    """
    結果をCSVファイルに保存
    
    Args:
        results_df: 結果データフレーム
        output_dir: 出力ディレクトリ
        map_name: マップ名
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 全メトリクスの詳細
    detailed_csv_path = output_dir / "detailed_metrics.csv"
    results_df.to_csv(detailed_csv_path, index=False)
    print(f"\nDetailed metrics saved to: {detailed_csv_path}")
    
    # 主要メトリクスのサマリー
    summary_df = results_df[['model', 'mAP', 'F1', 'Precision', 'Recall']].copy()
    summary_csv_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary metrics saved to: {summary_csv_path}")
    
    # クラス別APのサマリー
    ap_summary_df = results_df[[
        'model', 'AP_pedestrian', 'AP_vehicle', 
        'AP_traffic_light', 'AP_traffic_sign'
    ]].copy()
    ap_summary_csv_path = output_dir / "ap_by_class_metrics.csv"
    ap_summary_df.to_csv(ap_summary_csv_path, index=False)
    print(f"AP by class metrics saved to: {ap_summary_csv_path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Compute mAP and F1 metrics for single detection models")
    argparser.add_argument(
        "--map",
        type=str,
        default="Town02",
        help="Map name (Town02, Town03, Town05)"
    )
    argparser.add_argument(
        "--iou_th",
        type=float,
        default=0.5,
        help="IoU threshold for matching"
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="singleModelEvaluation/results",
        help="Output directory for results"
    )
    
    args = argparser.parse_args()

    print(f"Computing single model metrics...")
    print(f"Map: {args.map}")
    print(f"IoU Threshold: {args.iou_th}")

    # すべてのモデルのメトリクスを計算
    results_df = compute_all_single_models(
        map_name=args.map,
        iou_threshold=args.iou_th
    )

    # 結果を保存
    output_dir = Path(args.output_dir) / args.map
    save_results_to_csv(results_df, output_dir, args.map)

    # コンソールに結果を表示
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(results_df[['model', 'mAP', 'F1', 'Precision', 'Recall']].to_string(index=False))
    print("="*80)