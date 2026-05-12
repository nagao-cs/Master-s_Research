import argparse
from pathlib import Path
import os
import csv
from datetime import datetime

from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox, ClassifiedBoundingBox, ClassifyCategory
from .classifier.detectionClassifier import DetectionClassifier
from .dataset import fileReader
from .metrics import f1Score

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="compute F1 score with varying model combinations")
    argparser.add_argument(
        "--iou_th",
        type=float,
        default=0.5,
        help="IoU threshold"
    )
    argparser.add_argument(
        "--map",
        type=str,
        default="Town02",
    )
    argparser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Fixed model 1 (e.g., yolov8n)"
    )
    argparser.add_argument(
        "--model2_list",
        type=str,
        nargs="+",
        required=True,
        help="List of models to vary as model 2 (e.g., yolov11n yolov5n yolov8l)"
    )
    argparser.add_argument(
        "--model3",
        type=str,
        required=False,
        help="Fixed model 3 (e.g., rtdetr)"
    )
    argparser.add_argument(
        "--is_adaptive",
        type=str,
        required=True,
        choices=['True', 'False']
    )
    argparser.add_argument(
        "--integrate",
        type=str,
        nargs="+",
        default=["affirmative", "consensus", "conf_base"],
        help="Integration methods"
    )
    argparser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    args = argparser.parse_args()
    print(args)

    mapName = args.map
    iouThreshold = args.iou_th
    integrate_ways = args.integrate
    is_adaptive = (args.is_adaptive == 'True')
    
    model1 = args.model1
    model2_list = args.model2_list
    model3 = args.model3 if args.model3 else ""

    cwd = Path(__file__).parent
    baseDir = cwd.parent.parent

    groundTruthDatasetDir = baseDir / "output" / "label" / f"{mapName}" / "front"
    
    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(f"{groundTruthDatasetDir} does not exist")

    # 出力ファイル名を設定
    if args.output_csv is None:
        output_csv_path = baseDir / "result" / f"F1_results_{model1}_*_{model3}.csv"
    else:
        output_csv_path = Path(args.output_csv)
    
    os.makedirs(output_csv_path.parent, exist_ok=True)

    # 結果を記録するリスト
    results = []

    # 全組み合わせをテスト
    detectionClassifier = DetectionClassifier(iouThreshold=iouThreshold)
    
    for integrate_way in integrate_ways:
        for model2 in model2_list:
            modelNameList = [model1, model2, model3]
            if model3:
                targetModelCombination = f"{'_'.join(modelNameList)}"
            else:
                targetModelCombination = f"{'_'.join([model1, model2])}"

            # 検出結果ディレクトリを決定
            if is_adaptive:
                detectionDatasetDir = baseDir / "adaptiveDetectionResult" / "escalation" / "labels" / \
                    f"{mapName}" / f"{targetModelCombination}" / f"{integrate_way}"
            else:  # 常にNバージョンの場合
                detectionDatasetDir = baseDir / "NversionDetectionResult" / "labels" / \
                    f"{mapName}" / f"{targetModelCombination}" / f"{integrate_way}"

            # ディレクトリが存在しない場合はスキップ
            if not os.path.exists(detectionDatasetDir):
                print(f"⊘ Skipping (not found): {detectionDatasetDir}")
                results.append({
                    'map': mapName,
                    'model1': model1,
                    'model2': model2,
                    'model3': model3,
                    'models': targetModelCombination,
                    'integrate': integrate_way,
                    'iou_threshold': iouThreshold,
                    'adaptive': is_adaptive,
                    'f1_score': 'N/A',
                    'precision': 'N/A',
                    'recall': 'N/A',
                    'status': 'SKIPPED'
                })
                continue

            classifiedBoundingBoxList = []

            groundTruthFileList = sorted(os.listdir(groundTruthDatasetDir))
            detectionFileList = sorted(os.listdir(detectionDatasetDir))

            try:
                for groudTruthFile, detectionFile in zip(groundTruthFileList, detectionFileList):
                    groudTruthFilePath = os.path.join(groundTruthDatasetDir, groudTruthFile)
                    detectionFilePath = os.path.join(detectionDatasetDir, detectionFile)

                    if not os.path.exists(groudTruthFilePath) or not os.path.exists(detectionFilePath):
                        continue

                    groundTruthBoundingBoxList = fileReader.convertGroundTruthFileToBoundingBoxList(
                        groudTruthFilePath)
                    detectionBoundingBoxList = fileReader.convertDetectionFileToBoundingBoxList(
                        detectionFilePath)

                    classifiedBoundingBoxListPerFrame = detectionClassifier.classify(
                        groundTruthBoundingBoxList, detectionBoundingBoxList)

                    classifiedBoundingBoxList.extend(classifiedBoundingBoxListPerFrame)

                # F1 スコア計算
                f1, precision, recall = f1Score.computeF1Score(classifiedBoundingBoxList)

                results.append({
                    'map': mapName,
                    'model1': model1,
                    'model2': model2,
                    'model3': model3,
                    'models': targetModelCombination,
                    'integrate': integrate_way,
                    'iou_threshold': iouThreshold,
                    'adaptive': is_adaptive,
                    'f1_score': f"{f1:.4f}",
                    'precision': f"{precision:.4f}",
                    'recall': f"{recall:.4f}",
                    'status': 'SUCCESS'
                })

                print(f"✓ {targetModelCombination} + {integrate_way} (IoU={iouThreshold}): F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

            except Exception as e:
                print(f"✗ Error processing {targetModelCombination}: {str(e)}")
                results.append({
                    'map': mapName,
                    'model1': model1,
                    'model2': model2,
                    'model3': model3,
                    'models': targetModelCombination,
                    'integrate': integrate_way,
                    'iou_threshold': iouThreshold,
                    'adaptive': is_adaptive,
                    'f1_score': 'ERROR',
                    'precision': 'ERROR',
                    'recall': 'ERROR',
                    'status': f'ERROR: {str(e)}'
                })

    # CSV ファイルに結果を出力
    if results:
        fieldnames = ['map', 'model1', 'model2', 'model3', 'models', 'integrate', 'iou_threshold', 'adaptive', 
                     'f1_score', 'precision', 'recall', 'status']
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Results saved to: {output_csv_path}")