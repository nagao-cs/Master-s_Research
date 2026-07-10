from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse
import cv2
import csv
from datetime import datetime

from src.boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from src.boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from src.boundingBox.boundingBox import DetectionBoundingBox
from src.eval_lib.evaluator import Evaluator
from .models.ObjectDetector import Detector
from ..config import DATASET_DIR


def get_ground_truth_dir(dataset: str, mapName: str) -> Path:
    """データセットに応じてGround Truthディレクトリを取得"""
    if dataset == "KITTI":
        return Path(f"/mnt/d/kitti/tracking/labels/{mapName}")
    elif dataset == "CARLA":
        return Path(f"mnt/c/output/label/{mapName}/front")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_models(model_names: list[str], dataset):
    """複数のモデルを読み込む"""
    from .models.factory import build_model

    models = {}
    for model_name in model_names:
        if model_name == "none":
            continue
        models[model_name] = build_model(model_name, dataset, device='cuda')
        print(f"✓ Loaded model: {model_name}")

    return models


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="N-version Object Detection"
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
        help="Map name: Town01, Town02, etc.",
        default="Town02"
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=["CARLA", "KITTI"],
        required=True
    )
    argparser.add_argument(
        "--integrate",
        type=str,
        default="affirmative"
    )

    args = argparser.parse_args()
    print(args)

    modelNameList: list[str] = [m for m in args.models if m != "none"]
    mapName: str = args.map
    dataset: str = args.dataset
    integrate_way: str = args.integrate

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    if integrate_way == "affirmative":
        integrator = ConfidenceBaseIntegrator(
            iouThreshold=IOU_THRESHOLD,
            confidenceThreshold=0.0
        )
    elif integrate_way == "conf_base":
        integrator = ConfidenceBaseIntegrator(
            iouThreshold=IOU_THRESHOLD,
            confidenceThreshold=CONF_THRESHOLD
        )
    elif integrate_way == "consensus":
        integrator = MajorityIntegrator(
            iouThreshold=IOU_THRESHOLD,
            maxVersion=len(modelNameList)
        )
    else:
        raise ValueError(f"Unknown integrate_way: {integrate_way}")

    print("\nLoading models...")
    models: dict[str, Detector] = load_models(modelNameList, dataset=args.dataset)

    if dataset == "KITTI":
        input_image_dir = Path(f"/mnt/d/kitti/tracking/images/{args.map}")
    elif dataset == "CARLA":
        input_image_dir = Path(f"mnt/c/output/image/{mapName}/original/front")

    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    print(f"Dataset: {dataset}")
    print(f"Map: {mapName}")
    print(f"Models: {modelNameList}")
    print(f"Integration method: {integrate_way}")

    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(input_image_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {input_image_dir},\n execution file is {Path(__file__)}")

    input_image_path_list: list[Path] = sorted(
        [input_image_path for input_image_path in input_image_dir.iterdir() if input_image_path.is_file()]
    )

    # 計測開始
    start: float = time.time()
    for input_image_path in tqdm(input_image_path_list, desc="[detection]"):
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"{input_image_path} does not exist")

        modelDict = {}
        for model_name in modelNameList:
            detection_result = models[model_name].predict(image_path=input_image_path)
            modelDict[model_name] = detection_result

        integratedBoundingBoxList, _ = integrator(modelDict)
        outputDetectionList.append(integratedBoundingBoxList)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"\nTotal object detection time: {executionTime:.2f} seconds")

    modelsTag = '_'.join(modelNameList)
    outputLabelDir = DATASET_DIR / f"n_version_detection/{dataset}/labels/{mapName}/{modelsTag}/{integrate_way}"
    outputImageDir = DATASET_DIR / f"n_version_detection/{dataset}/images/{mapName}/{modelsTag}/{integrate_way}"
    os.makedirs(outputLabelDir, exist_ok=True)
    os.makedirs(outputImageDir, exist_ok=True)

    index: int = 0
    for inputImagePath, outputLabelList in tqdm(zip(input_image_path_list, outputDetectionList), desc="[result save]", total=len(outputDetectionList)):
        outputImagePath: Path = outputImageDir / f"{index:06}.png"
        outputLabelPath: Path = outputLabelDir / f"{index:06}.txt"

        outputImage = cv2.imread(str(inputImagePath))
        if outputImage is None:
            raise FileNotFoundError(f"Could not read image: {inputImagePath}")

        with open(outputLabelPath, 'w') as outputFile:
            for boundingBox in outputLabelList:
                outputImage = boundingBox.drawBoundingBoxOnImage(outputImage)

                xCenter = boundingBox.xCenter
                yCenter = boundingBox.yCenter
                width = boundingBox.width
                height = boundingBox.height
                classId = boundingBox.classId
                confidenceScore = boundingBox.confidenceScore
                outputFile.write(
                    f"{classId} {xCenter} {yCenter} {width} {height} {confidenceScore}\n")
        cv2.imwrite(filename=str(outputImagePath), img=outputImage)
        index += 1

    print(f"✓ Results saved to: {outputLabelDir}")

    # -----------
    # 評価を実行
    # -----------
    print("\nEvaluating detections...")
    try:
        ground_truth_dir = get_ground_truth_dir(dataset, mapName)
        print(ground_truth_dir)

        if not ground_truth_dir.exists():
            print(f"⚠ Ground truth directory not found: {ground_truth_dir}")
            evaluation_result = None
        else:
            evaluator = Evaluator(iou_threshold=IOU_THRESHOLD)
            evaluation_result = evaluator.evaluate(
                gt_dataset_dir=ground_truth_dir,
                detection_dataset_dir=outputLabelDir,
            )

            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"mAP:       {evaluation_result.mAP:.4f}")
            print(f"F1 Score:  {evaluation_result.f1_score:.4f}")
            print(f"Precision: {evaluation_result.precision:.4f}")
            print(f"Recall:    {evaluation_result.recall:.4f}")
            print(f"Time:      {executionTime:.2f}s")
            print("="*60 + "\n")

    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
        evaluation_result = None

    # -----------
    # 結果をCSVで保存
    # -----------
    csv_output_dir = DATASET_DIR / f"n_version_detection/{dataset}/eval_result"
    os.makedirs(csv_output_dir, exist_ok=True)

    csv_path = csv_output_dir / "results.csv"

    # CSVファイルが存在しない場合はヘッダーを書き込む
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'dataset', 'models', 'integrate_way', 'map',
                      'mAP', 'F1_Score', 'Precision', 'Recall', 'Execution_Time_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        if evaluation_result:
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset': dataset,
                'models': modelsTag,
                'integrate_way': integrate_way,
                'map': mapName,
                'mAP': f"{evaluation_result.mAP:.4f}",
                'F1_Score': f"{evaluation_result.f1_score:.4f}",
                'Precision': f"{evaluation_result.precision:.4f}",
                'Recall': f"{evaluation_result.recall:.4f}",
                'Execution_Time_s': f"{executionTime:.2f}"
            })
        else:
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset': dataset,
                'models': modelsTag,
                'integrate_way': integrate_way,
                'map': mapName,
                'mAP': 'N/A',
                'F1_Score': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A',
                'Execution_Time_s': f"{executionTime:.2f}"
            })

    print(f"✓ Evaluation results saved to: {csv_path}")