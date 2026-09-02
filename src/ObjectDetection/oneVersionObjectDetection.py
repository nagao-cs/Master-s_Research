from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse
import cv2
import csv
from datetime import datetime

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.eval_lib.evaluator import Evaluator
from .models.factory import build_model
from ..config import DATASET_DIR

def get_ground_truth_dir(dataset: str, mapName: str) -> Path:
    """データセットに応じてGround Truthディレクトリを取得"""
    if dataset == "KITTI":
        return Path(f"/mnt/d/kitti/tracking/labels/{mapName}")
    elif dataset == "CARLA":
        return Path(f"mnt/c/output/label/{mapName}/front")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="One version Object Detection"
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
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

    args = argparser.parse_args()
    print(args)

    modelName: str = args.model
    mapName: str = args.map
    dataset: str = args.dataset

    model = build_model(model_name=modelName, dataset=dataset, device='cuda')

    if dataset == "KITTI":
        input_image_dir = Path(f"/mnt/d/kitti/tracking/{args.map}/images/")
    elif dataset == "CARLA":
        input_image_dir =  Path(f"mnt/c/output/image/{mapName}/original/front")

    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    print(f"Dataset: {dataset}")
    print(f"Map: {mapName}")
    print(f"Model: {modelName}")

    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(input_image_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {input_image_dir},\n execution file is {Path(__file__)}")

    input_image_path_list: list[Path] = sorted(
        [input_image_path for input_image_path in input_image_dir.iterdir() if input_image_path.is_file()]
    )
    print(input_image_path_list)

    # 計測開始
    start: float = time.time()
    for input_image_path in tqdm(input_image_path_list, desc="[detection]"):
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"{input_image_path} does not exist")

        finalDetections = model.predict(image_path=input_image_path)
        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"\nTotal object detection time: {executionTime:.2f} seconds")

    outputLabelDir = DATASET_DIR / f"single_model_detection/{dataset}/{mapName}/{modelName}/labels"
    outputImageDir = DATASET_DIR / f"single_model_detection/{dataset}/{mapName}/{modelName}/images"
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
            evaluator = Evaluator(iou_threshold=0.5)
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
    csv_output_dir = DATASET_DIR / f"single_model_detection/{dataset}/{mapName}/eval_result"
    os.makedirs(csv_output_dir, exist_ok=True)
    
    csv_path = csv_output_dir / "results.csv"
    
    # CSVファイルが存在しない場合はヘッダーを書き込む
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'dataset', 'model', 'map', 
                      'mAP', 'F1_Score', 'Precision', 'Recall', 'Execution_Time_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        if evaluation_result:
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset': dataset,
                'model': modelName,
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
                'model': modelName,
                'map': mapName,
                'mAP': 'N/A',
                'F1_Score': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A',
                'Execution_Time_s': f"{executionTime:.2f}"
            })
    
    print(f"✓ Evaluation results saved to: {csv_path}")