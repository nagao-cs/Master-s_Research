import argparse
from pathlib import Path
import os
import csv
from datetime import datetime
from collections import Counter
from logging import getLogger
from tqdm import tqdm
import time

from src.adrod.stateController import EscalationController, EscalationLevel, EscalationConfig
from src.boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from src.boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from src.boundingBox.boundingBox import DetectionBoundingBox, GroundTruthBoundingBox, ClassifiedBoundingBox
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.dataset import fileReader
from src.Evaluation.metrics import f1Score, mAP
from src.ObjectDetection.models.FLOPsDict import FLOPs_Dict


def load_detection_cache(baseDir: Path, mapName: str, modelNameList: list[str]) -> dict:
    """キャッシュから検出結果を読み込み"""
    detectionBaseDir = baseDir / "oneVersionDetectionResult" / "labels" / mapName
    detection_cache = {}

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

        modelDetections = []
        for detectionFile in detectionFiles:
            detectionFilePath = modelDetectionDir / detectionFile
            detectionBBoxList = fileReader.convertDetectionFileToBoundingBoxList(
                str(detectionFilePath)
            )
            modelDetections.append(detectionBBoxList)

        detection_cache[modelName] = modelDetections

    return detection_cache


def execute_detection(baseDir: Path, mapName: str, modelNameList: list[str], 
                      integrate_way: str, is_adaptive: bool, 
                      escalation_config: EscalationConfig = None, 
                      iou_threshold: float = 0.5) -> Path:
    """検出実行（実際には キャッシュから読み込んで統合）"""
    
    detection_cache = load_detection_cache(baseDir, mapName, modelNameList)
    numFrames = len(detection_cache[modelNameList[0]])
    
    # Integrator初期化
    if integrate_way == "affirmative":
        integrator = ConfidenceBaseIntegrator(
            iouThreshold=iou_threshold,
            confidenceThreshold=0.0
        )
    elif integrate_way == "conf_base":
        integrator = ConfidenceBaseIntegrator(
            iouThreshold=iou_threshold,
            confidenceThreshold=escalation_config.uncertainty_conf_threshold if escalation_config else 0.5
        )
    elif integrate_way == "consensus":
        integrator = MajorityIntegrator(
            iouThreshold=iou_threshold,
            maxVersion=len(modelNameList)
        )
    
    outputDetectionList = []
    escalation_levels_record = []
    totalFLOPs = 0.0
    
    if is_adaptive and escalation_config:
        # 適応的処理（escalation）
        escalation_controller = EscalationController(escalation_config)
        baseModel = modelNameList[0]
        
        for frameIdx in tqdm(range(numFrames), desc="Processing frames (Adaptive)"):
            base_detections = detection_cache[baseModel][frameIdx]
            second_detections = detection_cache[modelNameList[1]][frameIdx]
            third_detections = detection_cache[modelNameList[2]][frameIdx]
            
            if escalation_controller.state == EscalationLevel.SOLO:
                escalation_controller.decide_escalation_level(
                    soloBBoxList=base_detections
                )
                if escalation_controller.state == EscalationLevel.SOLO:
                    exeState = EscalationLevel.SOLO
                    final_detections = base_detections
                elif escalation_controller.state == EscalationLevel.PAIR:
                    pair_dict = {
                        baseModel: base_detections,
                        modelNameList[1]: second_detections
                    }
                    pairDetList, BBoxGroupList = integrator(pair_dict)
                    escalation_controller.decide_escalation_level(
                        BBoxGroupList=BBoxGroupList)
                    if escalation_controller.state == EscalationLevel.ENSEMBLE:
                        ensemble_dict = {
                            baseModel: base_detections,
                            modelNameList[1]: second_detections,
                            modelNameList[2]: third_detections
                        }
                        exeState = EscalationLevel.ENSEMBLE
                        final_detections, _ = integrator(ensemble_dict)
                    else:
                        exeState = EscalationLevel.PAIR
                        final_detections = pairDetList
            elif escalation_controller.state == EscalationLevel.PAIR:
                pair_dict = {
                    baseModel: base_detections,
                    modelNameList[1]: second_detections
                }
                pairDetList, BBoxGroupList = integrator(pair_dict)
                escalation_controller.decide_escalation_level(
                    BBoxGroupList=BBoxGroupList)
                if escalation_controller.state == EscalationLevel.ENSEMBLE:
                    ensemble_dict = {
                        baseModel: base_detections,
                        modelNameList[1]: second_detections,
                        modelNameList[2]: third_detections
                    }
                    exeState = EscalationLevel.ENSEMBLE
                    final_detections, _ = integrator(ensemble_dict)
                else:
                    exeState = EscalationLevel.PAIR
                    final_detections = pairDetList
            elif escalation_controller.state == EscalationLevel.ENSEMBLE:
                ensemble_dict = {
                    baseModel: base_detections,
                    modelNameList[1]: second_detections,
                    modelNameList[2]: third_detections
                }
                exeState = EscalationLevel.ENSEMBLE
                final_detections, BBoxGroupList = integrator(ensemble_dict)
                escalation_controller.decide_escalation_level(
                    BBoxGroupList=BBoxGroupList)
            
            escalation_levels_record.append(exeState)
            outputDetectionList.append(final_detections)
            
            if exeState == EscalationLevel.SOLO:
                totalFLOPs += FLOPs_Dict[baseModel]
            elif exeState == EscalationLevel.PAIR:
                totalFLOPs += FLOPs_Dict[baseModel] + FLOPs_Dict[modelNameList[1]]
            else:
                totalFLOPs += FLOPs_Dict[baseModel] + FLOPs_Dict[modelNameList[1]] + FLOPs_Dict[modelNameList[2]]
            
            if escalation_controller.state == EscalationLevel.ENSEMBLE:
                escalation_controller.state = EscalationLevel.PAIR
    else:
        # 常にN-version処理
        for frameIdx in tqdm(range(numFrames), desc="Processing frames (N-version)"):
            modelDict = {
                modelName: detection_cache[modelName][frameIdx]
                for modelName in modelNameList
            }
            integratedBoundingBoxList, _ = integrator(modelDict)
            outputDetectionList.append(integratedBoundingBoxList)
            totalFLOPs += sum(FLOPs_Dict.get(m, 0) for m in modelNameList)
    
    # 出力ディレクトリ
    if is_adaptive:
        outputLabelDir = baseDir / "adaptiveDetectionResult" / "escalation" / "labels" / \
            f"{mapName}" / f"{'_'.join(modelNameList)}" / f"{integrate_way}"
    else:
        outputLabelDir = baseDir / "NversionDetectionResult" / "labels" / \
            f"{mapName}" / f"{'_'.join(modelNameList)}" / f"{integrate_way}"
    
    os.makedirs(outputLabelDir, exist_ok=True)
    
    # 結果を保存
    for idx, detections in enumerate(outputDetectionList):
        output_path = outputLabelDir / f"{idx:06d}.txt"
        with open(output_path, 'w') as f:
            for bbox in detections:
                f.write(
                    f"{bbox.classId} {bbox.xCenter} {bbox.yCenter} "
                    f"{bbox.width} {bbox.height} {bbox.confidenceScore}\n"
                )
    
    print(f"✓ Detection saved to: {outputLabelDir}")
    print(f"cost: {totalFLOPs / 17400}")
    level_counter = Counter(escalation_levels_record)
    print(f"Single Frames: {level_counter.get(EscalationLevel.SOLO)}, Two : {level_counter.get(EscalationLevel.PAIR)}, Three : {level_counter.get(EscalationLevel.ENSEMBLE)}")
    return outputLabelDir


def evaluate_detection(baseDir: Path, mapName: str, detectionLabelDir: Path, 
                       iou_threshold: float = 0.5) -> dict:
    """検出結果を評価"""
    
    groundTruthDatasetDir = baseDir / "output" / "label" / f"{mapName}" / "front"
    
    if not os.path.exists(groundTruthDatasetDir):
        raise FileNotFoundError(f"{groundTruthDatasetDir} does not exist")
    
    classifiedBoundingBoxList = []
    detectionClassifier = DetectionClassifier(iouThreshold=iou_threshold)
    
    groundTruthFileList = sorted(os.listdir(groundTruthDatasetDir))
    detectionFileList = sorted(os.listdir(detectionLabelDir))
    
    for groudTruthFile, detectionFile in zip(groundTruthFileList, detectionFileList):
        groudTruthFilePath = os.path.join(groundTruthDatasetDir, groudTruthFile)
        detectionFilePath = os.path.join(detectionLabelDir, detectionFile)
        
        if not os.path.exists(groudTruthFilePath) or not os.path.exists(detectionFilePath):
            continue
        
        groundTruthBoundingBoxList = fileReader.convertGroundTruthFileToBoundingBoxList(
            groudTruthFilePath)
        detectionBoundingBoxList = fileReader.convertDetectionFileToBoundingBoxList(
            detectionFilePath)
        
        classifiedBoundingBoxListPerFrame = detectionClassifier.classify(
            groundTruthBoundingBoxList, detectionBoundingBoxList)
        
        classifiedBoundingBoxList.extend(classifiedBoundingBoxListPerFrame)
    
    # F1スコア計算
    f1, precision, recall = f1Score.computeF1Score(classifiedBoundingBoxList)
    
    # mAP計算
    sortedClassifiedBoundingBoxList = sorted(
        classifiedBoundingBoxList, key=lambda bbox: bbox.confidenceScore, reverse=True)
    targetClassIdList = [0, 2, 9, 11]
    mAPValue, classIdApDict = mAP.computeMeanAP(
        sortedClassifiedBoundingBoxList, targetClassIdList)
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'mAP': mAPValue,
        'AP_pedestrian': classIdApDict.get(0, 0),
        'AP_vehicle': classIdApDict.get(2, 0),
        'AP_traffic_light': classIdApDict.get(9, 0),
        'AP_traffic_sign': classIdApDict.get(11, 0)
    }


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="End-to-end detection and evaluation pipeline"
    )
    argparser.add_argument(
        "--execution_type",
        type=str,
        choices=['adaptive', 'nversion'],
        default='nversion',
        help="Execution type: adaptive or nversion"
    )
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
        help="List of models to vary as model 2"
    )
    argparser.add_argument(
        "--model3",
        type=str,
        required=False,
        help="Fixed model 3 (e.g., rtdetr)"
    )
    argparser.add_argument(
        "--integrate",
        type=str,
        nargs="+",
        default=["affirmative", "conf_base"],
        help="Integration methods"
    )
    argparser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    
    # Escalation専用パラメータ
    argparser.add_argument(
        "--pair_to_solo",
        type=float,
        default=0.8
    )
    argparser.add_argument(
        "--uncertainty_conf",
        type=float,
        default=0.5
    )
    argparser.add_argument(
        "--uncertainty_ratio",
        type=float,
        default=0.25
    )
    
    args = argparser.parse_args()
    print(args)
    
    mapName = args.map
    iouThreshold = args.iou_th
    integrate_ways = args.integrate
    is_adaptive = args.execution_type == 'adaptive'
    
    model1 = args.model1
    model2_list = args.model2_list
    model3 = args.model3 if args.model3 else ""
    
    cwd = Path(__file__).parent
    baseDir = cwd.parent.parent
    
    if args.output_csv is None:
        output_csv_path = baseDir / "result" / f"pipeline_results_{args.execution_type}.csv"
    else:
        output_csv_path = Path(args.output_csv)
    
    os.makedirs(output_csv_path.parent, exist_ok=True)
    
    logger = getLogger('ultralytics')
    logger.disabled = True
    
    results = []
    
    for integrate_way in integrate_ways:
        for model2 in model2_list:
            modelNameList = [model1, model2, model3] if model3 else [model1, model2]
            targetModelCombination = f"{'_'.join(modelNameList)}"
            
            print(f"\n{'='*60}")
            print(f"Processing: {targetModelCombination} + {integrate_way}")
            print(f"{'='*60}")
            
            try:
                # Escalation設定（必要な場合）
                escalation_config = None
                if is_adaptive:
                    escalation_config = EscalationConfig(
                        uncertainty_conf_threshold=args.uncertainty_conf,
                        uncertainty_ratio_threshold=args.uncertainty_ratio,
                        pair_to_solo_threshold=args.pair_to_solo,
                        pair_to_ensemble_threshold=0.5,
                        ensemble_to_pair_threshold=0.5
                    )
                
                # 実行
                detectionLabelDir = execute_detection(
                    baseDir, mapName, modelNameList, integrate_way, is_adaptive,
                    escalation_config, iouThreshold
                )
                
                # 評価
                metrics = evaluate_detection(baseDir, mapName, detectionLabelDir, iouThreshold)
                
                result = {
                    'map': mapName,
                    'execution_type': args.execution_type,
                    'model1': model1,
                    'model2': model2,
                    'model3': model3 if model3 else 'N/A',
                    'models': targetModelCombination,
                    'integrate': integrate_way,
                    'iou_threshold': iouThreshold,
                    'f1_score': f"{metrics['f1_score']:.4f}",
                    'precision': f"{metrics['precision']:.4f}",
                    'recall': f"{metrics['recall']:.4f}",
                    'mAP': f"{metrics['mAP']:.4f}",
                    'AP_pedestrian': f"{metrics['AP_pedestrian']:.4f}",
                    'AP_vehicle': f"{metrics['AP_vehicle']:.4f}",
                    'AP_traffic_light': f"{metrics['AP_traffic_light']:.4f}",
                    'AP_traffic_sign': f"{metrics['AP_traffic_sign']:.4f}",
                    'status': 'SUCCESS'
                }
                
                results.append(result)
                print(f"✓ F1={metrics['f1_score']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, mAP={metrics['mAP']:.4f}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                results.append({
                    'map': mapName,
                    'execution_type': args.execution_type,
                    'model1': model1,
                    'model2': model2,
                    'model3': model3 if model3 else 'N/A',
                    'models': targetModelCombination,
                    'integrate': integrate_way,
                    'iou_threshold': iouThreshold,
                    'f1_score': 'ERROR',
                    'precision': 'ERROR',
                    'recall': 'ERROR',
                    'mAP': 'ERROR',
                    'AP_pedestrian': 'ERROR',
                    'AP_vehicle': 'ERROR',
                    'AP_traffic_light': 'ERROR',
                    'AP_traffic_sign': 'ERROR',
                    'status': f'ERROR: {str(e)}'
                })
    
    # CSV出力
    if results:
        fieldnames = ['map', 'execution_type', 'model1', 'model2', 'model3', 'models', 'integrate', 
                     'iou_threshold', 'f1_score', 'precision', 'recall', 'mAP', 
                     'AP_pedestrian', 'AP_vehicle', 'AP_traffic_light', 'AP_traffic_sign', 'status']
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ All results saved to: {output_csv_path}")