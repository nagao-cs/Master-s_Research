import os
from pathlib import Path
from src.Evaluation.dataset import fileReader
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.metrics import mAP, f1Score
from src.boundingBox.boundingBox import ClassifiedBoundingBox
from .evaluation_result import EvaluationResult

class Evaluator:
    """検出結果の評価を実行する"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)
    
    def evaluate(
        self,
        gt_dataset_dir: Path,
        detection_dataset_dir: Path,
        target_class_ids: list[int] = None
    ) -> EvaluationResult:
        """
        Args:
            gt_dataset_dir
            detection_dataset_dir
            target_class_ids: 評価対象のクラスID（デフォルト: [0, 2, 9, 11]）
        
        Returns:
            EvaluationResult: 評価結果
        """
        if target_class_ids is None:
            target_class_ids = [0, 2, 9, 11]
        
        classified_boxes = self._classify_detections(
            gt_dataset_dir, detection_dataset_dir
        )
        
        sorted_boxes = sorted(
            classified_boxes,
            key=lambda box: box.confidenceScore,
            reverse=True
        )
        
        map_value, class_ap_dict = mAP.computeMeanAP(
            sorted_boxes, target_class_ids
        )
        f1, precision, recall = f1Score.computeF1Score(classified_boxes)
        
        return EvaluationResult(
            mAP=map_value,
            class_ap_dict=class_ap_dict,
            f1_score=f1,
            precision=precision,
            recall=recall
        )
    
    def _classify_detections(
        self,
        gt_dataset_dir: Path,
        detection_dataset_dir: Path
    ) -> list[ClassifiedBoundingBox]:
        """バウンディングボックスを分類"""
        classified_list = []
        gt_files = sorted(os.listdir(gt_dataset_dir))
        detection_files = sorted(os.listdir(detection_dataset_dir))
        
        for gt_file, det_file in zip(gt_files, detection_files):
            gt_path = gt_dataset_dir / gt_file
            det_path = detection_dataset_dir / det_file
            
            if not gt_path.exists() or not det_path.exists():
                continue
            
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(
                str(gt_path)
            )
            det_boxes = fileReader.convertDetectionFileToBoundingBoxList(
                str(det_path)
            )
            
            classified = self.classifier.classify(gt_boxes, det_boxes)
            classified_list.extend(classified)
        
        return classified_list