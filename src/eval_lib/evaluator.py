import os
from pathlib import Path
from src.Evaluation.dataset import fileReader
from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.metrics import mAP, f1Score
from src.boundingBox.boundingBox import ClassifiedBoundingBox, DetectionBoundingBox
from .evaluation_result import EvaluationResult
from .reliability import CerOD, CovOD, analyze_frame

class Evaluator:
    """検出結果の評価を実行する"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)
    
    def evaluate(
        self,
        gt_dataset_dir: Path,
        detection_dataset_dir: Path,
        target_class_ids: list[int] | None = None,
        version_detections: list[list[list[DetectionBoundingBox]]] | None = None,
    ) -> EvaluationResult:
        """
        Args:
            gt_dataset_dir
            detection_dataset_dir
            target_class_ids: 評価対象のクラスID（デフォルト: 0〜12）
            version_detections: フレームごとの、実行した各バージョンの検出結果
                （frame → version → boxes）。渡すと Cov_OD / Cer_OD も計算する。
                フレーム順は GT ファイルを名前順に並べた順と同じであること。

        Returns:
            EvaluationResult: 評価結果
        """
        if target_class_ids is None:
            target_class_ids = [i for i in range(13)]
        
        print(f"gt_dir: {gt_dataset_dir}")
        print(f"det_dir: {detection_dataset_dir}")
        print(f"target_class: {target_class_ids}")
        
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

        cov_od = cer_od = None
        if version_detections is not None:
            cov_od, cer_od = self.evaluate_reliability(
                gt_dataset_dir, version_detections, target_class_ids
            )

        return EvaluationResult(
            mAP=map_value,
            class_ap_dict=class_ap_dict,
            f1_score=f1,
            precision=precision,
            recall=recall,
            cov_od=cov_od,
            cer_od=cer_od,
        )

    def evaluate_reliability(
        self,
        gt_dataset_dir: Path,
        version_detections: list[list[list[DetectionBoundingBox]]],
        target_class_ids: list[int],
    ) -> tuple[float, float]:
        """
        Cov_OD / Cer_OD を計算する（src/eval_lib/reliability.py）。
        評価対象クラス以外の GT・検出は除いてから集合を作る。
        """
        gt_files = sorted(os.listdir(gt_dataset_dir))
        if len(gt_files) != len(version_detections):
            raise ValueError(
                f"GT のフレーム数 {len(gt_files)} と検出のフレーム数 {len(version_detections)} が一致しません"
            )

        targets = set(target_class_ids)
        cov, cer = CovOD(), CerOD()
        for gt_file, versions in zip(gt_files, version_detections):
            gt = [
                b for b in fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_dataset_dir / gt_file))
                if b.classId in targets
            ]
            dets = [[b for b in boxes if b.classId in targets] for boxes in versions]
            frame = analyze_frame(gt, dets, self.iou_threshold)
            cov.update(frame)
            cer.update(frame)
        return cov.compute(), cer.compute()
    
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