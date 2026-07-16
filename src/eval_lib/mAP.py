import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict

from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory


class mAP:
    """物体検出のmAPを計算するクラス"""
    
    NUM_RECALL_POINTS = 101
    
    @staticmethod
    def _filter_by_class(
        class_id: int,
        bbox_list: list[ClassifiedBoundingBox]
    ) -> list[ClassifiedBoundingBox]:
        """指定クラスのバウンディングボックスをフィルタリング"""
        return [bbox for bbox in bbox_list if bbox.classId == class_id]
    
    @staticmethod
    def _compute_tp_fp_arrays(
        bbox_list: list[ClassifiedBoundingBox]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """TP/FPの累積配列を計算"""
        num_tp = 0
        num_fp = 0
        tp_array = np.zeros(len(bbox_list), dtype=np.int32)
        fp_array = np.zeros(len(bbox_list), dtype=np.int32)
        
        # gtの総数を数える
        num_gt = sum(
            1 for bbox in bbox_list 
            if bbox.classifyCategory in (ClassifyCategory.TP, ClassifyCategory.FN)
        )
        
        for i, bbox in enumerate(bbox_list):
            if bbox.classifyCategory == ClassifyCategory.TP:
                num_tp += 1
            elif bbox.classifyCategory == ClassifyCategory.FP:
                num_fp += 1
            
            tp_array[i] = num_tp
            fp_array[i] = num_fp
        
        return tp_array, fp_array, num_gt
    
    @staticmethod
    def _compute_precision_recall(
        tp_array: np.ndarray,
        fp_array: np.ndarray,
        num_gt: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """精度と再現率を計算"""
        if num_gt == 0:
            return np.zeros_like(tp_array, dtype=np.float32), np.zeros_like(tp_array, dtype=np.float32)
        
        # 0除算を避ける
        denominator = tp_array + fp_array
        denominator = np.where(denominator == 0, 1, denominator)
        
        precision = tp_array.astype(np.float32) / denominator
        recall = tp_array.astype(np.float32) / num_gt
        
        return precision, recall
    
    @staticmethod
    def _apply_11point_interpolation(
        precision: np.ndarray,
        recall: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """11点補間を適用"""
        # recall と precision の先頭に0と1を追加
        recall = np.concatenate([[0.0], recall, [1.0]])
        precision = np.concatenate([[1.0], precision, [0.0]])
        
        # 単調性を保証（右から左へ最大値を伝播）
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # 指定されたrecall値での精度を計算
        recall_levels = np.linspace(0, 1, mAP.NUM_RECALL_POINTS)
        interpolated_precision = np.zeros(mAP.NUM_RECALL_POINTS)
        
        for i, recall_level in enumerate(recall_levels):
            # recall_level以上の点から最大精度を取得
            valid_indices = recall >= recall_level
            if np.any(valid_indices):
                interpolated_precision[i] = np.max(precision[valid_indices])
        
        return interpolated_precision, recall_levels
    
    @staticmethod
    def _compute_ap(
        class_id: int,
        bbox_list: list[ClassifiedBoundingBox],
        save_pr_curve: bool = True
    ) -> Tuple[float, Path]:
        """平均精度(AP)を計算"""
        if not bbox_list:
            return 0.0, None
        
        tp_array, fp_array, num_gt = mAP._compute_tp_fp_arrays(bbox_list)
        
        if num_gt == 0:
            return 0.0, None
        
        precision, recall = mAP._compute_precision_recall(tp_array, fp_array, num_gt)
        interpolated_precision, recall_levels = mAP._apply_11point_interpolation(
            precision, recall
        )
        
        # AP = 精度の平均
        ap = float(np.mean(interpolated_precision))
        
        # PR曲線を保存
        curve_path = None
        if save_pr_curve:
            curve_path = mAP._save_pr_curve(
                class_id, interpolated_precision, recall_levels
            )
        
        return ap, curve_path
    
    @staticmethod
    def _save_pr_curve(
        class_id: int,
        precision: np.ndarray,
        recall: np.ndarray
    ) -> Path:
        """Precision-Recall曲線を保存"""
        figure_dir = Path(__file__).parent.parent / "prCurve"
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, marker='o', markersize=6, 
                 label='PR Curve', color='#1f77b4')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Recall', fontsize=13)
        plt.ylabel('Precision', fontsize=13)
        plt.title(f"Class {class_id} PR-Curve")
        plt.legend(fontsize=12)
        
        save_path = figure_dir / f"{class_id}_prCurve.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @staticmethod
    def computeMeanAP(
        bbox_list: list[ClassifiedBoundingBox],
        target_class_ids: list[int],
        save_pr_curves: bool = True
    ) -> Tuple[float, Dict[int, float]]:
        """
        平均精度(mAP)を計算
        
        Args:
            bbox_list: 分類済みバウンディングボックスのリスト
            target_class_ids: 評価対象のクラスID
            save_pr_curves: PR曲線を保存するか
        
        Returns:
            (mAP値, クラスごとのAP辞書)
        """
        class_ap_dict = {}
        
        for class_id in target_class_ids:
            filtered_bboxes = mAP._filter_by_class(class_id, bbox_list)
            ap_value, _ = mAP._compute_ap(
                class_id, filtered_bboxes, save_pr_curves
            )
            class_ap_dict[class_id] = ap_value
        
        # 有効なクラス（1つ以上のGTを持つ）のAPの平均
        valid_aps = [ap for ap in class_ap_dict.values()]
        mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
        
        return mean_ap, class_ap_dict