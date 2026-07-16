# Evaluation/metrics/confidenceScoreAnalysis.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory


@dataclass
class ConfidenceScoreStats:
    """confidenceScoreの統計情報"""
    category: str  # "TP" or "FP"
    class_id: int
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float

    def __str__(self) -> str:
        return (
            f"{self.category:3s} Class {self.class_id}: "
            f"count={self.count:3d}, mean={self.mean:.3f}, median={self.median:.3f}, "
            f"std={self.std:.3f}, min={self.min:.3f}, max={self.max:.3f}"
        )


class ConfidenceScoreAnalyzer:
    """TP/FPのconfidenceScore分析クラス"""

    def analyze_by_class(
        self,
        classified_boxes: List[ClassifiedBoundingBox]
    ) -> Dict[int, Dict[str, ConfidenceScoreStats]]:
        """
        クラスごとにTP/FPのconfidenceScoreを分析

        Args:
            classified_boxes: 分類済みバウンディングボックスのリスト

        Returns:
            {class_id: {"TP": ConfidenceScoreStats, "FP": ConfidenceScoreStats}}
        """
        # クラスごと、カテゴリごとにスコアを収集
        scores_by_class_category: Dict[int, Dict[str, List[float]]] = {}

        for classified_box in classified_boxes:
            class_id = classified_box.classId
            category = classified_box.classifyCategory

            # TPとFPのみ対象
            if category == ClassifyCategory.TP:
                category_str = "TP"
            elif category == ClassifyCategory.FP:
                category_str = "FP"
            else:
                continue  # FNはスキップ

            if class_id not in scores_by_class_category:
                scores_by_class_category[class_id] = {"TP": [], "FP": []}

            score = classified_box.confidenceScore
            if score is not None:
                scores_by_class_category[class_id][category_str].append(score)

        # 統計情報を計算
        results: Dict[int, Dict[str, ConfidenceScoreStats]] = {}

        for class_id, categories in scores_by_class_category.items():
            results[class_id] = {}

            for category_str, scores in categories.items():
                if len(scores) == 0:
                    # スコアがない場合はスキップ
                    continue

                scores_array = np.array(scores, dtype=np.float32)

                stats = ConfidenceScoreStats(
                    category=category_str,
                    class_id=class_id,
                    count=len(scores),
                    mean=float(np.mean(scores_array)),
                    median=float(np.median(scores_array)),
                    std=float(np.std(scores_array)),
                    min=float(np.min(scores_array)),
                    max=float(np.max(scores_array))
                )

                results[class_id][category_str] = stats

        return results

    def analyze_overall(
        self,
        classified_boxes: List[ClassifiedBoundingBox]
    ) -> Dict[str, ConfidenceScoreStats]:
        """
        全体のTP/FPのconfidenceScoreを分析（クラス横断）

        Args:
            classified_boxes: 分類済みバウンディングボックスのリスト

        Returns:
            {"TP": ConfidenceScoreStats, "FP": ConfidenceScoreStats}
        """
        tp_scores: List[float] = []
        fp_scores: List[float] = []

        for classified_box in classified_boxes:
            score = classified_box.confidenceScore
            if score is None:
                continue

            if classified_box.classifyCategory == ClassifyCategory.TP:
                tp_scores.append(score)
            elif classified_box.classifyCategory == ClassifyCategory.FP:
                fp_scores.append(score)

        results = {}

        for category_str, scores in [("TP", tp_scores), ("FP", fp_scores)]:
            if len(scores) == 0:
                continue

            scores_array = np.array(scores, dtype=np.float32)

            stats = ConfidenceScoreStats(
                category=category_str,
                class_id=-1,  # 全体を表す
                count=len(scores),
                mean=float(np.mean(scores_array)),
                median=float(np.median(scores_array)),
                std=float(np.std(scores_array)),
                min=float(np.min(scores_array)),
                max=float(np.max(scores_array))
            )

            results[category_str] = stats

        return results

    def print_analysis(
        self,
        classified_boxes: List[ClassifiedBoundingBox],
        by_class: bool = True,
        overall: bool = True
    ) -> None:
        """
        分析結果をコンソールに出力

        Args:
            classified_boxes: 分類済みバウンディングボックスのリスト
            by_class: クラスごとの統計を出力するか
            overall: 全体の統計を出力するか
        """
        print("\n" + "="*100)
        print("Confidence Score Analysis")
        print("="*100)

        if overall:
            print("\n【Overall Statistics】")
            overall_stats = self.analyze_overall(classified_boxes)
            for category_str, stats in overall_stats.items():
                print(f"  {stats}")

        if by_class:
            print("\n【Statistics by Class】")
            by_class_stats = self.analyze_by_class(classified_boxes)

            for class_id in sorted(by_class_stats.keys()):
                print(f"\n  Class {class_id}:")
                for category_str in ["TP", "FP"]:
                    if category_str in by_class_stats[class_id]:
                        stats = by_class_stats[class_id][category_str]
                        print(f"    {stats}")

        print("\n" + "="*100 + "\n")
