from dataclasses import dataclass
from typing import Dict

@dataclass
class EvaluationResult:
    """評価結果を保持するデータクラス"""
    mAP: float
    class_ap_dict: Dict[int, float]
    f1_score: float
    precision: float
    recall: float