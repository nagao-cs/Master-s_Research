from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EvaluationResult:
    """評価結果を保持するデータクラス"""
    mAP: float
    class_ap_dict: Dict[int, float]
    f1_score: float
    precision: float
    recall: float
    cov_od: Optional[float] = None   # version_detections を渡したときだけ計算
    cer_od: Optional[float] = None
