from enum import Enum
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel

from src.boundingBox.boundingBox import DetectionBoundingBox


class AdrodState(Enum):
    """モデル起動レベル"""
    SOLO = 1           # Level 1: ベースモデルのみ
    PAIR = 2           # Level 2: 2つのモデルで統合
    ENSEMBLE = 3       # Level 3: 3つ以上のモデル全部


@dataclass
class ThresholdConfig(BaseModel):
    """段階的起動の設定"""
    tau_p: float
    theta_u: float
    theta_high: float
    theta_low: float

    def showConfig(self):
        print(vars(self))

class StateDecisionStrategy(ABC):
    """
    各steteからstateへの遷移を決定する
    """
    
    @abstractmethod
    def decide(self, *args, **kwargs) -> AdrodState:
        """stateを決定"""
        pass
    
class SingleVersionSteteDecisionStrategy(StateDecisionStrategy):
    """Single version での遷移を担当"""
    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds: ThresholdConfig = thresholds
        
    def _calc_uncertain_detection_ratio(self, detection_list: list[DetectionBoundingBox]) -> float:
        # 不確実な検出の割合を計算
        num_uncertain_detection_instance: int = sum(
            1 for bbox in detection_list 
            if bbox.confidenceScore < self.thresholds.tau_p
        )
        num_detection_instance: int = len(detection_list)
        uncertain_ratio: float = num_uncertain_detection_instance / num_detection_instance
        return uncertain_ratio
    
    def decide(self, detection_list: list[DetectionBoundingBox]):
        """single version から two-versionに遷移するかを判定"""
        # 検出がなければsingle-versionを継続
        if not detection_list:
            return AdrodState.SOLO
        
        # 不確実な検出の割合を計算
        uncertain_ratio: float = self._calc_uncertain_detection_ratio(detection_list)
        
        # 不確実な検出が多ければtwo-versionに遷移
        return (
            AdrodState.PAIR 
            if uncertain_ratio >= self.thresholds.theta_u
            else AdrodState.SOLO
        )
    
class TwoVersionStateDecisionStrategy(StateDecisionStrategy):
    """two-version状態での判定"""
    
    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds: ThresholdConfig = thresholds
        
    def _calc_jaccar_coefficient(self, bbox_groups: list[list[DetectionBoundingBox]]) -> float:
        # 2つの検出で一致した検出の割合
        num_agreed_detection: int = sum(1 for group in bbox_groups if len(group) == 2)
        jaccard_value: float = num_agreed_detection / len(bbox_groups)
        return jaccard_value
    
    def decide(self, bbox_groups: list[list[DetectionBoundingBox]]) -> AdrodState:
        """two-version状態から遷移するかを判定"""
        if not bbox_groups:
            return AdrodState.SOLO
        
        # 2つの検出で一致した検出の割合
        jaccard_value: float = self._calc_jaccar_coefficient(bbox_groups)
        
        if jaccard_value >= self.thresholds.theta_high:
            return AdrodState.SOLO
        elif jaccard_value <= self.thresholds.theta_low:
            return AdrodState.ENSEMBLE
        else:
            return AdrodState.PAIR

class EnsembleStateDecisionStrategy(StateDecisionStrategy):
    """ENSEMBLE状態での遷移を担当"""
    
    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds: ThresholdConfig = thresholds
    
    def decide(self, bbox_groups: list[list[DetectionBoundingBox]]) -> AdrodState:
        return (
            AdrodState.PAIR
        )
        
class AdrodStateController:
    """
    各state からstate の遷移を制御する
    """
    
    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds = thresholds
        self._state = AdrodState.SOLO
        
        # 各状態の判定ストラテジーを準備
        self._strategies: list[StateDecisionStrategy] = {
            AdrodState.SOLO: SingleVersionSteteDecisionStrategy(self.thresholds),
            AdrodState.PAIR: TwoVersionStateDecisionStrategy(self.thresholds),
            AdrodState.ENSEMBLE: EnsembleStateDecisionStrategy(self.thresholds),
        }
        
        # stateの変化を記録
        self.exe_state_recorder: list[AdrodState] = []
        
        # record each state freaquency
        self.adrod_state_counter = {
            AdrodState.SOLO: 0,
            AdrodState.PAIR: 0,
            AdrodState.ENSEMBLE: 0,
        }
    
    @property
    def state(self) -> AdrodState:
        """現在の状態を取得"""
        return self._state
    
    @state.setter
    def state(self, new_state: AdrodState):
        """状態を設定"""
        if new_state in self._strategies:
            self._state = new_state
        else:
            raise ValueError(f"Invalid state: {new_state}")
    
    def decide_state(
        self,
        detections: Optional[list[DetectionBoundingBox]] = None,
        bbox_groups: Optional[list[list[DetectionBoundingBox]]] = None,
    ) -> None:
        """
        現在の状態と入力に基づいて次の状態を決定
        
        Args:
            detections: single-versionでの使用（ベースモデルの検出結果）
            bbox_groups: two-version状態での使用（グループ化された検出）
        
        Returns:
            次のAdrodState
        """
        strategy:StateDecisionStrategy = self._strategies[self._state]
        
        if self._state == AdrodState.SOLO:
            self._state = strategy.decide(detections or [])
        else:
            self._state = strategy.decide(bbox_groups or [])
        
        
    
    def record_execution(self, state: AdrodState):
        """実行されたstateを記録"""
        self.exe_state_recorder.append(state)
        self.adrod_state_counter[state] += 1
    
    def get_stats(self) -> dict:
        """起動統計情報を取得"""
        total = sum(self.adrod_state_counter.values())
        return {
            state.name: {
                'count': count,
                'percentage': f"{100 * count / total:.1f}%" if total > 0 else "0%"
            }
            for state, count in self.adrod_state_counter.items()
        }
    
    def reset_stats(self):
        """統計情報をリセット"""
        for state in self.adrod_state_counter:
            self.adrod_state_counter[state] = 0
    
    def reset_state(self):
        """状態を初期化"""
        self._state = AdrodState.SOLO