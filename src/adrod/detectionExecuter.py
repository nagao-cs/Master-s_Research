# src/adrod/frameProcessor.py に追加
from .stateController import AdrodStateController, AdrodState
from src.boundingBox.boundingBox import DetectionBoundingBox
from src.ObjectDetection.models.FLOPsDict import FLOPs_Dict
from dataclasses import dataclass
from typing import Callable

@dataclass
class FrameResult:
    """フレーム処理の結果"""
    detections: list[DetectionBoundingBox]
    state: AdrodState
    flops: float


class CacheFrameProcessor:
    """フレーム単位の処理を担当"""
    
    def __init__(
        self,
        controller: AdrodStateController,
        integrator: Callable,
        detection_cache: dict[str, list[list[DetectionBoundingBox]]],
        model_1: str,
        model_2: str,
        model_3: str
    ):
        self.controller = controller
        self.integrator = integrator
        self.cache = detection_cache
        self.models = [model_1, model_2, model_3]
    
    def process(self, frame_idx: int) -> FrameResult:
        """1フレームを処理して結果を返す"""
        detections = {
            model: self.cache[model][frame_idx]
            for model in self.models
        }
        
        if self.controller.state == AdrodState.SOLO:
            return self._process_solo(detections)
        elif self.controller.state == AdrodState.PAIR:
            return self._process_pair(detections)
        else:
            return self._process_ensemble(detections)
    
    def _process_solo(self, detections: dict) -> FrameResult:
        """SOLO状態での処理"""
        base = self.models[0]
        self.controller.decide_state(
            detections=detections[base]
        )
        
        if self.controller.state == AdrodState.SOLO:
            return FrameResult(
                detections=detections[base],
                state=AdrodState.SOLO,
                flops=FLOPs_Dict[base]
            )
        else:
            return self._process_pair(detections)
    
    def _process_pair(self, detections: dict) -> FrameResult:
        """PAIR状態での処理"""
        m1, m2 = self.models[0], self.models[1]
        pair_dict = {m1: detections[m1], m2: detections[m2]}
        result, bbox_groups = self.integrator(pair_dict)
        
        self.controller.decide_state(bbox_groups=bbox_groups)
        
        if self.controller.state == AdrodState.ENSEMBLE:
            return self._process_ensemble(detections)
        else:
            return FrameResult(
                detections=result,
                state=AdrodState.PAIR,
                flops=FLOPs_Dict[m1] + FLOPs_Dict[m2]
            )
    
    def _process_ensemble(self, detections: dict) -> FrameResult:
        """ENSEMBLE状態での処理"""
        ensemble_dict = {
            model: detections[model]
            for model in self.models
        }
        result, bbox_groups = self.integrator(ensemble_dict)
        
        self.controller.decide_state(bbox_groups=bbox_groups)
        
        total_flops = sum(FLOPs_Dict[m] for m in self.models)
        return FrameResult(
            detections=result,
            state=AdrodState.ENSEMBLE,
            flops=total_flops
        )