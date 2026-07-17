from collections import Counter

from src.boundingBox.boundingBox import DetectionBoundingBox
from .stateController import AdrodState
from .detectionExecuter import FrameResult



class ExecutionRecorder:
    """フレーム処理結果を集約・記録"""
    
    def __init__(self):
        """実行記録を初期化"""
        self.detection_record: list[list[DetectionBoundingBox]] = [] # 各フレームでの最終的な検出結果
        self.exe_state_record: list[AdrodState] = []
        self.total_flops: float = 0.0
        
        self.yolov8n_cost = 17400
    
    def record_frame_result(self, frame_result: FrameResult):
        """
        1フレームの処理結果を記録
        
        Args:
            frame_result: FrameProcessor.process() の戻り値
        """
        self.detection_record.append(frame_result.detections)
        self.exe_state_record.append(frame_result.state)
        self.total_flops += frame_result.flops
    
    def get_detections(self) -> list[list[DetectionBoundingBox]]:
        """記録された全フレームの検出結果を取得"""
        return self.detection_record
    
    def get_execution_states(self) -> list[AdrodState]:
        """記録された全フレームの実行状態を取得"""
        return self.exe_state_record
    
    def get_total_flops(self) -> float:
        """総計算量を取得"""
        return self.total_flops
    
    def get_statistics(self) -> dict:
        """実行統計を計算"""
        state_counter = Counter(self.exe_state_record)
        
        return {
            'state_distribution': dict(state_counter),
            'total_frames': len(self.detection_record),
            'total_flops': self.total_flops,
            'flops_cost': self.total_flops / self.yolov8n_cost,
        }
    
    def reset(self):
        """記録をリセット"""
        self.detection_record = []
        self.exe_state_record = []
        self.total_flops = 0.0