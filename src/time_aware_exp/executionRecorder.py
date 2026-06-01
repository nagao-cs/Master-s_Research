"""
executionRecorder.py
フレーム処理結果を集約・記録する。
State パターン適用後も FrameResult の構造が同じであれば変更不要。
"""
import time

from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.boundingBox.boundingBox import DetectionBoundingBox
from .state.AdrodState import AdrodState, FrameResult


class ExecutionRecorder:
    """フレーム処理結果を集約・記録"""

    def __init__(self) -> None:
        self.detection_record:  list[list[DetectionBoundingBox]] = []
        self.exe_state_record:  list[AdrodState] = []
        self.total_flops:       float = 0.0
        self.yolov8n_cost:      float = 17400.0
        self.start = 0.0
        self.end = 0.0

    def record_frame_result(self, frame_result: FrameResult) -> None:
        self.detection_record.append(frame_result.detections)
        self.exe_state_record.append(frame_result.state)
        self.total_flops += frame_result.flops
    
    def draw_state_transition_graph(self) -> Figure:
        num_frame = len(self.exe_state_record)
        x = np.arange(start=1, step=1, stop=num_frame+1)
        y = self.exe_state_record
        
        # 1. 新しいFigure（描画領域）を作成
        fig = plt.figure()
        ax = plt.axes()
        
        # 2. plt.step を使って階段状に描画
        ax.step(x, y, where='post')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('State')
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(True, linestyle='--')
        
        # 3. 完成した Figure オブジェクトを返す
        return fig

    def get_detections(self) -> list[list[DetectionBoundingBox]]:
        return self.detection_record

    def get_execution_states(self) -> list[AdrodState]:
        return self.exe_state_record

    def get_total_flops(self) -> float:
        return self.total_flops

    def get_statistics(self) -> dict:
        state_counter = Counter(self.exe_state_record)
        return {
            "state_distribution": dict(state_counter),
            "total_frames":       len(self.detection_record),
            "total_flops":        self.total_flops,
            "flops_cost":         self.total_flops / self.yolov8n_cost,
        }

    def set_start_time(self):
        self.start = time.time()
    
    def set_end_time(self):
        self.end = time.time()
    
    def get_execution_time(self):
        return self.end - self.start
    
    def reset(self) -> None:
        self.detection_record = []
        self.exe_state_record = []
        self.total_flops = 0.0