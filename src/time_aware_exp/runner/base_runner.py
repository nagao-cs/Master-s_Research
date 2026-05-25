import os
import time
from pathlib import Path

from tqdm import tqdm

from ..config.config import AdrodConfig
from ..state.stateContext import StateContext

from ..executionRecorder import ExecutionRecorder
from src.file_lib.file_writer import FileWriter
from src.eval_lib.evaluator import Evaluator


class BaseRunner:
    def __init__(self, cfg: AdrodConfig, base_dir: Path, cfg_path: Path):
        self.cfg = cfg
        self.base_dir = base_dir
        self.cfg_path = cfg_path

    def run(self):
        self.frame_source = self.build_detection_source()
        self.context = self.build_context()

        self.recorder = ExecutionRecorder()

        start = time.time()

        self.execute_detection()

        print(f"\nElapsed: {time.time() - start:.2f}s")

        self.save_result(self.recorder)

        self.evaluate()
    
    def execute_detection(self):
        raise NotImplementedError()

    def build_detection_source(self):
        raise NotImplementedError()

    def build_context(self):
        from ..factory.context_factory import build_context
        
        context: StateContext = build_context(self.cfg)
        return context

    def save_result(self, recorder):

        output_dir = self.cfg_path.parent / "result"

        os.makedirs(output_dir, exist_ok=True)

        writer = FileWriter(output_dir=output_dir)

        for idx, dets in enumerate(recorder.get_detections()):

            writer.write(
                file_name=f"{idx:06d}.txt",
                detections=dets
            )

        stats = recorder.get_statistics()

        print(f"State distribution : {stats['state_distribution']}")
        print(f"Cost               : {stats['flops_cost']:.2f}x YOLOv8n")

    def evaluate(self):

        output_dir = self.cfg_path.parent / "result"

        evaluator = Evaluator(
            iou_threshold=self.cfg.iou_threshold
        )

        result = evaluator.evaluate(
            gt_dataset_dir=(
                self.base_dir
                / "output"
                / "label"
                / f"{self.cfg.map}"
                / "front"
            ),
            detection_dataset_dir=output_dir,
        )

        print(f"mAP = {result.mAP}")
        print(f"AP = {result.class_ap_dict}")
        print(f"f1 = {result.f1_score}")
        print(f"precision = {result.precision}")
        print(f"recall = {result.recall}")