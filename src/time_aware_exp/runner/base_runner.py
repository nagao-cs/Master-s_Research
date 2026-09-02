import os
from pathlib                    import Path
import csv

from ..config.config            import AdrodConfig
from ..state.stateContext       import StateContext
from ..executionRecorder        import ExecutionRecorder
from ..data.dataset             import BaseDataset
from src.file_lib.file_writer   import FileWriter
from src.eval_lib.evaluator     import Evaluator


class BaseRunner:
    def __init__(self, cfg: AdrodConfig, base_dir: Path, cfg_path: Path):
        self.cfg      = cfg
        self.base_dir = base_dir
        self.cfg_path = cfg_path

    def run(self):
        self.dataset  =  self.build_dataset()
        self.context  =  self.build_context()
        self.recorder = ExecutionRecorder()
        self.context.ready_model()

        self.recorder.set_start_time()
        self.execute_detection()
        self.recorder.set_end_time()
        print(f"\nElapsed: {self.recorder.get_execution_time()}s")

        self.save_result()

    def build_dataset(self) -> BaseDataset:
        """サブクラスで使用する Dataset を生成して返す。"""
        raise NotImplementedError

    def execute_detection(self):
        raise NotImplementedError

    def build_context(self):
        from ..factory.context_factory import build_context

        context: StateContext = build_context(self.cfg)
        return context

    def save_result(self):
        output_dir = self.cfg_path.parent / "result"
        fig_dir = self.cfg_path.parent / "figure"
        csv_path = self.cfg_path.parent / "result.csv"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        writer = FileWriter(output_dir=output_dir)
        for idx, dets in enumerate(self.recorder.get_detections()):
            writer.write(file_name=f"{idx:06d}.txt", detections=dets)
            
        self.evaluate()

        stats = self.recorder.get_statistics()
        state_distribution = stats["state_distribution"]
        state_transition_graph = self.recorder.draw_state_transition_graph()
        state_transition_graph.savefig(fig_dir / "state_transition.png", dpi=300, bbox_inches='tight')
        print(f"State distribution : {state_distribution}")
        print(f"Cost               : {stats['flops_cost']:.2f}x YOLOv8n")
        
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "F1", "prec", "rec", 
                "mAP", 'AP_pedestrian', 'AP_vehicle', 'AP_traffic_light', 'AP_traffic_sign',
                "exe_time", "state_1", "state_2", "state_3", "cost"
            ])
            writer.writerow([
                self.result.f1_score, 
                self.result.precision, 
                self.result.recall, 
                self.result.mAP, 
                self.result.class_ap_dict.get(0, 0), 
                self.result.class_ap_dict.get(2, 0), 
                self.result.class_ap_dict.get(9, 0), 
                self.result.class_ap_dict.get(11, 0), 
                self.recorder.get_execution_time(), 
                state_distribution.get(1, 0), 
                state_distribution.get(2, 0), 
                state_distribution.get(3, 0), 
                stats["flops_cost"]
            ])
        

    def evaluate(self):
        output_dir = self.cfg_path.parent / "result"
        
        if self.cfg.dataset == "KITTI":
            target_class_ids = [0, 2]
        else :
            target_class_ids = [0, 2, 9, 11]

        evaluator = Evaluator(iou_threshold=self.cfg.iou_threshold)
        result = evaluator.evaluate(
            gt_dataset_dir=self.dataset.gt_dir,
            detection_dataset_dir=output_dir,
            target_class_ids=target_class_ids
        )

        self.result = result
