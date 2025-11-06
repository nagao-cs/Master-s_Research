from typing import Callable


class DetectionMetrics:
    def __init__(self):
        self.metrics = dict()  # 指標と関数のマッピング
        self.counters = {
            'intersection_fp': list(),
            'intersection_fn': list(),
            'union_fp': list(),
            'union_fn': list(),
            'total_instances': list(),
            'num_frames': 0,
            'num_inference': 0
        }

    def update_counters(self, analyzed_frame: dict, mode: str):
        intersection_fp = sum(
            len(boxes) for boxes in analyzed_frame['intersection_errors']['FP'].values())
        intersection_fn = sum(
            len(boxes) for boxes in analyzed_frame['intersection_errors']['FN'].values())
        union_fp = sum(len(boxes)
                       for boxes in analyzed_frame['union_errors']['FP'].values())
        union_fn = sum(len(boxes)
                       for boxes in analyzed_frame['union_errors']['FN'].values())
        total_instances = sum(len(boxes) for boxes in analyzed_frame['total_instances']['TP'].values()) + sum(
            len(boxes) for boxes in analyzed_frame['total_instances']['FN'].values()) + sum(
            len(boxes) for boxes in analyzed_frame['total_instances']['FP'].values())
        self.counters['intersection_fp'].append(intersection_fp)
        self.counters['intersection_fn'].append(intersection_fn)
        self.counters['union_fp'].append(union_fp)
        self.counters['union_fn'].append(union_fn)
        self.counters['total_instances'].append(total_instances)
        self.counters['num_frames'] += 1
        if mode == 'multi-version':
            self.counters['num_inference'] += 3
        else:
            self.counters['num_inference'] += 1

    def add_metric(self, func_name: str, func: Callable):
        self.metrics[func_name] = func

    def covod(self):
        IoE = 0.0
        for frame_idx in range(self.counters['num_frames']):
            IoE += (self.counters['intersection_fp'][frame_idx] +
                    self.counters['intersection_fn'][frame_idx]) / self.counters['total_instances'][frame_idx] if self.counters['total_instances'][frame_idx] > 0 else 0.0
        return 1 - (IoE / self.counters['num_frames'])

    def cerod(self):
        UoE = 0.0
        for frame_idx in range(self.counters['num_frames']):
            UoE += (self.counters['union_fp'][frame_idx] +
                    self.counters['union_fn'][frame_idx]) / self.counters['total_instances'][frame_idx] if self.counters['total_instances'][frame_idx] > 0 else 0.0
        return 1 - (UoE / self.counters['num_frames'])

    def precision(self):
        total_tp = sum(self.counters['total_instances'])
        total_fp = sum(self.counters['union_fp'])
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    def recall(self):
        total_tp = sum(self.counters['total_instances'])
        total_fn = sum(self.counters['union_fn'])
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    def compute(self):
        results = dict()
        for name, func in self.metrics.items():
            results[name] = func()
        return results

    def get_num_inference(self):
        return self.counters['num_inference']
