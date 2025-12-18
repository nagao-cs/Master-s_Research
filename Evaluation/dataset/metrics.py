from typing import Callable, Dict, List


class DetectionMetrics:
    def __init__(self):
        self.metrics = dict()  # 指標と関数のマッピング
        self.counters = {
            'intersection_fp': list(),
            'intersection_fn': list(),
            'union_fp': list(),
            'union_fn': list(),
            'total_instances': list(),
            'tp_instances': list(),
            'fp_instances': list(),
            'fn_instances': list(),
            'num_frames': 0,
            'num_inference': 0
        }

    def update_reliability_counters(self, analyzed_frame: dict, mode: str):
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
        tp_instance = sum(len(boxes)
                          for boxes in analyzed_frame['total_instances']['TP'].values())
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

    def update_accuracy_counter(self, analyze_frame: Dict[str, Dict[int, List]]) -> None:
        self.counters['tp_instances'].append(
            sum(len(boxes) for boxes in analyze_frame['TP'].values()))
        self.counters['fp_instances'].append(
            sum(len(boxes) for boxes in analyze_frame['FP'].values()))
        self.counters['fn_instances'].append(
            sum(len(boxes) for boxes in analyze_frame['FN'].values()))

    def add_metric(self, func_name: str, func: Callable):
        self.metrics[func_name] = func

    def precision_recall_by_num_detection(self):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        # --- フレームごとのTP/FP/FNが既に集計されている前提 ---
        tp_list = self.counters['tp_instances']
        fp_list = self.counters['fp_instances']
        fn_list = self.counters['fn_instances']

        # --- n_detを計算（検出数 = TP + FP） ---
        n_det_list = [tp + fp for tp, fp in zip(tp_list, fp_list)]

        # --- DataFrame化 ---
        df = pd.DataFrame({
            'n_det': n_det_list,
            'tp': tp_list,
            'fp': fp_list,
            'fn': fn_list
        })

        # --- フレーム単位でPrecision, Recallを計算 ---
        df['precision'] = df['tp'] / (df['tp'] + df['fp']).replace(0, np.nan)
        df['recall'] = df['tp'] / (df['tp'] + df['fn']).replace(0, np.nan)
        df['f1-score'] = df['tp'] / (df['tp'] + 0.5 * (df['fp'] + df['fn']))
        # --- n_detごとに平均を取る ---
        grouped = df.groupby('n_det').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1-score': 'mean',
            'tp': 'count'  # サンプル数確認用
        }).reset_index()

        # --- 可視化 ---
        plt.figure(figsize=(7, 5))
        plt.plot(grouped['n_det'], grouped['precision'],
                 marker='o', label='Precision')
        # plt.plot(grouped['n_det'], grouped['recall'],
        #  marker='s', label='Recall')
        # plt.plot(grouped['n_det'], grouped['f1-score'],
        #  marker='o', label='f1-score')
        plt.xlabel('n_det')
        plt.ylabel('Score')
        plt.title('Precision vs n_det')
        plt.legend()
        plt.grid(True)
        plt.show()

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
        total_tp = sum(self.counters['tp_instances'])
        total_fp = sum(self.counters['fp_instances'])

        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    def recall(self):
        total_tp = sum(self.counters['tp_instances'])
        total_fn = sum(self.counters['fn_instances'])
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    def accuracy(self):
        total_tp = sum(self.counters['tp_instances'])
        total_fp = sum(self.counters['fp_instances'])
        total_fn = sum(self.counters['fn_instances'])
        return total_tp / (total_tp + total_fp + total_fn)

    def compute(self):
        results = dict()
        for name, func in self.metrics.items():
            results[name] = func()
        return results

    def get_num_inference(self):
        return self.counters['num_inference']
