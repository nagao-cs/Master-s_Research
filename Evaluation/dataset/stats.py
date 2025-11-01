import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DetectionStats:
    def __init__(self):
        self.dataframe = pd.DataFrame(columns=[
            'frame_idx',
            'num_gt_instances',
            'num_detection_instances',
            'num_true_positives',
            'num_false_positives',
            'num_false_negatives',
        ])

    def update(self, frame_idx: int, gt: dict, analyzed_detctions: dict):
        num_gt_instances = sum(len(boxes) for boxes in gt.values())
        num_detection_instances = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['TP'].values()) + sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['FP'].values())
        num_true_positives = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['TP'].values())
        num_false_positives = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['FP'].values())
        num_false_negatives = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['FN'].values())

        new_row = {
            'frame_idx': frame_idx,
            'num_gt_instances': num_gt_instances,
            'num_detection_instances': num_detection_instances,
            'num_true_positives': num_true_positives,
            'num_false_positives': num_false_positives,
            'num_false_negatives': num_false_negatives,
        }
        self.dataframe = pd.concat(
            [self.dataframe, pd.DataFrame([new_row])], ignore_index=True)

    def plot_transition(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.dataframe['frame_idx'], self.dataframe['num_gt_instances'],
                 label='GT Instances', color='blue')
        plt.plot(self.dataframe['frame_idx'], self.dataframe['num_detection_instances'],
                 label='Detection Instances', color='orange')

        plt.xlabel('Frame Index')
        plt.ylabel('Number of Instances')
        plt.title('Detection Statistics Over Time')
        plt.legend()
        plt.grid()
        plt.show()
