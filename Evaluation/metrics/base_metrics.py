from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict
sys.path.append(str(Path(__file__).resolve().parent.parent))


def convert_original_data_to_dataframe(det_dir: str) -> pd.DataFrame:
    """
    検出結果のデータをDataFrameに変換する

    Args:
        det_dir (str): 検出データのディレクトリパス

    Returns:
        pd.DataFrame: 検出結果を格納したDataFrame
    """
    from utils import utils
    records = list()
    for file_name in os.listdir(det_dir):
        file_path = os.path.join(det_dir, file_name)
        frame_id = int(file_name.split('.')[0])
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                class_id = utils.class_Map.get((int(parts[0])), -1)
                if class_id == -1:
                    continue
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                # confidence = float(parts[5])
                records.append({
                    'frame_id': frame_id,
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    # 'confidence': confidence
                })
    df = pd.DataFrame.from_records(records)
    return df


def calculate_precision_recall(gt_dir, det_dir):
    """
    精度と再現率を計算する

    Args:
        gt_dir (str): 正解データのディレクトリパス
        det_dir (str): 検出データのディレクトリパス

    Returns:
        precision (float): 精度
        recall (float): 再現率
    """
    def iou(box1, box2):
        ax_center, ay_center, a_width, a_height = box1
        bx_center, by_center, b_width, b_height = box2
        axmin = ax_center - a_width / 2
        axmax = ax_center + a_width / 2
        aymin = ay_center - a_height / 2
        aymax = ay_center + a_height / 2
        bxmin = bx_center - b_width / 2
        bxmax = bx_center + b_width / 2
        bymin = by_center - b_height / 2
        bymax = by_center + b_height / 2
        area_a = (axmax - axmin) * (aymax - aymin)
        area_b = (bxmax - bxmin) * (bymax - bymin)

        abxmin = max(axmin, bxmin)
        abxmax = min(axmax, bxmax)
        abymin = max(aymin, bymin)
        abymax = min(aymax, bymax)
        intersection = max(0, abxmax - abxmin) * max(0, abymax - abymin)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0
    precision = 0.0
    recall = 0.0

    tp, fp, fn = 0, 0, 0
    gt_data = convert_original_data_to_dataframe(gt_dir)
    det_data = convert_original_data_to_dataframe(det_dir)

    for frame_id in gt_data['frame_id'].unique():
        gt_frame = gt_data[gt_data['frame_id'] == frame_id]
        det_frame = det_data[det_data['frame_id'] == frame_id]
        gt_used = set()
        det_used = set()
        for gt_index, gt_row in gt_frame.iterrows():
            matched = False
            max_iou = 0.0
            # print(gt_row[['x_center', 'y_center', 'width', 'height']])
            for det_index, det_row in det_frame.iterrows():
                if gt_row['class_id'] != det_row['class_id'] or det_index in det_used:
                    continue
                iou_value = iou(gt_row[['x_center', 'y_center', 'width', 'height']],
                                det_row[['x_center', 'y_center', 'width', 'height']])
                if iou_value >= 0.5 and iou_value > max_iou:
                    matched = True
                    max_iou = iou_value
                    det = det_index
            if matched:
                tp += 1
                det_used.add(det)
            if not matched:
                fn += 1
            gt_used.add(gt_index)

        for det_index, det_row in det_frame.iterrows():
            if det_index not in det_used:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Evaluate object detection results")
    argparser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode"
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03",
                 "Town04", "Town05", "Town10HD_Opt"],
        help="Map name: Town01, Town02, Town04, Town05, Town10HD",
        required=True
    )
    argparser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr"],
    )
    args = argparser.parse_args()
    debug = args.debug
    map = args.map
    models = args.models
    models.sort()
    gt_directory = f"C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front"
    if len(models) == 1:
        det_directory = f"C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{models[0]}/front"

    else:
        det_directory = f"C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/integrated_output/{map}_{'_'.join(models)}"
    precision, recall = calculate_precision_recall(gt_directory, det_directory)
    print(f"Precision: {precision}, Recall: {recall}")
