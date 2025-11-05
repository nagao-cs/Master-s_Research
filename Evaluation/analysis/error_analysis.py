import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


def convert_detection_data_to_dataframe(det_dir: str) -> pd.DataFrame:
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
                confidence = float(parts[5])
                size = width * height * utils.IM_WIDTH * utils.IM_HEIGHT
                records.append({
                    'frame_id': frame_id,
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'size': size
                })
    df = pd.DataFrame.from_records(records)
    return df


def convert_gt_data_to_dataframe(gt_dir: str) -> pd.DataFrame:
    """
    検出結果のデータをDataFrameに変換する

    Args:
        gt_dir (str): 検出データのディレクトリパス

    Returns:
        pd.DataFrame: 検出結果を格納したDataFrame
    """
    from utils import utils
    records = list()
    for file_name in os.listdir(gt_dir):
        file_path = os.path.join(gt_dir, file_name)
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
                distance = 0.0
                size = width * height * utils.IM_WIDTH * utils.IM_HEIGHT
                records.append({
                    'frame_id': frame_id,
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'distance': distance,
                    'size': size
                })
    df = pd.DataFrame.from_records(records)
    return df


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


def error_analysis(gt_data: pd.DataFrame, det_data: pd.DataFrame):
    """
    エラー解析を実行する

    Args:
        gt_data (pd.DataFrame): 正解データのDataFrame
        det_data (pd.DataFrame): 検出データのDataFrame
    """
    from utils import utils
    fp_records = list()
    fn_records = list()
    for frame_id in gt_data['frame_id'].unique():
        gt_frame = gt_data[gt_data['frame_id'] == frame_id]
        det_frame = det_data[det_data['frame_id'] == frame_id]
        gt_used = set()
        det_used = set()
        for gt_index, gt_row in gt_frame.iterrows():
            matched = False
            max_iou = 0.0
            for det_index, det_row in det_frame.iterrows():
                if gt_row['class_id'] != det_row['class_id'] or det_index in det_used:
                    continue
                iou_value = utils.iou(gt_row[['x_center', 'y_center', 'width', 'height', 'distance']],
                                      det_row[['x_center', 'y_center', 'width', 'height', 'confidence']])
                if iou_value >= 0.5 and iou_value > max_iou:
                    matched = True
                    max_iou = iou_value
                    det = det_index
            if matched:
                det_used.add(det)
            if not matched:
                fn_records.append(gt_row.to_dict())
            gt_used.add(gt_index)

        for det_index, det_row in det_frame.iterrows():
            if det_index not in det_used:
                fp_records.append(det_row.to_dict())
    fp_df = pd.DataFrame.from_records(fp_records)
    fn_df = pd.DataFrame.from_records(fn_records)
    return fp_df, fn_df


def plot_error_statistics(fp_df: pd.DataFrame, fn_df: pd.DataFrame):
    """
    エラー統計をプロットする

    Args:
        fp_df (pd.DataFrame): 偽陽性のDataFrame
        fn_df (pd.DataFrame): 偽陰性のDataFrame
    """
    class_names = {
        0: 'Pedestrian',
        1: 'bicycle',
        2: 'Vehicle',
        9: 'Traffic Light',
        11: 'Traffic Sign'
    }

    # サブプロット数を2x2に変更
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # FPの信頼度とサイズの関係
    ax = axes[0][0]  # 2次元インデックスの正しい使用方法
    for class_id in class_names.keys():
        class_fp = fp_df[fp_df['class_id'] == class_id]
        if not class_fp.empty:
            ax.scatter(class_fp['confidence'], class_fp['size'],
                       label=class_names[class_id], alpha=0.6)
    ax.set_xlabel('confidence score')
    ax.set_ylabel('size')
    ax.set_title('correlation between FP size and confidence')
    ax.legend()

    # FNのサイズと距離の関係
    ax = axes[0][1]  # 2次元インデックスの正しい使用方法
    for class_id in class_names.keys():
        class_fn = fn_df[fn_df['class_id'] == class_id]
        if not class_fn.empty:
            ax.scatter(class_fn['distance'], class_fn['size'],
                       label=class_names[class_id], alpha=0.6)
    ax.set_xlabel('distance')
    ax.set_ylabel('size')
    ax.set_title('correlation between FN size and distance')
    ax.legend()

    # クラスごとのFP数
    ax = axes[1][0]  # 2次元インデックスの正しい使用方法
    fp_counts = fp_df['class_id'].value_counts()
    ax.bar([class_names[i] for i in fp_counts.index], fp_counts.values)
    ax.set_title('number of FP per class')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # クラスごとのFN数
    ax = axes[1][1]  # 2次元インデックスの正しい使用方法
    fn_counts = fn_df['class_id'].value_counts()
    ax.bar([class_names[i] for i in fn_counts.index], fn_counts.values)
    ax.set_title('number of FN per class')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(f'analysis detection error', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confidence_size_by_class(fp_df: pd.DataFrame):
    """
    クラスごとのconfidence scoreとサイズの関係を可視化する

    Args:
        fp_df (pd.DataFrame): 偽陽性のDataFrame
    """
    class_names = {
        0: 'Pedestrian',
        1: 'bicycle',
        2: 'Vehicle',
        9: 'Traffic Light',
        11: 'Traffic Sign'
    }

    # クラスの数に基づいてサブプロットを作成
    n_classes = len(class_names)
    n_cols = 2
    n_rows = (n_classes + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()  # 2次元配列を1次元に変換

    # クラスごとのプロット
    for idx, (class_id, class_name) in enumerate(class_names.items()):
        ax = axes[idx]
        class_data = fp_df[fp_df['class_id'] == class_id]

        if not class_data.empty:
            # 散布図
            ax.scatter(class_data['confidence'],
                       class_data['size'], alpha=0.6, c='blue')

            # 統計情報の追加
            mean_conf = class_data['confidence'].mean()
            mean_size = class_data['size'].mean()
            count = len(class_data)

            stats_text = f'number of fp: {count}\naverage conf: {mean_conf:.3f}\naverage size: {mean_size:.3f}'
            ax.text(0.05, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'{class_name}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Size')
        ax.grid(True, alpha=0.3)

    # 未使用のサブプロットを非表示
    for idx in range(len(class_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        'correlationo between size & conf score by class', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    import argparse
    argparser = argparse.ArgumentParser(
        description="Evaluate object detection results")
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03",
                 "Town04", "Town05", "Town10HD_Opt"],
        help="Map name: Town01, Town02, Town04, Town05, Town10HD",
        required=True
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr"],
    )
    args = argparser.parse_args()
    map = args.map
    model = args.model
    print(f"map: {map}")
    gt_dir = f'C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front'
    print(f"models: {model}")
    det_dir = f'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{model}/front'

    gt_df = convert_gt_data_to_dataframe(gt_dir)
    det_df = convert_detection_data_to_dataframe(det_dir)

    fp_df, fn_df = error_analysis(gt_df, det_df)
    plot_error_statistics(fp_df, fn_df)
    plot_confidence_size_by_class(fp_df)
