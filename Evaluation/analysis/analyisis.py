import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


class Analytics:
    def __init__(self, dataset):
        self.dataset = dataset

    def num_gt_instance(self):
        gt_instances = np.array(self.dataset.num_gt_list)
        plt.xlim(0, 30)
        plt.ylim(0, 2000)
        plt.title("num gt histgram", fontsize=20)
        plt.xlabel("num gt instance")
        plt.ylabel("freq")

        plt.hist(gt_instances)
        plt.show()

    def num_detection_instance(self):
        detection_instances = np.array(self.dataset.num_detection_dict[0])
        plt.xlim(0, 30)
        plt.ylim(0, 2000)
        plt.title("num detection histgram", fontsize=20)
        plt.xlabel("num detection instance")
        plt.ylabel("freq")

        plt.hist(detection_instances)
        plt.show()

    def Prob_detect_match_GT(self):
        detection_instances = np.array(self.dataset.num_detection_dict[0])
        gt_instances = np.array(self.dataset.num_gt_list)

        # GTとDetectionの差分を計算
        difference = gt_instances - detection_instances

        # ヒストグラムの設定
        plt.figure(figsize=(10, 6))
        plt.title("Difference between GT and Detection instances", fontsize=20)
        plt.xlabel("GT - Detection (Difference in number of instances)")
        plt.ylabel("Frequency")

        # ヒストグラムを描画
        plt.hist(difference, bins=range(
            min(difference)-1, max(difference)+2), align='mid')

        # x軸の中心に垂直線を追加（差分0の位置）
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        plt.grid(True, alpha=0.3)
        plt.show()

        # 統計情報を表示
        print(f"Mean difference: {np.mean(difference):.2f}")
        print(f"Median difference: {np.median(difference):.2f}")

    def detection_statistics(self):
        detection_instances = np.array(self.dataset.num_detection_dict[0])
        gt_instances = np.array(self.dataset.num_gt_list)

        stats = {
            "過検出率": len(np.where(detection_instances > gt_instances)[0]) / len(gt_instances),
            "検出漏れ率": len(np.where(detection_instances < gt_instances)[0]) / len(gt_instances),
            "完全一致率": len(np.where(detection_instances == gt_instances)[0]) / len(gt_instances)
        }

        return stats

    def plot_detection_over_time(self):
        frames = range(len(self.dataset.num_gt_list))
        gt_instances = np.array(self.dataset.num_gt_list)
        detection_instances = np.array(self.dataset.num_detection_dict[0])

        plt.figure(figsize=(15, 6))
        plt.plot(frames, gt_instances, label='Ground Truth', alpha=0.7)
        plt.plot(frames, detection_instances, label='Detection', alpha=0.7)
        plt.title("Detection Performance Over Time")
        plt.xlabel("Frame Number")
        plt.ylabel("Number of Instances")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        # plt.savefig("C:\CARLA_Latest\WindowsNoEditor/result.")

    def plot_confidence_statistics(self):
        """
        クラスごとのconfidence値の統計とヒストグラムを表示
        """
        class_conf_dict = defaultdict(list)
        for frame in range(self.dataset.num_frame):
            for class_id in [0, 2, 9, 11]:
                for box in self.dataset.results[0][frame]['TP'].get(class_id, []):
                    class_conf_dict[class_id].append(box[4])
                for box in self.dataset.results[0][frame]['FP'].get(class_id, []):
                    class_conf_dict[class_id].append(box[4])
        class_names = {
            0: 'Pedestrian',
            2: 'Vehicle',
            9: 'Traffic Light',
            11: 'Stop Sign'
        }

        # 0.1刻みのビンを定義
        bins = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 0.9, 1.0]

        # 統計情報を表示
        print("\n--- クラスごとの検出confidence統計 ---")
        print(f"{'クラス名':<15} {'検出数':<8} {'平均conf':<10} {'最小conf':<10} {'最大conf':<10}")
        print("-" * 60)

        # ヒストグラムの描画設定
        n_classes = len(class_conf_dict)
        n_cols = 2
        n_rows = (n_classes + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()

        for idx, (class_id, confs) in enumerate(class_conf_dict.items()):
            if not confs:  # 検出が0件の場合をスキップ
                continue

            confs = np.array(confs)
            class_name = class_names.get(class_id, f'Class {class_id}')

            # 統計情報を表示
            print(
                f"{class_name:<15} {len(confs):<8d} {confs.mean():.3f}     {confs.min():.3f}     {confs.max():.3f}")

            # ヒストグラムの描画
            ax = axes[idx]
            ax.hist(confs, bins=bins)

            ax.set_title(f'{class_name} Confidence Distribution')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')

        # 未使用のサブプロットを非表示
        for idx in range(len(class_conf_dict), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def class_wise_fn_rate(self):
        class_names = {
            0: 'Pedestrian',
            2: 'Vehicle',
            9: 'Traffic Light',
            11: 'Stop Sign'
        }
        term_frequency = defaultdict(float)
        for class_id in class_names.keys():
            pass


if __name__ == "__main__":
    # main()
    import sys
    debug = True if len(sys.argv) > 1 and (sys.argv[1] == 'debug') else False

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
        required=False,
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr"],
    )
    args = argparser.parse_args()
    debug = args.debug
    map = args.map
    models = args.models
    if map:
        print(f"map: {map}")
        gt_dir = f'C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front'
    if models:
        version = len(models)
        print(f"models: {models}")
        det_dirs = [
            f'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{model}/front' for model in models]
    from utils import dataset
    ds = dataset.Dataset(gt_dir, det_dirs, version, debug)
    analyzer = Analytics(ds)
    # analyzer.num_gt_instance()
    # analyzer.num_detection_instance()
    # analyzer.Prob_detect_match_GT()
    analyzer.plot_detection_over_time()
    # print(analyzer.detection_statistics())
    analyzer.plot_confidence_statistics()
