import os
import pickle
import utils
from utils import classify
from utils import dataset


class Evaluation:
    def __init__(self, dataset):
        self.dataset = dataset

    def pricision(self):
        pass

    def cov_od(self):
        total_objs = self.dataset.total_obj_list
        common_fps = self.dataset.common_fp_list
        common_fns = self.dataset.common_fn_list
        cov_od = 0.0
        for frame in range(self.dataset.num_frame):
            frame_obj = total_objs[frame]
            frame_common_fp = common_fps[frame]
            frame_common_fn = common_fns[frame]
            num_fp = sum(len(bboxes) for bboxes in frame_common_fp.values())
            num_fn = sum(len(bboxes) for bboxes in frame_common_fn.values())
            cov_od += ((num_fp + num_fn) /
                       frame_obj) if frame_obj > 0 else 0
        cov_od = 1 - (cov_od/self.dataset.num_frame)
        return cov_od

    def cer_od(self):
        total_objs = self.dataset.total_obj_list
        all_fps = self.dataset.all_fp_list
        all_fns = self.dataset.all_fn_list
        cer_od = 0.0
        avg_all_fp = 0
        for frame in range(self.dataset.num_frame):
            frame_obj = total_objs[frame]
            frame_all_fp = all_fps[frame]
            frame_all_fn = all_fns[frame]
            num_fp = sum(len(bboxes) for bboxes in frame_all_fp.values())
            num_fn = sum(len(bboxes) for bboxes in frame_all_fn.values())
            avg_all_fp += num_fp
            cer_od += (num_fp + num_fn) / frame_obj if frame_obj > 0 else 0

        cer_od = 1 - (cer_od/self.dataset.num_frame)
        # print(f"avg_all_fp = {avg_all_fp/self.dataset.num_frame}")
        return cer_od

    def gt_based_adaptive_cov_od(self, threshold):
        total_objs = self.dataset.total_obj_list
        common_fps = self.dataset.common_fp_list
        common_fns = self.dataset.common_fn_list
        adaptive_cov_od = 0.0
        count_inference = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]

            if num_gt >= threshold:
                frame_obj = total_objs[frame]
                frame_common_fp = common_fps[frame]
                frame_common_fn = common_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_common_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_common_fn.values())
                adaptive_cov_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                count_inference += self.dataset.num_version
            else:
                num_fp = len(self.dataset.fp_detection(version=0, frame=frame))
                num_fn = len(self.dataset.fn_detection(version=0, frame=frame))
                adaptive_cov_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                count_inference += 1
        adaptive_cov_od = 1 - (adaptive_cov_od/self.dataset.num_frame)
        print(f"        gt-based 推論回数: {count_inference}")
        return adaptive_cov_od

    def gt_based_adaptive_cer_od(self, threshold):
        total_objs = self.dataset.total_obj_list
        all_fps = self.dataset.all_fp_list
        all_fns = self.dataset.all_fn_list
        adaptive_cer_od = 0.0
        count_inference = 0
        avg_all_fp = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]

            if num_gt >= threshold:
                frame_obj = total_objs[frame]
                frame_all_fp = all_fps[frame]
                frame_all_fn = all_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_all_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_all_fn.values())
                adaptive_cer_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                count_inference += self.dataset.num_version
            else:
                num_fp = len(self.dataset.fp_detection(version=0, frame=frame))
                num_fn = len(self.dataset.fn_detection(version=0, frame=frame))
                adaptive_cer_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                count_inference += 1
            avg_all_fp += num_fp
        # print(f"avg_all_fp:{avg_all_fp/self.dataset.num_frame}")
        adaptive_cer_od = 1 - (adaptive_cer_od/self.dataset.num_frame)
        # print(f"adaptive_conut: {adaptive_conut}")
        return adaptive_cer_od, count_inference

    def avg_accuracy(self):
        results = self.dataset.results
        total_accuracy = 0.0
        for frame in range(self.dataset.num_frame):
            frame_accuracy = 0.0
            for version in range(self.dataset.num_version):
                TP = sum(len(v)
                         for v in results[version][frame]['TP'].values())
                FP = sum(len(v)
                         for v in results[version][frame]['FP'].values())
                FN = sum(len(v)
                         for v in results[version][frame]['FN'].values())
                accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
                frame_accuracy += accuracy
            frame_accuracy /= self.dataset.num_version
            total_accuracy += frame_accuracy
        avg_accuracy = total_accuracy / self.dataset.num_frame
        return avg_accuracy

    def detect_based_adaptive_cov_od(self, adaptive_threshold):
        total_objs = self.dataset.total_obj_list
        common_fps = self.dataset.common_fp_list
        common_fns = self.dataset.common_fn_list
        adaptive_cov_od = 0.0
        count_inference = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]
            num_detection = self.dataset.num_detection_dict[0][frame]

            if num_detection >= adaptive_threshold:
                frame_obj = total_objs[frame]
                frame_common_fp = common_fps[frame]
                frame_common_fn = common_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_common_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_common_fn.values())
                adaptive_cov_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                count_inference += self.dataset.num_version
            else:
                num_fp = len(self.dataset.fp_detection(version=0, frame=frame))
                num_fn = len(self.dataset.fn_detection(version=0, frame=frame))
                adaptive_cov_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                count_inference += 1
        adaptive_cov_od = 1 - (adaptive_cov_od/self.dataset.num_frame)
        print(
            f"        detection_based 推論回数: {count_inference} (num_obj_threshold={adaptive_threshold})")
        return adaptive_cov_od

    def detect_based_adaptive_cer_od(self, adaptive_threshold):
        total_objs = self.dataset.total_obj_list
        all_fps = self.dataset.all_fp_list
        all_fns = self.dataset.all_fn_list
        adaptive_cer_od = 0.0
        count_inference = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]
            num_detection = self.dataset.num_detection_dict[0][frame]
            if num_detection >= adaptive_threshold:
                frame_obj = total_objs[frame]
                frame_all_fp = all_fps[frame]
                frame_all_fn = all_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_all_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_all_fn.values())
                adaptive_cer_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                count_inference += self.dataset.num_version
            else:
                num_fp = len(self.dataset.fp_detection(version=0, frame=frame))
                num_fn = len(self.dataset.fn_detection(version=0, frame=frame))
                adaptive_cer_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                count_inference += 1
        adaptive_cer_od = 1 - (adaptive_cer_od/self.dataset.num_frame)
        return adaptive_cer_od, count_inference

    def conf_adaptive_cov_od(self, k, conf_threshold):
        total_objs = self.dataset.total_obj_list
        common_fps = self.dataset.common_fp_list
        common_fns = self.dataset.common_fn_list
        adaptive_cov_od = 0.0
        count_inference = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]
            topK_detection = self.dataset.topK_detection(
                version=0, frame=frame, k=k)
            if len(topK_detection) > 0:
                avg_conf = sum(box[4]
                               for box in topK_detection) / len(topK_detection)
            else:
                avg_conf = 1

            if avg_conf < conf_threshold:
                frame_obj = total_objs[frame]
                frame_common_fp = common_fps[frame]
                frame_common_fn = common_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_common_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_common_fn.values())
                adaptive_cov_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                count_inference += self.dataset.num_version
            else:
                num_fp = len(self.dataset.fp_detection(version=0, frame=frame))
                num_fn = len(self.dataset.fn_detection(version=0, frame=frame))
                adaptive_cov_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                count_inference += 1
        adaptive_cov_od = 1 - (adaptive_cov_od/self.dataset.num_frame)
        print(
            f"        topK conf 推論回数: {count_inference} (k={k}, conf_threshold={conf_threshold})")
        return adaptive_cov_od

    def conf_adaptive_cer_od(self, k, conf_threshold):
        total_objs = self.dataset.total_obj_list
        all_fps = self.dataset.all_fp_list
        all_fns = self.dataset.all_fn_list
        adaptive_cer_od = 0.0
        count_inference = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]
            topK_detection = self.dataset.topK_detection(
                version=0, frame=frame, k=k)
            if len(topK_detection) > 0:
                avg_conf = sum(box[4]
                               for box in topK_detection) / len(topK_detection)
            else:
                avg_conf = 1
            if avg_conf < conf_threshold:
                frame_obj = total_objs[frame]
                frame_all_fp = all_fps[frame]
                frame_all_fn = all_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_all_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_all_fn.values())
                adaptive_cer_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                count_inference += self.dataset.num_version
            else:
                num_fp = len(self.dataset.fp_detection(version=0, frame=frame))
                num_fn = len(self.dataset.fn_detection(version=0, frame=frame))
                adaptive_cer_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                count_inference += 1
        adaptive_cer_od = 1 - (adaptive_cer_od/self.dataset.num_frame)
        return adaptive_cer_od, count_inference


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
        required=True,
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr"],
    )
    argparser.add_argument(
        "--numobj",
        type=int,
        default=utils.ADAPTIVE_THRESHOLD,
        required=False
    )
    argparser.add_argument(
        "--k",
        type=int,
        default=5
    )
    argparser.add_argument(
        "--conf",
        type=float,
        default=0.7,
    )
    args = argparser.parse_args()
    debug = args.debug
    map = args.map
    models = args.models
    numobj_threshold = args.numobj
    k = args.k
    conf_threshold = args.conf
    print(f"map: {map}")
    gt_dir = f'C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front'
    version = len(models)
    print(f"models: {models}")
    det_dirs = [
        f'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{model}/front' for model in models]
    ds = dataset.Dataset(gt_dir, det_dirs, version, debug)
    eval_instance = Evaluation(ds)
    # print(f"    cov_od: {Evaluation(ds).cov_od()}")
    # print(
    # f"    gt_based_adaptive_cov_od: {Evaluation(ds).gt_based_adaptive_cov_od()}")
    # print(
    # f"    detection_based_adaptive_cov_od: {Evaluation(ds).detect_based_adaptive_cov_od(numobj_threshold)}")
    # print(
    # f"    topK conf cov_od: {Evaluation(ds).conf_adaptive_cov_od(k, conf_threshold)}")
    print()
    # print(f"    cer_od: {Evaluation(ds).cer_od()}")
    # print(
    # f"    gt_based_adaptive_cer_od: {Evaluation(ds).gt_based_adaptive_cer_od()} ")
    # print(
    # f"    detection_based_adaptive_cer_od: {Evaluation(ds).detect_based_adaptive_cer_od(numobj_threshold)}")
    # print(
    # f"    topK conf cov_od: {Evaluation(ds).conf_adaptive_cer_od(k, conf_threshold)}")
    # print()
    # # # print(
    # # # f"            avg_accuracy: {Evaluation(ds).avg_accuracy()}")

    # # 閾値とCOV-ODの関係を評価
    import matplotlib.pyplot as plt
    import numpy as np
    # # 閾値とCOV-OD, CER-OD, 推論回数の関係を評価
    # thresholds = np.arange(0.2, 1.1, 0.1)
    # cov_od_values = []
    # cer_od_values = []
    # detection_counts = []

    # eval_instance = Evaluation(ds)
    # for threshold in thresholds:
    #     cov_od = eval_instance.conf_adaptive_cov_od(
    #         k=k, conf_threshold=threshold)
    #     cer_od, num_inf = eval_instance.conf_adaptive_cer_od(
    #         k=k, conf_threshold=threshold)
    #     # 推論回数を計算（実装に応じて修正が必要）
    #     detection_counts.append(num_inf)

    #     cov_od_values.append(cov_od)
    #     cer_od_values.append(cer_od)
    #     print(
    #         f"閾値 {threshold}: COV-OD = {cov_od:.4f}, CER-OD = {cer_od:.4f}, 推論回数 = {detection_counts}")

    # # グラフの描画（2つのサブプロット）
    # fig, (ax1, ax2) = plt.subplots(
    #     2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # # 上部プロット（COV-ODとCER-OD）
    # ax1.plot(thresholds, cov_od_values, 'b-o',
    #          linewidth=2, markersize=8, label='Cov-OD')
    # ax1.plot(thresholds, cer_od_values, 'r-^',
    #          linewidth=2, markersize=8, label='Cer-OD')
    # ax1.grid(True, alpha=0.3)
    # ax1.set_title(
    #     "Confidence Threshold & Conf based Cov-OD/Cer-OD", fontsize=14)
    # ax1.set_ylabel("value", fontsize=12)
    # ax1.set_ylim(0, 1)
    # ax1.set_xlim(0.1, 1.1)  # x軸の範囲を設定
    # ax1.legend(fontsize=12)

    # # データ点の値を表示（上部プロット）
    # for x, y1, y2 in zip(thresholds, cov_od_values, cer_od_values):
    #     ax1.annotate(f'{y1:.3f}', (x, y1), textcoords="offset points",
    #                  xytext=(0, 10), ha='center', color='blue')
    #     ax1.annotate(f'{y2:.3f}', (x, y2), textcoords="offset points",
    #                  xytext=(0, -20), ha='center', color='red')

    # # 下部プロット（推論回数）
    # width = 0.08  # バーの幅を調整
    # ax2.bar(thresholds, detection_counts,
    #         width=width, color='green', alpha=0.6)
    # ax2.grid(True, alpha=0.3)
    # ax2.set_xlabel("conf threshold", fontsize=12)
    # ax2.set_ylabel("Inference count", fontsize=12)
    # ax2.set_xlim(0.1, 1.1)  # x軸の範囲を上部プロットと合わせる

    # # データ点の値を表示（下部プロット）
    # for x, y in zip(thresholds, detection_counts):
    #     ax2.annotate(f'{y}', (x, y), textcoords="offset points",
    #                  xytext=(0, 5), ha='center', color='darkgreen')

    # # x軸のティックを設定（両方のプロット）
    # xticks = np.arange(0.2, 1.1, 0.1)
    # ax1.set_xticks(xticks)
    # ax2.set_xticks(xticks)
    # ax1.set_xticklabels([f'{x:.1f}' for x in xticks])
    # ax2.set_xticklabels([f'{x:.1f}' for x in xticks])

    # plt.tight_layout()

    # # グラフの保存（オプション）
    # # plt.savefig('adaptive_threshold_analysis_with_counts.png', dpi=300, bbox_inches='tight')

    # plt.show()
    # 閾値とCOV-OD, CER-OD, 推論回数の関係を評価
    thresholds = range(1, 21)
    cov_od_values = []
    cer_od_values = []
    detection_counts = []

    eval_instance = Evaluation(ds)
    for threshold in thresholds:
        cov_od = eval_instance.gt_based_adaptive_cov_od(threshold)
        cer_od, detection_count = eval_instance.gt_based_adaptive_cer_od(
            threshold)
        # 推論回数を計算（実装に応じて修正が必要）

        cov_od_values.append(cov_od)
        cer_od_values.append(cer_od)
        detection_counts.append(detection_count)
        print(
            f"閾値 {threshold}: COV-OD = {cov_od:.4f}, CER-OD = {cer_od:.4f}, 推論回数 = {detection_count}")

    # グラフの描画（2つのサブプロット）
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # 上部プロット（COV-ODとCER-OD）
    ax1.plot(thresholds, cov_od_values, 'b-o',
             linewidth=2, markersize=8, label='COV-OD')
    ax1.plot(thresholds, cer_od_values, 'r-^',
             linewidth=2, markersize=8, label='CER-OD')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "Number of Object Threshold & Detection based CovOD/CerOD", fontsize=14)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=12)

    # データ点の値を表示（上部プロット）
    for x, y1, y2 in zip(thresholds, cov_od_values, cer_od_values):
        ax1.annotate(f'{y1:.3f}', (x, y1), textcoords="offset points",
                     xytext=(0, 10), ha='center', color='blue')
        ax1.annotate(f'{y2:.3f}', (x, y2), textcoords="offset points",
                     xytext=(0, -20), ha='center', color='red')
   # 下部プロット（推論回数）
    ax2.bar(thresholds, detection_counts, color='green', alpha=0.6)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Num obj threshold", fontsize=12)
    ax2.set_ylabel("inference count", fontsize=12)

    # データ点の値を表示（下部プロット）
    for x, y in zip(thresholds, detection_counts):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points",
                     xytext=(0, 5), ha='center', color='darkgreen')

    # x軸の整数値を強制（両方のプロット）
    ax1.set_xticks(thresholds)
    ax2.set_xticks(thresholds)

    plt.tight_layout()

    # グラフの保存（オプション）
    # plt.savefig('adaptive_threshold_analysis_with_counts.png', dpi=300, bbox_inches='tight')

    plt.show()
