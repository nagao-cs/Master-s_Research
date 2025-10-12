import os
import pickle
import utils
import classify
import dataset


class Evaluation:
    def __init__(self, dataset):
        self.dataset = dataset

    def cov_od(self):
        total_objs = self.dataset.total_obj()
        common_fps = self.dataset.common_fp()
        common_fns = self.dataset.common_fn()
        cov_od = 0.0
        for frame in range(self.dataset.num_frame):
            frame_obj = total_objs[frame]
            frame_common_fp = common_fps[frame]
            frame_common_fn = common_fns[frame]
            num_fp = sum(len(bboxes) for bboxes in frame_common_fp.values())
            num_fn = sum(len(bboxes) for bboxes in frame_common_fn.values())
            cov_od += ((num_fp + num_fn) /
                       frame_obj) if frame_obj > 0 else 0

            if frame_obj == 0 and (num_fp > 0 or num_fn > 0):
                print(
                    f"frame: {frame}, obj: {frame_obj}, fp: {num_fp}, fn: {num_fn}")
            # print(
            #     f"frame: {frame}, obj: {frame_obj}, fp: {num_fp}, fn: {num_fn}, cov_od: {(num_fp + num_fn) / frame_obj if frame_obj > 0 else 0}")
        cov_od = 1 - (cov_od/self.dataset.num_frame)
        return cov_od

    def cer_od(self):
        total_objs = self.dataset.total_obj()
        all_fps = self.dataset.all_fp()
        all_fns = self.dataset.all_fn()
        cer_od = 0.0
        for frame in range(self.dataset.num_frame):
            frame_obj = total_objs[frame]
            frame_all_fp = all_fps[frame]
            frame_all_fn = all_fns[frame]
            num_fp = sum(len(bboxes) for bboxes in frame_all_fp.values())
            num_fn = sum(len(bboxes) for bboxes in frame_all_fn.values())
            cer_od += (num_fp + num_fn) / frame_obj if frame_obj > 0 else 0
            if frame_obj == 0 and (num_fp > 0 or num_fn > 0):
                print(
                    f"frame: {frame}, obj: {frame_obj}, fp: {num_fp}, fn: {num_fn}")
            # print(
            #     f"frame: {frame}, obj: {frame_obj}, fp: {num_fp}, fn: {num_fn}, cer_od: {(num_fp + num_fn) / frame_obj if frame_obj > 0 else 0}")
        cer_od = 1 - (cer_od/self.dataset.num_frame)
        return cer_od

    def adaptive_cov_od(self):
        total_objs = self.dataset.total_obj_list
        common_fps = self.dataset.common_fp_list
        common_fns = self.dataset.common_fn_list
        adaptive_cov_od = 0.0
        adaptive_conut = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]

            if num_gt >= utils.ADAPTIVE_THRESHOLD:
                frame_obj = total_objs[frame]
                frame_common_fp = common_fps[frame]
                frame_common_fn = common_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_common_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_common_fn.values())
                adaptive_cov_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                if frame_obj == 0 and (num_fp > 0 or num_fn > 0):
                    print(
                        f"frame: {frame}, obj: {frame_obj}, fp: {num_fp}, fn: {num_fn}")
            else:
                num_fp = sum(len(bboxes)
                             for bboxes in common_fps[frame].values())
                num_fn = sum(len(bboxes)
                             for bboxes in common_fns[frame].values())
                adaptive_cov_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                adaptive_conut += 1
        adaptive_cov_od = 1 - (adaptive_cov_od/self.dataset.num_frame)
        # print(f"adaptive_conut: {adaptive_conut}")
        return adaptive_cov_od

    def adaptive_cer_od(self):
        total_objs = self.dataset.total_obj()
        all_fps = self.dataset.all_fp()
        all_fns = self.dataset.all_fn()
        adaptive_cer_od = 0.0
        adaptive_conut = 0
        for frame in range(self.dataset.num_frame):
            num_gt = self.dataset.num_gt_list[frame]

            if num_gt >= utils.ADAPTIVE_THRESHOLD:
                frame_obj = total_objs[frame]
                frame_all_fp = all_fps[frame]
                frame_all_fn = all_fns[frame]
                num_fp = sum(len(bboxes)
                             for bboxes in frame_all_fp.values())
                num_fn = sum(len(bboxes)
                             for bboxes in frame_all_fn.values())
                adaptive_cer_od += (num_fp + num_fn) / \
                    frame_obj if frame_obj > 0 else 0
                if frame_obj == 0 and (num_fp > 0 or num_fn > 0):
                    print(
                        f"frame: {frame}, obj: {frame_obj}, fp: {num_fp}, fn: {num_fn}")
            else:
                num_fp = sum(len(bboxes)
                             for bboxes in all_fps[frame].values())
                num_fn = sum(len(bboxes)
                             for bboxes in all_fns[frame].values())
                adaptive_cer_od += (num_fp + num_fn) / \
                    (num_gt + num_fp) if (num_gt + num_fp) > 0 else 0
                adaptive_conut += 1
        adaptive_cer_od = 1 - (adaptive_cer_od/self.dataset.num_frame)
        # print(f"adaptive_conut: {adaptive_conut}")
        return adaptive_cer_od

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


if __name__ == "__main__":
    # main()
    import itertools
    import sys
    debug = True if len(sys.argv) > 1 and (sys.argv[1] == 'debug') else False
    maps = [
        "Town02",
        # 'Town01_Opt',
        # 'Town05_Opt',
        # 'Town10HD_Opt'
    ]
    models = [
        "yolov8n",
        # "yolov5n",
        # "yolov11n"
    ]

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
        default="Town01",
        choices=["Town01", "Town02", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, Town04, Town05, Town10HD"
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
    print(f"map: {map}")
    gt_dir = f'C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front'
    version = len(models)
    print(f"models: {models}")
    det_dirs = [
        f'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{model}/front' for model in models]
    ds = dataset.Dataset(gt_dir, det_dirs, version, debug)
    # print(*ds.results[0][:3], sep='\n')
    # print("common_fp", *ds.common_fp()[:3], sep='\n')
    # print("common_fn", *ds.common_fn()[:3], sep='\n')
    # print("all_fp", * ds.all_fp()[:3], sep='\n')
    # print("all_fn", * ds.all_fn()[:3], sep='\n')
    print(f"    cov_od: {Evaluation(ds).cov_od()}")
    print(f"    adaptive_cov_od: {Evaluation(ds).adaptive_cov_od()}")
    print()
    print(f"    cer_od: {Evaluation(ds).cer_od()}")
    print(f"    adaptive_cer_od: {Evaluation(ds).adaptive_cer_od()} ")
    print()
    # # print(
    # # f"            avg_accuracy: {Evaluation(ds).avg_accuracy()}")
