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
            cov_od += (num_fp + num_fn) / frame_obj if frame_obj > 0 else 0
        cov_od = 1 - (cov_od/self.dataset.num_frame)
        return cov_od


if __name__ == "__main__":
    # main()
    map = 'Town10HD_Opt'
    gt_dir = f'C:/CARLA_Latest/WindowsNoEditor/output/label/{map}/front'
    models = ["yolov8n", "yolov5n", "SSD"]

    det_dirs = [
        f'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{map}/labels/{model}_results/front' for model in models]
    dataset = dataset.Dataset(gt_dir, det_dirs, len(models))

    print(f"cov_od: {Evaluation(dataset).cov_od()}")
