import classify
import utils


class Dataset:
    def __init__(self, gt_dir, det_dirs, num_version):
        self.results = [classify.classify(gt_dir, det_dir)
                        for det_dir in det_dirs]
        self.num_version = num_version
        self.num_frame = len(self.results[0])

    def common_fp(self):
        if self.num_version == 1:
            return self.results[0]['FP']

        common_fp = list()
        for frame in range(self.num_frame):
            frame_common_fp = self.results[0][frame]['FP']  # 最初のバージョンのFPを基準にする
            for version in range(1, self.num_version):
                frame_fp = self.results[version][frame]['FP']
                for class_id, boxes in frame_fp.items():
                    if class_id not in frame_common_fp:
                        continue
                    for box in boxes:
                        # 共通FP内のboxと最大のIoUを持つboxを探す
                        max_iou = 0.0
                        matched_box = None
                        for common_box in frame_common_fp[class_id]:
                            iou = utils.iou(box, common_box)
                            if iou > max_iou:
                                max_iou = iou
                                matched_box = common_box
                        if max_iou > utils.IoU_THRESHOLD:
                            frame_common_fp[class_id].remove(matched_box)

            common_fp.append(frame_common_fp)
        return common_fp

    def common_fn(self):
        num_version = len(self.results)
        num_frame = self.num_frame
        if num_version == 1:
            return self.results[0]['FN']

        common_fn = list()
        for frame in range(num_frame):
            frame_common_fn = self.results[0][frame]['FN']  # 最初のバージョンのFNを基準にする
            for version in range(1, num_version):
                frame_fn = self.results[version][frame]['FN']
                for class_id, boxes in frame_fn.items():
                    if class_id not in frame_common_fn:
                        continue
                    for box in boxes:
                        # 共通FN内のboxと最大のIoUを持つboxを探す
                        max_iou = 0.0
                        matched_box = None
                        for common_box in frame_common_fn[class_id]:
                            iou = utils.iou(box, common_box)
                            if iou > max_iou:
                                max_iou = iou
                                matched_box = common_box
                        if max_iou > utils.IoU_THRESHOLD:
                            frame_common_fn[class_id].remove(matched_box)

            common_fn.append(frame_common_fn)
        return common_fn

    def all_fp(self):
        all_fp = list()

        for frame in range(self.num_frame):
            frame_all_fp = dict()
            frame_all_fp = self.results[0][frame]['FP']
            for version in range(1, self.num_version):
                frame_fp = self.results[version][frame]['FP']
                for class_id, boxes in frame_fp.items():
                    if class_id not in frame_all_fp:
                        frame_all_fp[class_id] = list()
                    for box in boxes:
                        # 共通FP内に一定以上のIoUを持つboxがなければ追加
                        for existing_box in frame_all_fp[class_id]:
                            if utils.iou(box, existing_box) > utils.IoU_THRESHOLD:
                                break
                        else:
                            frame_all_fp[class_id].append(box)
            all_fp.append(frame_all_fp)
        return all_fp

    def total_obj(self):
        total_obj = list()
        all_fp = self.all_fp()
        for frame in range(self.num_frame):
            frame_total_obj = 0
            frame_total_obj += sum(len(boxes)
                                   for boxes in self.results[0][frame]['TP'].values())
            frame_total_obj += sum(len(boxes)
                                   for boxes in self.results[0][frame]['FN'].values())
            frame_total_obj += sum(len(boxes)
                                   for boxes in all_fp[frame].values())
            total_obj.append(frame_total_obj)
        return total_obj
