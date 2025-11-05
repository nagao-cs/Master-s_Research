import classify
import utils


class Dataset:
    def __init__(self, gt_dir, det_dirs, num_version, debug=False):
        self.num_version = num_version
        self.results = [classify.classify(gt_dir, det_dir)
                        for det_dir in det_dirs]
        if debug:
            for version in range(num_version):
                self.results[version] = self.results[version][:10]
        self.num_frame = len(self.results[0])
        self.num_gt_list = self.num_gt()
        self.num_detection_dict = self.num_detections()
        self.common_fp_list = self.common_fp()
        self.common_fn_list = self.common_fn()
        self.all_fp_list = self.all_fp()
        self.all_fn_list = self.all_fn()
        self.total_obj_list = self.total_obj()

    def topK_detection(self, version, frame, k):
        detections = list()
        for boxes in self.results[version][frame]['TP'].values():
            for box in boxes:
                detections.append(box)
        for boxes in self.results[version][frame]['FP'].values():
            for box in boxes:
                detections.append(box)
        detections.sort(key=lambda box: box[4], reverse=True)
        return detections[:k]

    def fp_detection(self, version, frame):
        fp_detection = list()
        for boxes in self.results[version][frame]['FP'].values():
            for box in boxes:
                fp_detection.append(box)
        return fp_detection

    def fn_detection(self, version, frame):
        fn_detection = list()
        for boxes in self.results[version][frame]['FN'].values():
            for box in boxes:
                fn_detection.append(box)
        return fn_detection

    def num_detections(self):
        num_detection_dict = {version: list()
                              for version in range(self.num_version)}
        for version in range(self.num_version):
            for frame in range(self.num_frame):
                num_tp = sum(len(boxes)
                             for boxes in self.results[0][frame]['TP'].values())
                num_fp = sum(len(boxes)
                             for boxes in self.results[0][frame]['FP'].values())
                num_detection_dict[version].append(num_tp + num_fp)

        return num_detection_dict

    def num_gt(self) -> list[int]:
        num_gt = list()
        for frame in range(self.num_frame):
            frame_num_gt = 0
            frame_num_gt += sum(len(boxes)
                                for boxes in self.results[0][frame]['TP'].values())
            frame_num_gt += sum(len(boxes)
                                for boxes in self.results[0][frame]['FN'].values())
            num_gt.append(frame_num_gt)
        return num_gt

    def common_tp(self):
        common_tp = list()
        for frame in range(self.num_frame):
            frame_common_tp = self.results[0][frame]['TP']  # 最初のバージョンのTPを基準にする
            for version in range(1, self.num_version):
                frame_tp = self.results[version][frame]['TP']
                for class_id, boxes in frame_tp.items():
                    if class_id not in frame_common_tp:
                        continue
                    for box in boxes:
                        # 共通TP内のboxと最大のIoUを持つboxを探す
                        max_iou = 0.0
                        matched_box = None
                        for common_box in frame_common_tp[class_id]:
                            iou = utils.iou(box, common_box)
                            if iou > max_iou:
                                max_iou = iou
                                matched_box = common_box
                        if max_iou > utils.IoU_THRESHOLD:
                            frame_common_tp[class_id].remove(matched_box)

            common_tp.append(frame_common_tp)
        return common_tp

    def common_fp(self):
        common_fps = list()

        for frame in range(self.num_frame):
            frame_common_fp = dict()
            version_fps = [self.results[version][frame]['FP'].copy()
                           for version in range(self.num_version)]

            base_fps = version_fps[0]
            for class_id in base_fps.keys():
                frame_common_fp[class_id] = list()

                for base_box in base_fps[class_id]:
                    is_common = True
                    for subject_fps in version_fps[1:]:
                        if class_id not in subject_fps:
                            is_common = False
                        matched_box = None
                        highest_iou = 0.0
                        for subject_box in subject_fps[class_id]:
                            iou = utils.iou(base_box, subject_box)
                            if iou > highest_iou and iou > utils.IoU_THRESHOLD:
                                highest_iou = iou
                                matched_box = subject_box
                        if matched_box is not None:
                            subject_fps[class_id].remove(matched_box)
                        else:
                            is_common = False
                    if is_common:
                        frame_common_fp[class_id].append(base_box)
            common_fps.append(frame_common_fp)
        return common_fps

    def common_fn(self):
        common_fns = list()

        for frame in range(self.num_frame):
            frame_common_fn = dict()

            base_fns = self.results[0][frame]['FN']
            for class_id in base_fns.keys():
                frame_common_fn[class_id] = list()
                for base_box in base_fns[class_id]:
                    is_common = True
                    for version in range(1, self.num_version):
                        other_fns = self.results[version][frame]['FN']
                        if class_id not in other_fns:
                            is_common = False
                            break
                        subject_boxes = other_fns[class_id]
                        if base_box not in subject_boxes:
                            is_common = False
                            break
                    if is_common:
                        frame_common_fn[class_id].append(base_box)
            common_fns.append(frame_common_fn)
        return common_fns

    def all_fp(self):
        all_fps = list()

        for frame in range(self.num_frame):
            frame_all_fp = dict()  # 各フレームの全FP
            version_fps = [self.results[version][frame]['FP'].copy()
                           for version in range(self.num_version)]
            for version in range(self.num_version):
                base_fps = version_fps[version]
                for class_id in base_fps.keys():
                    if class_id not in frame_all_fp:
                        frame_all_fp[class_id] = []
                    for base_box in base_fps[class_id]:
                        for subject_fps in version_fps[version+1:]:
                            matched_box = None
                            highest_iou = 0.0
                            for subject_box in subject_fps.get(class_id, []):
                                iou = utils.iou(base_box, subject_box)
                                if iou > highest_iou and iou > utils.IoU_THRESHOLD:
                                    highest_iou = iou
                                    matched_box = subject_box
                            if matched_box is not None:
                                subject_fps[class_id].remove(matched_box)
                        frame_all_fp[class_id].append(base_box)
            all_fps.append(frame_all_fp)
        return all_fps

    def all_fn(self):
        all_fns = list()

        for frame in range(self.num_frame):
            frame_all_fn = dict()  # 各フレームの全FN

            for version in range(self.num_version):
                version_fn = self.results[version][frame]['FN']
                for class_id, boxes in version_fn.items():
                    if class_id not in frame_all_fn:
                        frame_all_fn[class_id] = list()
                    for subject_box in boxes:
                        if subject_box not in frame_all_fn[class_id]:
                            frame_all_fn[class_id].append(subject_box)
            all_fns.append(frame_all_fn)
        return all_fns

    def total_obj(self):
        total_obj = list()
        all_fp = self.all_fp()
        for frame in range(self.num_frame):
            frame_total_obj = 0
            frame_total_obj += self.num_gt_list[frame]
            frame_total_obj += sum(len(boxes)
                                   for boxes in all_fp[frame].values())
            total_obj.append(frame_total_obj)
        return total_obj
