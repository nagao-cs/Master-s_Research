from utils import utils


class DetectionAnalyzer:
    def __init__(self, iou_th: float):
        self.iou_th = iou_th

    def analyze_frame(self, gt: dict, dets: dict[dict]) -> dict:
        classified_results = self._classify_frame(
            gt, dets)  # これでdetsのバージョン数が切り替わっても使える
        intersection_errors = self._intersection_of_errors(classified_results)
        union_errors = self._union_of_errors(classified_results)
        return {
            'classified_results': classified_results,
            'intersection_errors': intersection_errors,
            'union_errors': union_errors
        }

    def _intersection_of_errors(self, classified_dets) -> dict:
        intersection_errors = {'FP': dict(), 'FN': dict()}
        for error_type in ['FP', 'FN']:
            for version, classified_det in classified_dets.items():
                if version == 0:
                    base_errors = classified_det[error_type]
                    for class_id in base_errors.keys():
                        intersection_errors[error_type][class_id] = list()
                        for boxes in base_errors[class_id]:
                            intersection_errors[error_type][class_id].append(
                                boxes)
                else:
                    current_errors = classified_det[error_type]
                    used_boxes = set()
                    for class_id in intersection_errors[error_type].keys():
                        if class_id not in current_errors:
                            del intersection_errors[error_type][class_id]
                            continue
                        for base_box in intersection_errors[error_type][class_id]:
                            best_iou = 0.0
                            for curr_box in current_errors[class_id]:
                                if curr_box in used_boxes:
                                    continue
                                print("base_box", base_box)
                                print("curr_box", curr_box)
                                iou = utils.iou(base_box, curr_box)
                                if iou > best_iou:
                                    best_iou = iou
                            if best_iou >= self.iou_th:
                                used_boxes.add(curr_box)
                            else:
                                intersection_errors[error_type][class_id].remove(
                                    base_box)
            return intersection_errors

    def _union_of_errors(self, dets: dict[dict]) -> dict:
        union_errors = {'FP': dict(), 'FN': dict()}
        for error_type in ['FP', 'FN']:
            for version, classified_det in dets.items():
                current_errors = classified_det[error_type]
                for class_id in current_errors.keys():
                    if class_id not in union_errors[error_type]:
                        union_errors[error_type][class_id] = list()
                    for box in current_errors[class_id]:
                        # すでに追加されているか確認
                        already_added = False
                        for existing_box in union_errors[error_type][class_id]:
                            iou = utils.iou(box, existing_box)
                            if iou >= self.iou_th:
                                already_added = True
                                break
                        if not already_added:
                            union_errors[error_type][class_id].append(box)
        return union_errors

    def _classify_frame(self, gt, dets) -> dict[dict]:
        frame_results = dict()
        # 各バージョンの検出結果を分類
        for version, det in dets.items():
            frame_results[version] = self._classify(gt, det)
        return frame_results

    def _classify(self, gt, det) -> dict:
        det_results = {'TP': dict(), 'FP': dict(), 'FN': dict()}
        # クラスごとに処理
        for class_id in gt.keys():
            gt_boxes = gt[class_id]
            det_boxes = det.get(class_id, [])

            if class_id not in det_results['TP']:
                det_results['TP'][class_id] = list()
            if class_id not in det_results['FN']:
                det_results['FN'][class_id] = list()
            if class_id not in det_results['FP']:
                det_results['FP'][class_id] = list()

            # GTとDetectionのマッチング
            used_gt = set()  # マッチ済みのGTを記録

            for det_box in det_boxes:
                best_iou = 0.0
                best_gt_idx = -1

                # 未使用のGTと最もIoUが高いものを探す
                for i, gt_box in enumerate(gt_boxes):
                    if i in used_gt:
                        continue
                    iou = utils.iou(gt_box, det_box)
                    if iou >= utils.IoU_THRESHOLD and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                # マッチング結果に基づいて分類
                if best_gt_idx >= 0:
                    det_results['TP'][class_id].append(det_box)
                    used_gt.add(best_gt_idx)
                else:
                    det_results['FP'][class_id].append(det_box)

            # 未使用のGTをFNとして追加
            for i, gt_box in enumerate(gt_boxes):
                if i not in used_gt:
                    det_results['FN'][class_id].append(gt_box)
        return det_results
