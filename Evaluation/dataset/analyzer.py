from utils import utils


class DetectionAnalyzer:
    def __init__(self, iou_th: float):
        self.iou_th = iou_th

    def analyze_frame(self, gt: dict, dets: dict[dict]) -> dict:
        classified_results = self._classify_frame(
            gt, dets)  # これでdetsのバージョン数が切り替わっても使える
        intersection_errors = self._intersection_of_errors(classified_results)
        union_errors = self._union_of_errors(classified_results)
        total_instances = self._total_instances(classified_results)
        return {
            'classified_results': classified_results,
            'intersection_errors': intersection_errors,
            'union_errors': union_errors,
            'total_instances': total_instances
        }

    def _intersection_of_errors(self, classified_dets) -> dict:
        intersection_errors = {'FP': dict(), 'FN': dict()}
        for error_type in ['FP', 'FN']:
            for version, classified_det in classified_dets.items():
                current_errors = classified_det[error_type]

                # 最初のバージョンの場合 → 初期化
                if version == 0:
                    for class_id, boxes in current_errors.items():
                        intersection_errors[error_type][class_id] = boxes.copy(
                        )
                    continue

                # 2バージョン目以降 → 共通部分を更新
                new_intersection = dict()
                for class_id, base_boxes in intersection_errors[error_type].items():
                    if class_id not in current_errors:
                        continue  # 現在のversionに存在しないclassは共通でない
                    matched_boxes = []
                    used = set()
                    for base_box in base_boxes:
                        best_iou = 0.0
                        best_box = None
                        for curr_box in current_errors[class_id]:
                            if curr_box in used:
                                continue
                            iou = utils.iou(base_box, curr_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_box = curr_box
                        if best_iou >= self.iou_th:
                            matched_boxes.append(base_box)
                            used.add(best_box)
                    if matched_boxes:
                        new_intersection[class_id] = matched_boxes
                intersection_errors[error_type] = new_intersection
        return intersection_errors

    def _union_of_errors(self, classified_dets) -> dict:
        union_errors = {'FP': dict(), 'FN': dict()}
        for error_type in ['FP', 'FN']:
            for version, classified_det in classified_dets.items():
                current_errors = classified_det[error_type]
                # 最初のバージョンの場合 → 初期化
                if version == 0:
                    for class_id, boxes in current_errors.items():
                        union_errors[error_type][class_id] = boxes.copy(
                        )
                    continue
                for class_id, boxes in current_errors.items():
                    if class_id not in union_errors[error_type]:
                        union_errors[error_type][class_id] = []
                    for box in boxes:
                        already_added = False
                        for existing_box in union_errors[error_type][class_id]:
                            if utils.iou(box, existing_box) >= self.iou_th:
                                already_added = True
                                break
                        if not already_added:
                            union_errors[error_type][class_id].append(box)
        return union_errors

    def _total_instances(self, dets: dict[dict]) -> dict:
        total_instances = {"TP": dict(), "FP": dict(), "FN": dict()}
        union_errors = self._union_of_errors(dets)
        for error_type in ['FP', 'FN']:
            for class_id in union_errors[error_type].keys():
                total_instances[error_type][class_id] = union_errors[error_type][class_id]

        for version, classified_det in dets.items():
            tp_results = classified_det['TP']
            for class_id in tp_results.keys():
                if class_id not in total_instances['TP']:
                    total_instances['TP'][class_id] = list()
                for box in tp_results[class_id]:
                    # すでに追加されているか確認
                    already_added = False
                    for existing_box in total_instances['TP'][class_id]:
                        iou = utils.iou(box, existing_box)
                        if iou >= self.iou_th:
                            already_added = True
                            break
                    if not already_added:
                        total_instances['TP'][class_id].append(box)
        return total_instances

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
