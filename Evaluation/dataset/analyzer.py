from utils import utils


class DetectionAnalyzer:
    def __init__(self, iou_th: float):
        self.iou_th = iou_th

    def analyze_frame(self, gt_path: str, det_paths: list[str]) -> dict:
        frame_results = dict()

    def _classify_frame(self, gt, dets) -> dict[dict]:
        frame_results = dict()
        # 各バージョンの検出結果を分類
        for version, det in enumerate(dets):
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
