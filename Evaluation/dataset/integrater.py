import numpy as np
from math import ceil


class DetectionIntegrator:
    def __init__(self, iou_th: float, num_version: int):
        self.iou_th = iou_th
        self.num_version = num_version
        self.majority_threshold = ceil(num_version / 2)

    def integrate_detections(self, dets: dict, mode: str) -> dict:
        """
        各クラスごとに、各サイクルで検出されたboxを全てプールし、
        最後に過半数以上のバージョンで登場したboxのみ残す。
        """
        if self.num_version == 1 or mode == "1version":
            return {0: dets[0]}
        # 全クラスIDを取得
        all_classes = set()
        for v in dets.values():
            all_classes |= set(v.keys())

        # 各クラスで統合処理
        accumulated_boxes = {cls: [] for cls in all_classes}

        base_det = dets[0]
        for version in range(1, self.num_version):
            subj_det = dets[version]

            for class_id in all_classes:
                boxes1 = np.array(base_det.get(class_id, []))
                boxes2 = np.array(subj_det.get(class_id, []))

                if len(boxes1) == 0 and len(boxes2) == 0:
                    continue
                elif len(boxes1) == 0:
                    accumulated_boxes[class_id].extend(boxes2.tolist())
                    continue
                elif len(boxes2) == 0:
                    accumulated_boxes[class_id].extend(boxes1.tolist())
                    continue

                # IoU行列でマッチング
                sim_matrix = self._iou_matrix(boxes1, boxes2)
                matched_pairs, unmatched_rows, unmatched_cols = self._greedy_match(
                    sim_matrix)

                # マッチしたペアは両方追加（平均しない）
                for r, c, _ in matched_pairs:
                    accumulated_boxes[class_id].append(boxes1[r].tolist())
                    accumulated_boxes[class_id].append(boxes2[c].tolist())

                # 未マッチもそのまま追加
                for idx in unmatched_rows:
                    accumulated_boxes[class_id].append(boxes1[idx].tolist())
                for idx in unmatched_cols:
                    accumulated_boxes[class_id].append(boxes2[idx].tolist())

            # 次の統合へ
            base_det = subj_det

        # --- 最終フィルタリング ---
        filtered_result = self._filter_majority(accumulated_boxes)
        return {0: filtered_result}

    def _iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        x11 = boxes1[:, 0] - boxes1[:, 2] / 2
        y11 = boxes1[:, 1] - boxes1[:, 3] / 2
        x12 = boxes1[:, 0] + boxes1[:, 2] / 2
        y12 = boxes1[:, 1] + boxes1[:, 3] / 2

        x21 = boxes2[:, 0] - boxes2[:, 2] / 2
        y21 = boxes2[:, 1] - boxes2[:, 3] / 2
        x22 = boxes2[:, 0] + boxes2[:, 2] / 2
        y22 = boxes2[:, 1] + boxes2[:, 3] / 2

        xi1 = np.maximum(x11[:, None], x21[None, :])
        yi1 = np.maximum(y11[:, None], y21[None, :])
        xi2 = np.minimum(x12[:, None], x22[None, :])
        yi2 = np.minimum(y12[:, None], y22[None, :])

        inter_w = np.maximum(0, xi2 - xi1)
        inter_h = np.maximum(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union = area1[:, None] + area2[None, :] - inter_area
        return inter_area / np.clip(union, 1e-8, None)

    def _greedy_match(self, sim_matrix: np.ndarray):
        N, M = sim_matrix.shape
        flat_idx = np.argsort(sim_matrix.ravel())[::-1]
        rows, cols = np.unravel_index(flat_idx, sim_matrix.shape)
        scores = sim_matrix[rows, cols]

        valid = scores >= self.iou_th
        rows, cols, scores = rows[valid], cols[valid], scores[valid]

        used_rows = np.zeros(N, bool)
        used_cols = np.zeros(M, bool)

        matched_pairs = []
        for r, c, s in zip(rows, cols, scores):
            if used_rows[r] or used_cols[c]:
                continue
            matched_pairs.append((r, c, s))
            used_rows[r] = True
            used_cols[c] = True

        unmatched_rows = np.where(~used_rows)[0].tolist()
        unmatched_cols = np.where(~used_cols)[0].tolist()

        return matched_pairs, unmatched_rows, unmatched_cols

    def _filter_majority(self, boxes_dict: dict) -> dict:
        """
        IoUによってクラスタリングし、
        各クラスタの出現回数が過半数以上なら平均化して残す。
        """
        result = {}
        for class_id, boxes in boxes_dict.items():
            if len(boxes) == 0:
                result[class_id] = []
                continue
            boxes = np.array(boxes)
            used = np.zeros(len(boxes), bool)
            kept = []
            for i in range(len(boxes)):
                if used[i]:
                    continue
                ref = boxes[i]
                ious = self._iou_vector(ref, boxes)
                cluster_idx = np.where(ious >= self.iou_th)[0]
                if len(cluster_idx) >= self.majority_threshold:
                    cluster_boxes = boxes[cluster_idx]
                    xywh = cluster_boxes[:, :4].mean(axis=0)
                    conf = cluster_boxes[:, 4].mean()
                    kept.append(np.concatenate([xywh, [conf]]).tolist())
                used[cluster_idx] = True
            result[class_id] = kept
        return result

    def _iou_vector(self, box, boxes):
        x1, y1, w1, h1 = box[:4]
        x2, y2, w2, h2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x11, y11, x12, y12 = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
        x21, y21, x22, y22 = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2
        xi1 = np.maximum(x11, x21)
        yi1 = np.maximum(y11, y21)
        xi2 = np.minimum(x12, x22)
        yi2 = np.minimum(y12, y22)
        inter_w = np.maximum(0, xi2 - xi1)
        inter_h = np.maximum(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter_area
        return inter_area / np.clip(union, 1e-8, None)
