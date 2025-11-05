from typing import List, Dict
import numpy as np
from .detection import Detection


class DetectionIntegrator:
    def __init__(self, num_models: int, iou_threshold: float = 0.5, confidence_threshold: float = 0.3):
        self.num_models = num_models
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def integrate_frame(self, detections: List[Detection]) -> List[Detection]:
        """フレームごとの検出結果を統合"""
        # 信頼度でフィルタリング
        # filtered_dets = [
        #     d for d in detections if d.confidence >= self.confidence_threshold]
        # if not filtered_dets:
        #     return []

        # クラスごとにグループ化
        class_groups: Dict[int, List[Detection]] = dict()
        for det in detections:
            if det.class_id not in class_groups:
                class_groups[det.class_id] = list()
            class_groups[det.class_id].append(det)

        integrated_detections = list()
        # クラスごとに統合
        for class_id, class_dets in class_groups.items():
            clusters = self._cluster_by_iou(class_dets)
            for cluster in clusters:
                if len(cluster) >= (self.num_models / 2):  # 多数決
                    merged_det = self._merge_cluster(cluster)
                    integrated_detections.append(merged_det)

        return integrated_detections

    def _cluster_by_iou(self, detections: List[Detection]) -> List[List[Detection]]:
        """IoUベースで検出をクラスタリング"""
        clusters = list()
        used = set()

        for i, det in enumerate(detections):
            if i in used:
                continue

            current_cluster = [det]
            used.add(i)

            for j, other_det in enumerate(detections):
                if j in used:
                    continue
                if self._calculate_iou(det.bbox, other_det.bbox) >= self.iou_threshold:
                    current_cluster.append(other_det)
                    used.add(j)

            clusters.append(current_cluster)

        return clusters

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """IoUを計算"""
        ax_center, ay_center, aw, ah = box1
        axmin, aymin = ax_center - aw / 2, ay_center - ah / 2
        axmax, aymax = ax_center + aw / 2, ay_center + ah / 2

        bx_center, by_center, bw, bh = box2
        bxmin, bymin = bx_center - bw / 2, by_center - bh / 2
        bxmax, bymax = bx_center + bw / 2, by_center + bh / 2

        inter_xmin = max(axmin, bxmin)
        inter_ymin = max(aymin, bymin)
        inter_xmax = min(axmax, bxmax)
        inter_ymax = min(aymax, bymax)

        inter_area = max(0, inter_xmax - inter_xmin) * \
            max(0, inter_ymax - inter_ymin)
        area_a = (axmax - axmin) * (aymax - aymin)
        area_b = (bxmax - bxmin) * (bymax - bymin)
        union_area = area_a + area_b - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _merge_cluster(cluster: List[Detection]) -> Detection:
        """クラスタ内の検出を1つに統合"""
        # 重みつき平均でbboxを計算
        weights = [det.confidence for det in cluster]
        weighted_boxes = np.average([det.bbox for det in cluster],
                                    weights=weights, axis=0)

        # 最大の信頼度を採用
        max_conf = max(det.confidence for det in cluster)

        return Detection(
            bbox=weighted_boxes,
            confidence=max_conf,
            class_id=cluster[0].class_id,  # 全て同じクラス
            frame_id=cluster[0].frame_id,
            model_id=-1  # 統合結果を示す特別な値
        )
