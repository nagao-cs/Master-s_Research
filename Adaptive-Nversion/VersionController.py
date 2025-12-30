from enum import Enum
from typing import Dict, List, Tuple


class VersionState(Enum):
    ONE = 1
    N = 2


class VersionController:
    def __init__(self, conf_threshold: float, agreement_threshold: float, maxVersion: int):
        self.state = VersionState.ONE  # 初期状態
        self.conf_threshold = conf_threshold
        self.agreement_threshold = agreement_threshold
        self.maxVersion = maxVersion

    def update_state(self, detections: list = None, detection_dict: dict[object, list] = None):
        """
        detections:
          - ONE状態: 1モデルの検出結果
          - N状態:   Nモデルの検出結果
        """
        if self.state == VersionState.ONE:
            if self._should_switch_to_N(detections):
                self.state = VersionState.N

        elif self.state == VersionState.N:
            if self._should_switch_to_ONE(detection_dict):
                self.state = VersionState.ONE

    def _should_switch_to_N(self, boundingbox_list: list):
        min_conf = 1.0
        for boundingbox in boundingbox_list:
            conf = boundingbox['confidence']
            min_conf = min(min_conf, conf)
        return min_conf < self.conf_threshold

    def _should_switch_to_ONE(self, detection_model_dict):
        agreement = self._calc_agreement(detection_model_dict)
        return agreement > self.agreement_threshold

    def _calc_agreement(self, detection_model_dict):
        agreementScore = 0.0

        if detection_model_dict == None:
            agreementScore = 1.0
            return agreementScore

        groupClassDict = self._groupingDetections(detection_model_dict)
        numAllMatchedGroup = 0
        numGroups = 0
        for groups in groupClassDict.items():
            for group in groups:
                numGroups += 1
                if len(group) == self.maxVersion:
                    numAllMatchedGroup += 1

        agreementScore = numAllMatchedGroup / numGroups
        return agreementScore

    def _groupingDetections(self, detections: Dict[int, Dict[int, List[Tuple]]]) -> List:
        # Find all unique class IDs present across all versions
        all_class_ids = set()
        for v_dets in detections.values():
            all_class_ids.update(v_dets.keys())

        groupsClassDict = dict()

        for class_id in all_class_ids:
            # 1. Flatten all detections for the current class
            all_dets = []
            for version_id, v_dets in detections.items():
                boxes = v_dets.get(class_id, [])
                for b in boxes:
                    # Input assumed to be: (x_center, y_center, width, height, confidence)
                    x, y, w, h, conf = b
                    all_dets.append({
                        'version_id': version_id,
                        'x_center': float(x),
                        'y_center': float(y),
                        'width': float(w),
                        'height': float(h),
                        'confidence': float(conf)
                    })

            if not all_dets:
                continue

            is_processed = [False] * len(all_dets)
            groups = list()

            # 2. Grouping (Clustering based on IoU)
            for i in range(len(all_dets)):
                if is_processed[i]:
                    continue

                base = all_dets[i]
                group = [base]
                is_processed[i] = True

                # Iteratively add other unprocessed detections to the group
                for j in range(i + 1, len(all_dets)):
                    if is_processed[j]:
                        continue

                    cand = all_dets[j]

                    if cand['version_id'] == base['version_id']:
                        continue

                    matched = False
                    for member in group:
                        # Check IoU against any existing member in the group
                        if self._iou(member, cand) >= self.iou_th:
                            matched = True
                            break

                    if matched:
                        group.append(cand)
                        is_processed[j] = True

                groups.append(group)
            groupsClassDict[class_id] = groups

        return groupsClassDict
