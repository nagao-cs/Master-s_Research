from enum import Enum


class VersionState(Enum):
    ONE = 1
    N = 2


class VersionController:
    def __init__(self, conf_threshold: float, agreement_threshold: float):
        self.state = VersionState.ONE  # 初期状態
        self.conf_threshold = conf_threshold
        self.agreement_threshold = agreement_threshold

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

    def _should_switch_to_N(self, base_bboxes):
        min_conf = min([b.conf for b in base_bboxes], default=1.0)
        return min_conf < self.threshold

    def _should_switch_to_ONE(self, all_detections):
        agreement = self._calc_agreement(all_detections)
        return agreement > self.threshold

    def _calc_agreement(self, all_detections):
        # IoU一致率や投票一致率
        pass
