from enum import Enum
from typing import Dict, List, Tuple, Optional
from ObjectDetection.boundingbox.boundingBox import BoundingBox


class VersionState(Enum):
    ONE = 1
    N = 2


class VersionController:
    def __init__(self, confidenceScoreThreshold: float, agreementScoreThreshold: float, maxVersion: int):
        self.state = VersionState.ONE  # 初期状態
        self.confidenceScoreThreshold = confidenceScoreThreshold
        self.agreementScoreThreshold = agreementScoreThreshold
        self.maxVersion = maxVersion

    def updateState(self, detections: Optional[list[BoundingBox]] = None, groupedBoundingBoxList: Optional[list[list[BoundingBox]]] = None):
        """
        detections:
          - ONE状態: 1モデルの検出結果
          - N状態:   Nモデルの検出結果
        """
        if self.state == VersionState.ONE:
            if self._shouldSwitchToNversion(detections):
                self.state = VersionState.N

        elif self.state == VersionState.N:
            if self._shouldSwitchToOneVersion(groupedBoundingBoxList):
                self.state = VersionState.ONE

    def _shouldSwitchToNversion(self, boundingBoxList: list[BoundingBox]) -> bool:
        minConfidenceScore: float = 1.0
        for boundingBox in boundingBoxList:
            confidenceScore = boundingBox.confidenceScore
            minConfidenceScore = min(minConfidenceScore, confidenceScore)
        return minConfidenceScore < self.confidenceScoreThreshold

    def _shouldSwitchToOneVersion(self, groupedBoundingBoxList: list[list[BoundingBox]]) -> bool:
        agreementScore = self._calcAgreementScore(groupedBoundingBoxList)
        return agreementScore > self.agreementScoreThreshold

    def _calcAgreementScore(self, groupedBoundingBoxList: list[list[BoundingBox]]) -> float:
        agreementScore = 0.0

        if groupedBoundingBoxList == None:
            agreementScore = 1.0
            return agreementScore

        numAllMatchedGroup = 0
        numGroups = 0
        for groupedBoundingBox in groupedBoundingBoxList:
            numGroups += 1
            if len(groupedBoundingBox) == self.maxVersion:
                numAllMatchedGroup += 1

        agreementScore = numAllMatchedGroup / numGroups
        return agreementScore
