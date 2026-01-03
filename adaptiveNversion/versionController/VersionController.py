from enum import Enum
from typing import Optional
import sys

from boundingBox.boundingBox import DetectionBoundingBox


class VersionState(Enum):
    ONE = 1
    N = 2


class VersionController:
    def __init__(self, confidenceScoreThreshold: float, agreementScoreThreshold: float, maxVersion: int):
        self.state = VersionState.ONE  # 初期状態
        self.confidenceScoreThreshold = confidenceScoreThreshold
        self.agreementScoreThreshold = agreementScoreThreshold
        self.maxVersion = maxVersion

    def updateState(self, detections: Optional[list[DetectionBoundingBox]] = None, groupedBoundingBoxList: Optional[list[list[DetectionBoundingBox]]] = None):
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

    def outputCurrentStateAtConsole(self):
        currentState = self.state
        status = "SINGLE" if currentState == VersionState.ONE else "MULTI"
        sys.stdout.write(f"\r[{status:6s}] Currently executing...")
        sys.stdout.flush()

    def _shouldSwitchToNversion(self, boundingBoxList: list[DetectionBoundingBox]) -> bool:
        minConfidenceScore: float = 1.0
        for boundingBox in boundingBoxList:
            confidenceScore = boundingBox.confidenceScore
            minConfidenceScore = min(minConfidenceScore, confidenceScore)
        return minConfidenceScore < self.confidenceScoreThreshold

    def _shouldSwitchToOneVersion(self, groupedBoundingBoxList: list[list[DetectionBoundingBox]]) -> bool:
        agreementScore = self._calcAgreementScore(groupedBoundingBoxList)
        return agreementScore > self.agreementScoreThreshold

    def _calcAgreementScore(self, groupedBoundingBoxList: list[list[DetectionBoundingBox]]) -> float:
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
