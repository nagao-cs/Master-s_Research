from enum import Enum
from typing import Optional
import sys

from src.boundingBox.boundingBox import DetectionBoundingBox


class VersionState(Enum):
    ONE = 1
    N = 2


class VersionController:
    def __init__(self, confidenceScoreThreshold: float, agreementScoreThreshold: float, maxVersion: int):
        self.state = VersionState.ONE  # 初期状態
        self.confidenceScoreThreshold = confidenceScoreThreshold
        self.agreementScoreThreshold = agreementScoreThreshold
        self.maxVersion = maxVersion
        self.uncertainty_ratio_threshold: float = 0.25

    def updateState(self, BBoxList: Optional[list[DetectionBoundingBox]]):
        """
        detections:
          - ONE状態: 1モデルの検出結果
          - N状態:   Nモデルの検出結果
        """
        if self.state == VersionState.ONE:
            if self._shouldSwitchToNversion(BBoxList):
                self.state = VersionState.N

        # elif self.state == VersionState.N:
        #     if self._shouldSwitchToOneVersion(groupedBoundingBoxList):
        #         self.state = VersionState.ONE

    def outputCurrentStateAtConsole(self):
        currentState = self.state
        status = "SINGLE" if currentState == VersionState.ONE else "MULTI"
        sys.stdout.write(f"\r[{status:6s}] Currently executing...")
        sys.stdout.flush()

    def _shouldSwitchToNversion(self, boundingBoxList: list[DetectionBoundingBox]) -> bool:
        if len(boundingBoxList) == 0:
            return False
        numUncertainDet: int = sum(
            1 for BBox in boundingBoxList if BBox.confidenceScore < self.confidenceScoreThreshold)
        uncertainDetRatio: float = numUncertainDet / len(boundingBoxList)

        if uncertainDetRatio >= self.uncertainty_ratio_threshold:
            return True
        return False

    def _shouldSwitchToOneVersion(self, groupedBoundingBoxList: list[list[DetectionBoundingBox]]) -> bool:
        agreementScore: float = self._calcAgreementScore(
            groupedBoundingBoxList)

        if agreementScore >= self.agreementScoreThreshold:
            return True
        return False

    def _calcAgreementScore(self, groupedBoundingBoxList: list[list[DetectionBoundingBox]]) -> float:
        agreementScore: float = 0.0
        numGroups: int = len(groupedBoundingBoxList)

        # groupの数が0ならすべての検出結果が一致しているので一致度は1.0になる
        if numGroups == 0:
            agreementScore = 1.0
            return agreementScore

        numAllDetectionResultsMatchedGroup: int = 0
        for groupedBoundingBox in groupedBoundingBoxList:
            if len(groupedBoundingBox) == self.maxVersion:
                numAllDetectionResultsMatchedGroup += 1

        agreementScore = numAllDetectionResultsMatchedGroup / numGroups
        return agreementScore
