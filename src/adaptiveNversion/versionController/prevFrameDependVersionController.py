from enum import Enum
from typing import Optional
import sys

from src.boundingBox.boundingBox import DetectionBoundingBox


class VersionState(Enum):
    ONE = 1
    N = 2


class PrevFrameDependVersionController:
    def __init__(self, numDetDiffThreshold: float, maxVersion: int):
        self.state = VersionState.ONE  # 初期状態
        self.numDetDiffThreshold: float = numDetDiffThreshold
        self.prevFrameNumDet: int = 0
        self.maxVersion = maxVersion

    def updateState(self, BBoxList: Optional[list[DetectionBoundingBox]] = None, BBoxGroupList: list[list[DetectionBoundingBox]] = None):
        """
        BBoxListがNoneならNバージョン
        BBoxGroupListがNoneなら1バージョン
        """
        if self.state == VersionState.ONE:
            if self._shouldSwitchToNversion(BBoxList):
                self.state = VersionState.N
            self.prevFrameNumDet: int = len(BBoxList)

    def _shouldSwitchToNversion(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        currentNumDet: int = len(BBoxList)
        if abs(self.prevFrameNumDet - currentNumDet) >= self.numDetDiffThreshold:
            return True
        return False

    def _shouldSwitchToOneVersion(self, groupedBoundingBoxList: list[list[DetectionBoundingBox]]) -> bool:
        agreementScore: float = self._calcAgreementScore(
            groupedBoundingBoxList)

        if agreementScore > self.agreementScoreThreshold:
            return True
        elif agreementScore <= self.agreementScoreThreshold:
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
