from enum import Enum
from typing import Optional
import sys

from src.boundingBox.boundingBox import DetectionBoundingBox


class UseCaseVersionState(Enum):
    ONE = 1
    COV_STATE = 2
    CER_STATE = 3
    N = 4


class UseCaseVersionController:
    def __init__(self, confidenceScoreThreshold: float, numObjThreshold: int, maxVersion: int):
        self.state = UseCaseVersionState.ONE  # 初期状態
        self.confidenceScoreThreshold: float = confidenceScoreThreshold
        self.numObjThreshold: int = numObjThreshold
        self.maxVersion: int = maxVersion

    def updateState(self, BBoxList: list[DetectionBoundingBox]):
        # -----------
        # 1バージョンの時
        # -----------
        if self.state == UseCaseVersionState.ONE:
            # -----------
            # Cov(多く検出する)状態にするか
            # -----------
            if self._shouldSwitchToCov(BBoxList):
                self.state = UseCaseVersionState.COV_STATE
            # -----------
            # Cer(確実な検出)状態にするか
            # -----------
            elif self._shouldSwitchToCer(BBoxList):
                self.state = UseCaseVersionState.CER_STATE
        # ----------
        # Covの時
        # ----------
        elif self.state == UseCaseVersionState.COV_STATE:
            if self._shouldSwitchCovToOneVersion(BBoxList):
                self.state = UseCaseVersionState.ONE
        # ----------
        # Cerの時
        # ----------
        elif self.state == UseCaseVersionState.CER_STATE:
            if self._shouldSwitchToCov(BBoxList):
                self.state = UseCaseVersionState.COV_STATE
            elif self._shouldSwitchToOneVersion(BBoxList):
                self.state = UseCaseVersionState.ONE

    def _hasTrafficLight(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        for BBox in BBoxList:
            if BBox.classId == 9:
                return True

    def _shouldSwitchToCov(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        if self._hasTrafficLight(BBoxList):
            return True

        return False

    def _hasLowConfDet(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        for BBox in BBoxList:
            if BBox.confidenceScore < self.confidenceScoreThreshold:
                return True

        return False

    def _shouldSwitchToCer(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        # ----------
        # 閾値以上の物体があるか確認
        # ----------
        # if len(BBoxList) > self.numObjThreshold:
        #     return True

        # ----------
        # 閾値以下の信頼度があるか
        # ----------
        if self._hasLowConfDet(BBoxList):
            return True

        return False

    def _shouldSwitchToNversion(self, boundingBoxList: list[DetectionBoundingBox]) -> bool:
        minConfidenceScore: float = 1.0
        for boundingBox in boundingBoxList:
            confidenceScore = boundingBox.confidenceScore
            minConfidenceScore = min(minConfidenceScore, confidenceScore)

        if minConfidenceScore < self.confidenceScoreThreshold:
            return True
        elif minConfidenceScore >= self.confidenceScoreThreshold:
            return False

    def _shouldSwitchToOneVersion(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        if self._hasTrafficLight(BBoxList):
            return False
        if self._hasLowConfDet(BBoxList):
            return False

        return True

    def _shouldSwitchCovToOneVersion(self, BBoxList: list[DetectionBoundingBox]) -> bool:
        if self._hasTrafficLight(BBoxList):
            return False
        return True

    def outputCurrentStateAtConsole(self):
        currentState = self.state
        status = "SINGLE" if currentState == UseCaseVersionState.ONE else "MULTI"
        sys.stdout.write(f"\r[{status:6s}] Currently executing...")
        sys.stdout.flush()
