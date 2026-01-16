from adaptiveNversion.versionController.VersionController import VersionState
from pathlib import Path
import csv
import os


class StatsRecorder:
    def __init__(self, modelNameList: list[str]):
        self.model: str = f"{'_'.join(modelNameList)}"
        self.numInference: int = 0
        self.numTransitionFromOneToN: int = 0
        self.numTransitionFromNToOne: int = 0
        self.totalExecutionTime: float = 0.0
        self.throuput: float = 0.0
        self.numProcessedImage: int = 0
        self.numOneVersionDetectionCount: int = 0
        self.numNVersionDetectionCount: int = 0
        self._previousState: VersionState = VersionState.ONE
        self._maxVersion = len(modelNameList)

    def update(self, versionsState: VersionState):
        if versionsState == VersionState.ONE:
            self.numInference += 1
            self.numOneVersionDetectionCount += 1
            if self._previousState == VersionState.N:
                self.numTransitionFromNToOne += 1
        elif versionsState == VersionState.N:
            self.numInference += self._maxVersion
            self.numNVersionDetectionCount += 1
            if self._previousState == VersionState.ONE:
                self.numTransitionFromOneToN += 1

        self._previousState = versionsState
        self.numProcessedImage += 1

    def registerExecutionTime(self, executionTime: float):
        self.totalExecutionTime = executionTime
        self.throuput = self.numProcessedImage / self.totalExecutionTime

    def writeStatsToCsvFile(self, statsWriteCsvFilePath: Path):
        header: list[str] = ['model', 'num Inferece', 'num Oneversion Detection', 'num N-Version Detection', 'num Transition from One to N',
                             'num Transition from N to One', 'execution Time', 'throuput']
        writeContent: list = [self.model, self.numInference, self.numOneVersionDetectionCount, self.numNVersionDetectionCount, self.numTransitionFromOneToN,
                              self.numTransitionFromNToOne, self.totalExecutionTime, self.throuput]
        if not os.path.exists(statsWriteCsvFilePath):
            with open(statsWriteCsvFilePath, mode='w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(header)
                writer.writerow(writeContent)

        elif os.path.exists(statsWriteCsvFilePath):
            with open(statsWriteCsvFilePath, mode='a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(writeContent)
