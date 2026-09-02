from pathlib import Path
import csv
import os

from src.adaptiveNversion.versionController.usecaseVersionController import UseCaseVersionState


class StatsRecorder:
    def __init__(self, modelNameList: list[str]):
        self.model: str = f"{'_'.join(modelNameList)}"
        self.numInference: int = 0
        self.numTransitionFromOneToN: int = 0
        self.numTransitionFromNToOne: int = 0
        self.stateList: list[UseCaseVersionState] = []

        self.totalExecutionTime: float = 0.0
        self.throuput: float = 0.0
        self.numProcessedImage: int = 0
        self.numOneVersionDetectionCount: int = 0
        self.numNVersionDetectionCount: int = 0
        self._previousState: UseCaseVersionState = UseCaseVersionState.ONE
        self._maxVersion = len(modelNameList)

    def update(self, versionsState: UseCaseVersionState):
        self.stateList.append(versionsState)
        if versionsState == UseCaseVersionState.ONE:
            self.numInference += 1
            self.numOneVersionDetectionCount += 1
            if self._previousState == UseCaseVersionState.N:
                self.numTransitionFromNToOne += 1
        elif versionsState == UseCaseVersionState.N:
            self.numInference += self._maxVersion
            self.numNVersionDetectionCount += 1
            if self._previousState == UseCaseVersionState.ONE:
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

    def saveStateTransition(self, stateTransitionCsvFilePath: Path):
        columnName: str = self.model
        numFrame: int = len(self.stateList)

        if os.path.exists(stateTransitionCsvFilePath):
            with open(stateTransitionCsvFilePath, "r", newline="") as csvFile:
                reader = list(csv.reader(csvFile))
                header = reader[0]
                rows = reader[1:]
        else:
            header = ["frame_id"]
            rows = [[str(i)] for i in range(numFrame)]

        # フレーム数チェック
        if len(rows) < numFrame:
            for i in range(len(rows), numFrame):
                rows.append([str(i)])

        # すでに列が存在する場合は上書き
        if columnName in header:
            colIdx = header.index(columnName)
        else:
            header.append(columnName)
            colIdx = len(header) - 1
            for row in rows:
                row.append("")

        for i, state in enumerate(self.stateList):
            if state == UseCaseVersionState.ONE:
                rows[i][colIdx] = "1"
            elif state == UseCaseVersionState.COV_STATE:
                rows[i][colIdx] = str("Cov")
            elif state == UseCaseVersionState.CER_STATE:
                rows[i][colIdx] = str("Cer")
            else:
                raise ValueError(f"Unknown UseCaseVersionState: {state}")

        with open(stateTransitionCsvFilePath, "w", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(header)
            writer.writerows(rows)
