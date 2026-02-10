from .integrator.integrator import Integrator
from src.boundingBox.boundingBox import DetectionBoundingBox


class UseCaseNversionExecutor:
    def __init__(self, detectors: list, detectionIntegrator: Integrator, covComb: list[object], cerCovb: list[object]):
        self.detectors: list = detectors
        self.detectionIntegrator: Integrator = detectionIntegrator
        self.setBaseDetector(detectors[0])
        self.covDetectors: list[object] = covComb
        self.cerDetectors: list[object] = cerCovb

    def addDetector(self, detector: object):
        self.detectors.append(detector)

    def setBaseDetector(self, detector: object):
        self.baseDetector = detector

    def executeOneVersionDetection(self, imagePath: str) -> list[DetectionBoundingBox]:
        return self.baseDetector.predict(imagePath)

    def executeNMinusOneVersionDetection(self, imagePath: str, baseDetection: list[DetectionBoundingBox]) -> list[DetectionBoundingBox]:
        detectionsModelDict: dict[object, list[DetectionBoundingBox]] = {
            self.baseDetector: baseDetection}

        for detector in self.detectors:
            if detector == self.baseDetector:
                continue
            detections: list[DetectionBoundingBox] = detector.predict(
                imagePath)
            detectionsModelDict[detector] = detections

        integratedBoundingBoxList, groupedBoudingBoxList = self.detectionIntegrator(
            detectionsModelDict)
        return integratedBoundingBoxList, groupedBoudingBoxList

    def executeNVersionDetection(self, imagePath: str) -> tuple[list[DetectionBoundingBox], list[list[DetectionBoundingBox]]]:
        detectionsModelDict: dict[object, list[DetectionBoundingBox]] = dict()
        for detector in self.detectors:
            detections: list[DetectionBoundingBox] = detector.predict(
                imagePath)
            detectionsModelDict[detector] = detections

        integratedBoundingBoxList, groupedBoudingBoxList = self.detectionIntegrator(
            detectionsModelDict)
        return integratedBoundingBoxList, groupedBoudingBoxList

    def exeCovDetection(self, imagePath: str) -> tuple[list[DetectionBoundingBox], list[list[DetectionBoundingBox]]]:
        detectionsModelDict: dict[object, list[DetectionBoundingBox]] = dict()
        for detector in self.covDetectors:
            detections: list[DetectionBoundingBox] = detector.predict(
                imagePath)
            detectionsModelDict[detector] = detections

        integratedBoundingBoxList, groupedBoudingBoxList = self.detectionIntegrator(
            detectionsModelDict)
        return integratedBoundingBoxList, groupedBoudingBoxList

    def exeCerDetection(self, imagePath: str) -> tuple[list[DetectionBoundingBox], list[list[DetectionBoundingBox]]]:
        detectionsModelDict: dict[object, list[DetectionBoundingBox]] = dict()
        for detector in self.cerDetectors:
            detections: list[DetectionBoundingBox] = detector.predict(
                imagePath)
            detectionsModelDict[detector] = detections

        integratedBoundingBoxList, groupedBoudingBoxList = self.detectionIntegrator(
            detectionsModelDict)
        return integratedBoundingBoxList, groupedBoudingBoxList
