from .integrator import MajorityIntegrator
from boundingBox.boundingBox import DetectionBoundingBox


class NversionExecutor:
    def __init__(self, detectors: list, detectionIntegrator: MajorityIntegrator):
        self.detectors: list = detectors
        self.detectionIntegrator: MajorityIntegrator = detectionIntegrator
        self.setBaseDetector(detectors[0])

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
