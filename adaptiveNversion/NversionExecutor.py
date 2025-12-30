from .integrator import MajorityIntegrator
from ObjectDetection.boundingbox.boundingBox import BoundingBox


class NversionExecutor:
    def __init__(self, detectors: list, detectionIntegrator: MajorityIntegrator):
        self.detectors: list = detectors
        self.baseDetector: object = detectors[0]
        self.detectionIntegrator: MajorityIntegrator = detectionIntegrator

    def addDetector(self, detector: object):
        self.detectors.append(detector)

    def setBaseDetector(self, detector: object):
        self.baseDetector = detector

    def executeOneVersionDetection(self, imagePath: str) -> list[BoundingBox]:
        return self.baseDetector.predict(imagePath)

    def executeNVersionDetection(self, imagePath: str) -> tuple[list[BoundingBox], list[list[BoundingBox]]]:
        detectionsModelDict: dict[object, list[BoundingBox]] = dict()
        for detector in self.detectors:
            detections: list[BoundingBox] = detector.predict(imagePath)
            detectionsModelDict[detector] = detections

        integratedBoundingBoxList, groupedBoudingBoxList = self.detectionIntegrator(
            detectionsModelDict)
        return integratedBoundingBoxList, groupedBoudingBoxList
