class NversionExecutor:
    def __init__(self, detectors: list, integrator: object):
        self.detectors: list = detectors
        self.base_detector: object = detectors[0]
        self.integrator = integrator

    def add_detector(self, detector: object):
        self.detectors.append(detector)

    def set_base_detector(self, detector: object):
        self.base_detector = detector

    def execute_1version(self, image_path: str):
        return self.base_detector.predict(image_path)

    def execute_Nversion(self, image_path: str):
        detections_by_model = dict()
        for detector in self.detectors:
            detections = detector.predict(image_path)
            detections_by_model[detector] = detections

        integrated_detections = self.integrator(detections_by_model)
        return integrated_detections, detections_by_model
