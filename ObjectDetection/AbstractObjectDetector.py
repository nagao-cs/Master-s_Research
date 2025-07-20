from abc import ABC, abstractmethod

class AbstractObjectDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def predict(self, image):
        pass
    
    @abstractmethod
    def draw_bbox(self, image, bboxes):
        pass
    
    @abstractmethod
    def save_result(self, image, bboxes, output_image_path, output_label_path):
        pass