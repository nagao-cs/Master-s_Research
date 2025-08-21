from ultralytics import YOLO
import cv2
import numpy as np
import csv
from models.AbstractObjectDetector import AbstractObjectDetector
import os

class Yolov8nDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            self.model = YOLO("C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\yolov8n.pt")
            print(f"YOLOv8n model loaded")
            
        except Exception as e:
            print(f"Error loading YOLOv8n model: {e}")
            self.model = None
    
    def predict(self, image):
        if self.model is None:
            print("Model is not loaded")
            return []
        image = cv2.imread(image)
        if image is None:
            print(f"Could not read image: {image}")
            return []
        detections = self.model(image)
        bboxes = list()
        bboxes = detections[0].boxes
        output = list()
        for bbox in bboxes:
            if bbox.conf > 0.1:
                xmin, ymin, xmax, ymax = bbox.xyxy[0].tolist()
                size = (xmax - xmin) * (ymax - ymin)
                if size < 100:
                    continue
                class_id = int(bbox.cls[0])
                conf = bbox.conf[0].item()
                label = self.model.names[class_id] if class_id < len(self.model.names) else 'unknown'
                output.append({
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'class_id': class_id,
                    'confidence': conf,
                    'label': label
                })
        return output

    def draw_bbox(self, image, bboxes):
        for bbox in bboxes:
            xmin = int(bbox['xmin'])
            xmax = int(bbox['xmax'])
            ymin = int(bbox['ymin'])
            ymax = int(bbox['ymax'])
            label = bbox['label']
            conf = bbox['confidence']
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
    
    def save_result(self, image_path, bboxes, map, camera, index):
        image  = cv2.imread(image_path)
        # === 保存先のディレクトリを作成 ===
        output_image_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection/output/{map}/images/yolov8n_results/{camera}"
        output_label_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection/output/{map}/labels/yolov8n_results/{camera}"
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # === バウンディングボックスを描画した画像を保存 ===
        bbox_image = self.draw_bbox(image, bboxes)
        output_image_path = os.path.join(output_image_dir, f"{index}.png")
        cv2.imwrite(output_image_path, bbox_image)
        print(f"Saved image with bounding boxes to {output_image_path}")
        
        # === 検出結果をcsvファイルに保存 ===
        output_label_path = os.path.join(output_label_dir, f"{index}.csv")
        with open(output_label_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['class_id', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])
            for bbox in bboxes:
                writer.writerow([bbox['class_id'], bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax'], bbox['confidence']])
        print(f"Saved labels to {output_label_path}")