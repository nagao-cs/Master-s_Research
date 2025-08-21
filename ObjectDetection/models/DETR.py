from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from models.AbstractObjectDetector import AbstractObjectDetector
import cv2
import numpy as np
import os
import csv


class DETRDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.processor = None
        self.load_model()
        
    def load_model(self):
        try:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        except Exception as e:
            print(f"Error loading DETR model: {e}")
            self.model = None
            self.processor = None
    
    def predict(self, image_path):
        if self.model is None or self.processor is None:
            print("DETR model or processor is not loaded")
            return []
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []
        
        input = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**input)
        
        height, width = image.shape[:2]
        target_sizes = torch.tensor([[height, width]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        bboxes = list()
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            conf = score.item()
            class_id = label.item()
            label_name = self.model.config.id2label[class_id]
            bboxes.append({
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'class_id': class_id,
                'confidence': conf,
                'label': label_name
            })
        return bboxes
    
    def draw_bbox(self, image, bboxes):
        for bbox in bboxes:
            xmin = int(bbox['xmin'])
            xmax = int(bbox['xmax'])
            ymin = int(bbox['ymin'])
            ymax = int(bbox['ymax'])
            label = bbox['label']
            conf = bbox['confidence']
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image
    
    def save_result(self, image_path, bboxes, map, camera, index):
        image = cv2.imread(image_path)
        
        # === 保存先のディレクトリを作成 ===
        output_image_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection/output/{map}/images/DETR_results/{camera}"
        output_label_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection/output/{map}/labels/DETR_results/{camera}"
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