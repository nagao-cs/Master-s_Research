import tensorflow_hub as hub
import cv2
import numpy as np
from AbstractObjectDetector import AbstractObjectDetector
import os
import csv
import tensorflow as tf

class FastRCNNDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            self.model = hub.load("https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/TensorFlow2/640x640/1")
            print(f"Fast R-CNN model loaded")
        except Exception as e:
            print(f"Error loading Fast R-CNN model: {e}")
            self.model = None
    
    def predict(self, image_path):
        if self.model is None:
            print("Model is not loaded")
            return []
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
        image = np.array(image, dtype=np.uint8)
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        
        detections = self.model(image_tensor)
        # - detection_boxes: バウンディングボックスの座標 (ymin, xmin, ymax, xmax)
        # - detection_classes: 検出されたオブジェクトのクラスID
        # - detection_scores: 検出の信頼度スコア
        # - num_detections: 検出されたオブジェクトの数
        num_detections = int(detections['num_detections'][0])
        bboxes = detections['detection_boxes'][0].numpy()[:num_detections]
        classes = detections['detection_classes'][0].numpy().astype(int)[:num_detections]
        scores = detections['detection_scores'][0].numpy()[:num_detections]
        
        output = list()
        for i in range(num_detections):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = bboxes[i]
                class_id = classes[i]
                conf = scores[i]
                label = f"Class {class_id}"
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
        original_width, original_height = image.shape[1], image.shape[0]
        for bbox in bboxes:
            xmin = int(bbox['xmin'] * original_width)
            xmax = int(bbox['xmax'] * original_width)
            ymin = int(bbox['ymin'] * original_height)
            ymax = int(bbox['ymax'] * original_height)
            label = bbox['label']
            conf = bbox['confidence']
            
            # === バウンディングボックスを描画 ===
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
    
    def save_result(self, image_path, bboxes, map, camera, index):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        original_width, original_height = image.shape[1], image.shape[0]
        # === 保存先のディレクトリを作成 ===
        output_image_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\output/{map}/images/FastRCNN_results/{camera}"
        output_label_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\output/{map}/labels/FastRCNN_results/{camera}"
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # === 画像とラベルの保存パス ===
        output_image_path = os.path.join(output_image_dir, f"{index}.png")
        output_label_path = os.path.join(output_label_dir, f"{index}.csv")
        
        # === バウンディングボックスを描画した画像を保存 ===
        bbox_image = self.draw_bbox(image, bboxes)
        cv2.imwrite(output_image_path, bbox_image)
        print(f"Saved image with bounding boxes to {output_image_path}")
        
        # === 検出結果をcsvファイルに保存 ===
        with open(output_label_path, 'w', newline='') as csvfile:
            fieldnames = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for bbox in bboxes:
                writer.writerow({
                    'class_id': bbox['class_id'],
                    'xmin': bbox['xmin'] * original_width,
                    'xmax': bbox['xmax'] * original_width,
                    'ymin': bbox['ymin'] * original_height,
                    'ymax': bbox['ymax'] * original_height,
                    'confidence': bbox['confidence']
                })
        
        print(f"Results saved to {output_image_path} and {output_label_path}")