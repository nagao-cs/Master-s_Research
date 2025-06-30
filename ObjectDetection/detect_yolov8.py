import os
from ultralytics import YOLO
from PIL import Image
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

SIZE_THRESHOLD=0.01

def predict_image(model, image, size_threshold, conf_threshold=0.25):
    output = list()
    results = model(image, conf=conf_threshold, verbose=False)
    result = results[0]
    bboxes = result.boxes
    for bbox in bboxes:
        if bbox.conf > conf_threshold:
            xmin, ymin, xmax, ymax = bbox.xyxy[0].tolist()
            size = (xmax-xmin)*(ymax-ymin)
            if size < size_threshold:
                continue
            class_id = int(bbox.cls[0])
            conf = bbox.conf[0].item()
            label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else 'unknown'
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
        
def draw_bbox(image, bboxes):
    for bbox in bboxes:
        xmin = int(bbox['xmin'])
        xmax = int(bbox['xmax'])
        ymin = int(bbox['ymin'])
        ymax = int(bbox['ymax'])
        label = bbox['label']
        conf = bbox['confidence']
        
        # バウンディングボックスを描画
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # ラベルと信頼度を描画
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def detect_and_save_results(input_image_dir, output_image_dir, output_label_dir, map, camera, model_path='yolov8n.pt', confidence_threshold=0.25):
    # 出力ディレクトリが存在しない場合は作成
    # output_image_dir = os.path.join(output_image_dir, map, camera)
    # output_label_dir = os.path.join(output_label_dir, map, camera)
    # os.makedirs(output_image_dir, exist_ok=True)
    # os.makedirs(output_label_dir, exist_ok=True)

    # YOLOv8モデルのロード
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully.")

    # ディレクトリ内の画像を処理
    count = 0
    for path in os.listdir(input_image_dir):
        image_path = os.path.join(input_image_dir, path)
        image = cv2.imread(image_path)
        output = predict_image(model, image, size_threshold=SIZE_THRESHOLD, conf_threshold=confidence_threshold)
        print(f"processed {path}")
        annotated_image = draw_bbox(image, output)
        annotated_image_path = os.path.join(output_image_dir, f"{count:06d}.png")
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"saveed annotated image to {annotated_image_path}")
        labels = list()
        for bbox in output:
            xmin = bbox['xmin']
            xmax = bbox['xmax']
            ymin = bbox['ymin']
            ymax = bbox['ymax']
            cls_id = bbox['class_id']
            conf = bbox['confidence']
            
            labels.append([cls_id, xmin, xmax, ymin, ymax, conf])
        label_path = os.path.join(output_label_dir, f"{count:06d}.csv")
        with open(label_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['class_id', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])
            for label in labels:
                writer.writerow(label)
        print(f"saved labels to {label_path}")
        count += 1

if __name__ == "__main__":
    yolov8_model_to_use = 'yolov8n.pt' 
    conf_threshold = 0.5
    input_base_dir = "C:\CARLA_Latest\WindowsNoEditor\output\image"
    output_image_dir = "yolov8_results/images"
    output_label_dir = "yolov8_results/labels"
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    map = "Town01_Opt"
    cameras = ["front", "left_1", "right_1"]
    for camera in cameras:
        input_images_directory = os.path.join(input_base_dir, map, camera)
        if not os.path.exists(input_images_directory):
            print(f"Input directory does not exist: {input_images_directory}")
            continue
        
        detect_and_save_results(input_images_directory, output_image_dir, output_label_dir, map, camera, yolov8_model_to_use, conf_threshold)
    
    print("All images processed. Check the 'yolov8_results' directory.")