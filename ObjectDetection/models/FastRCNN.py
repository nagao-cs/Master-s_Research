import tensorflow_hub as hub
import cv2
import numpy as np
from models.AbstractObjectDetector import AbstractObjectDetector
import tensorflow as tf
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FastRCNNDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = hub.load(
                "https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/TensorFlow2/640x640/1")
            print(f"Fast R-CNN model loaded")
        except Exception as e:
            print(f"Error loading Fast R-CNN model: {e}")
            self.model = None

    def predict(self, image_path):
        """
        他のモデルと同じ形式で出力：
        正規化座標 (center_x, center_y, width, height, confidence, class_id)
        """
        if self.model is None:
            print("Model is not loaded")
            return []

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []

        # 元の画像サイズを保存（正規化用）
        orig_height, orig_width = image.shape[:2]

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # モデル入力サイズにリサイズ
        input_size = 640
        resized_image = cv2.resize(image, (input_size, input_size))
        image_array = np.array(resized_image, dtype=np.uint8)
        image_tensor = tf.convert_to_tensor(image_array)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # 推論
        detections = self.model(image_tensor)
        num_detections = int(detections['num_detections'][0])
        bboxes = detections['detection_boxes'][0].numpy()[:num_detections]
        classes = detections['detection_classes'][0].numpy().astype(int)[
            :num_detections]
        scores = detections['detection_scores'][0].numpy()[:num_detections]

        output = []
        confidence_threshold = 0.25

        for i in range(num_detections):
            if scores[i] < confidence_threshold:
                continue

            # 検出結果（リサイズ後の画像座標）
            ymin_resized, xmin_resized, ymax_resized, xmax_resized = bboxes[i]

            # リサイズ後の座標 -> 元の画像座標に戻す
            scale_x = orig_width / input_size
            scale_y = orig_height / input_size

            xmin = xmin_resized * scale_x
            xmax = xmax_resized * scale_x
            ymin = ymin_resized * scale_y
            ymax = ymax_resized * scale_y

            # ピクセル座標 -> 正規化座標（YOLO形式）
            center_x_norm = ((xmin + xmax) / 2) / orig_width
            center_y_norm = ((ymin + ymax) / 2) / orig_height
            width_norm = (xmax - xmin) / orig_width
            height_norm = (ymax - ymin) / orig_height

            # 正規化座標が [0, 1] 範囲内か確認
            if not (0 <= center_x_norm <= 1 and 0 <= center_y_norm <= 1):
                continue
            if not (0 < width_norm <= 1 and 0 < height_norm <= 1):
                continue

            class_id = classes[i] - 1
            confidence = float(scores[i])

            output.append({
                'x_center': center_x_norm,
                'y_center': center_y_norm,
                'width': width_norm,
                'height': height_norm,
                'class_id': class_id,
                'confidence': confidence,
                'label': f"Class {class_id}"
            })
        return output
