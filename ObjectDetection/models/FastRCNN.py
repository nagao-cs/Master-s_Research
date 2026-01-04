import tensorflow_hub as hub
import cv2
import numpy as np
import tensorflow as tf
from .AbstractObjectDetector import AbstractObjectDetector
from ..utils import utils
from boundingBox.boundingBox import DetectionBoundingBox


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
            raise RuntimeError(f"Error loading fastrcnn model: {e}")

    def predict(self, imagePath):
        """
        他のモデルと同じ形式で出力：
        正規化座標 (center_x, center_y, width, height, confidence, class_id)
        """
        image = cv2.imread(imagePath)
        if image is None:
            raise FileExistsError(f"Could not read image: {imagePath}")

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

        outputBoundingBoxList = list()

        for i in range(num_detections):
            confidenceScore = float(scores[i])
            if confidenceScore < utils.CONF_THRESHOLD:
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
            size = (xmax - xmin) * (ymax - ymin)
            if size < utils.SIZE_THRESHOLD:
                continue

            # ピクセル座標 -> 正規化座標（YOLO形式）
            xCenter = ((xmin + xmax) / 2) / orig_width
            yCenter = ((ymin + ymax) / 2) / orig_height
            width = (xmax - xmin) / orig_width
            height = (ymax - ymin) / orig_height

            # 正規化座標が [0, 1] 範囲内か確認
            if not (0 <= xCenter <= 1 and 0 <= yCenter <= 1):
                continue
            if not (0 < width <= 1 and 0 < height <= 1):
                continue

            classId = classes[i] - 1
            label = utils.COCO_LABELS.get(classId, 'unknown')

            boundingBoxInstance = DetectionBoundingBox(
                xCenter, yCenter, width, height, classId, label, confidenceScore)

            outputBoundingBoxList.append(boundingBoxInstance)

        return outputBoundingBoxList
