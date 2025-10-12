from abc import ABC, abstractmethod


class DetectionResult:
    def __init__(self):
        pass


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

    def draw_bbox(self, image, bboxes):
        import cv2
        im_width = image.shape[1]
        im_height = image.shape[0]
        for bbox in bboxes:
            x_center = bbox['x_center'] * im_width
            y_center = bbox['y_center'] * im_height
            width = bbox['width'] * im_width
            height = bbox['height'] * im_height
            xmin = int(x_center - width / 2)
            xmax = int(x_center + width / 2)
            ymin = int(y_center - height / 2)
            ymax = int(y_center + height / 2)
            label = bbox['label']
            conf = bbox['confidence']
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def save_result(self, image_path, bboxes, map, camera, index, model_name):
        import os
        import cv2
        """検出結果を保存（共通処理）"""
        image = cv2.imread(image_path)

        # 出力ディレクトリの設定
        output_image_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection/output/{map}/images/{model_name}/{camera}"
        output_label_dir = f"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection/output/{map}/labels/{model_name}/{camera}"
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # 画像の保存
        bbox_image = self.draw_bbox(image, bboxes)
        output_image_path = os.path.join(output_image_dir, f"{index}.png")
        cv2.imwrite(output_image_path, bbox_image)

        # ラベルの保存
        output_label_path = os.path.join(output_label_dir, f"{index}.txt")
        with open(output_label_path, 'w') as f:
            for bbox in bboxes:
                x_center = bbox['x_center']
                y_center = bbox['y_center']
                width = bbox['width']
                height = bbox['height']
                conf = bbox['confidence']
                label = bbox['label']
                class_id = bbox['class_id']
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
