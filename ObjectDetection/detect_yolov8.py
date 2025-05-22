import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

def detect_and_save_results(image_dir, output_dir, model_path='yolov8n.pt', confidence_threshold=0.25):
    """
    指定されたディレクトリ内の各画像に対してYOLOv8で物体検出を行い、
    結果を画像とテキストで保存します。

    Args:
        image_dir (str): 物体検出を行う画像が保存されているディレクトリのパス。
        output_dir (str): 結果を保存するディレクトリのパス。
        model_path (str): 使用するYOLOv8モデルのパス（例: 'yolov8n.pt'）。
                          事前にダウンロード済みのモデル、またはUltralyticsが自動ダウンロードするモデル名を指定。
        confidence_threshold (float): 検出の信頼度閾値。この値以上の検出のみを保存。
    """

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    os.makedirs(os.path.join(os.path.abspath(output_dir), 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.abspath(output_dir), 'labels'), exist_ok=True)

    # YOLOv8モデルのロード
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully.")

    # サポートする画像ファイルの拡張子
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # ディレクトリ内の画像を処理
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, filename)
            base_name = os.path.splitext(filename)[0]

            print(f"Processing image: {filename}")

            try:
                # 画像をPILで開く（OpenCVで読み込むことも可能だが、UltralyticsはPIL/numpy形式を好む場合がある）
                # results = model(image_path, conf=confidence_threshold) # これでも動作する
                
                # 画像をOpenCVで読み込み、RGBに変換
                img_cv2 = cv2.imread(image_path)
                if img_cv2 is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

                # YOLOv8による物体検出を実行
                # resultsはリストを返す
                results = model(img_rgb, conf=confidence_threshold, verbose=False) # verbose=Falseでログ出力を抑制

                # 検出結果の処理
                for r in results:
                    # 検出されたオブジェクトの bounding box、クラス、信頼度を取得
                    boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 (ピクセル座標)
                    confidences = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy().astype(int)
                    class_names = model.names # モデルが持つクラス名

                    # 結果描画用の画像を用意 (元のOpenCV画像をコピーして使用)
                    annotated_img = img_cv2.copy()

                    # ラベルファイルに書き込むためのリスト
                    label_lines = []

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = map(int, boxes[i])
                        conf = confidences[i]
                        cls_id = class_ids[i]
                        cls_name = class_names[cls_id]

                        # バウンディングボックスとラベルを画像に描画
                        color = (0, 255, 0) # 緑色
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # YOLO形式のテキストファイルに書き込むための情報を準備
                        # 形式: <class_id> <center_x> <center_y> <width> <height> (正規化された座標)
                        img_height, img_width, _ = img_cv2.shape
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        label_lines.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

                    # 検出結果が描画された画像を保存
                    output_image_path = os.path.join(output_dir, 'images', f"{base_name}_detected.jpg")
                    cv2.imwrite(output_image_path, annotated_img)
                    print(f"Saved annotated image to: {output_image_path}")

                    # 検出結果のテキストファイルを保存
                    output_label_path = os.path.join(output_dir, 'labels', f"{base_name}.txt")
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(label_lines))
                    print(f"Saved detection labels to: {output_label_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # ここに入力ディレクトリと出力ディレクトリのパスを指定してください
    input_images_directory = "C:\CARLA_Latest\WindowsNoEditor\PythonAPI\examples\output\images"  # 例: "my_images"
    output_results_directory = "yolov8_results"      # 例: "output_detections"

    # 使用するYOLOv8モデルのパス
    # 事前にダウンロードしたい場合は以下のように実行:
    # from ultralytics import YOLO
    # YOLO('yolov8n.pt').export(format='pt') # 'pt'形式でダウンロード
    yolov8_model_to_use = 'yolov8n.pt' # 'yolov8s.pt', 'yolov8m.pt' なども選択可能

    # 信頼度閾値
    conf_threshold = 0.5 # 検出結果の信頼度がこの値以上の場合に保存

    detect_and_save_results(input_images_directory, output_results_directory, yolov8_model_to_use, conf_threshold)
    print("All images processed. Check the 'yolov8_results' directory.")