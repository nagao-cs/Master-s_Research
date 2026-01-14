import os
import cv2


class DatasetWriter:
    """生成されたデータセットをファイルシステムに保存するクラス"""

    def __init__(self, config):
        self.cfg = config

    def save_frame(self, frame_id, results):
        """
        frame_id: 連番
        results: LabelGenerator.process_frameの戻り値
        """
        for cam_name, data in results.items():
            # 保存パスの構築
            # 例: dataset/images/front/000001.png
            img_dir = os.path.join(self.cfg.output_img_dir, cam_name)
            bbox_dir = os.path.join(
                self.cfg.base_output_dir, "bbox_debug", cam_name)  # デバッグ用
            lbl_dir = os.path.join(self.cfg.output_label_dir, cam_name)

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            # 1. 画像保存
            img_path = os.path.join(img_dir, f"{frame_id:06d}.png")
            cv2.imwrite(img_path, data['original_image'])

            # 2. BBox画像保存（デバッグ用）
            bbox_path = os.path.join(bbox_dir, f"{frame_id:06d}.png")
            cv2.imwrite(bbox_path, data['bbox_image'])

            # 3. ラベル保存 (YOLO format: class x_center y_center width height)
            # distはYOLO標準ではないが、元のコードに合わせて保存する場合は残すか、削除する
            lbl_path = os.path.join(lbl_dir, f"{frame_id:06d}.txt")
            with open(lbl_path, 'w') as f:
                for label in data['labels']:
                    # label: [class_id, xc, yc, w, h, dist]
                    # YOLO標準形式にするならdistを除外: f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    cls, xc, yc, w, h, dist = label
                    f.write(
                        f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {dist:.2f}\n")
