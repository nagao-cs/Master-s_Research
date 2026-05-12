import os
import cv2
from pathlib import Path

from .configuration import SimulationConfig


class DatasetWriter:
    """生成されたデータセットをファイルシステムに保存するクラス"""

    def __init__(self, config):
        self.cfg: SimulationConfig = config

    def save_frame(self, frame_id, results):
        """
        frame_id: 連番
        results: LabelGenerator.process_frameの戻り値
        """
        for cam_name, data in results.items():
            # 保存パスの構築
            imageDir: Path = self.cfg.outputImageDir / cam_name
            bbox_dir: Path = self.cfg.outputBBoxImageDir / cam_name
            labelDir: Path = self.cfg.outputLabelDir / cam_name

            os.makedirs(imageDir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)
            os.makedirs(labelDir, exist_ok=True)

            # 1. 画像保存
            img_path = os.path.join(imageDir, f"{frame_id:06d}.png")
            cv2.imwrite(img_path, data['original_image'])

            # 2. BBox画像保存（デバッグ用）
            bbox_path = os.path.join(bbox_dir, f"{frame_id:06d}.png")
            cv2.imwrite(bbox_path, data['bbox_image'])

            # 3. ラベル保存 (YOLO format: class x_center y_center width height)
            # distはYOLO標準ではないが、元のコードに合わせて保存する場合は残すか、削除する
            lbl_path = os.path.join(labelDir, f"{frame_id:06d}.txt")
            print(lbl_path)
            with open(lbl_path, 'w') as f:
                for label in data['labels']:
                    # label: [class_id, xc, yc, w, h, dist]
                    # YOLO標準形式にするならdistを除外: f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    cls, xc, yc, w, h, dist = label
                    f.write(
                        f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {dist:.2f}\n")
