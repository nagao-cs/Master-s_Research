import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

# === 設定 ===
# collectDataset.py で指定した出力先
cwd = Path(__file__).parent
SOURCE_DATASET_DIR = cwd.parent / "trainDataset"
# 学習用に整形したデータを保存する先
OUTPUT_DIR = cwd.parent / "trainDataset" / "yoloDataset"
TRAIN_RATIO = 0.8  # 学習データの割合

# ID_MAPPING = {
#     0: 0,   # Pedestrian -> 0
#     2: 1,   # Vehicle -> 1
#     9: 2,   # TrafficLight -> 2
#     11: 3   # TrafficSign -> 3
# }


def convert_label(src_path, dst_path):
    """
    ラベルファイルを読み込み、ID変換と6列目(dist)の削除を行って保存
    """
    converted_lines = []
    with open(src_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            classId, x, y, w, h = parts[0:5]  # 6列目のdistは無視
            # newClassId = ID_MAPPING[int(classId)]

            # converted_lines.append(f"{newClassId} {x} {y} {w} {h}\n")
            converted_lines.append(f"{classId} {x} {y} {w} {h}\n")

    # 変換後のラベルがある場合のみ保存
    if converted_lines:
        with open(dst_path, 'w') as f:
            f.writelines(converted_lines)
        return True
    return False


def main():
    # ディレクトリ作成
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 全画像のパスを取得（サブディレクトリも検索）
    # 例: dataset/images/front/*.png
    imageDirectory = os.path.join(SOURCE_DATASET_DIR, "images", "front")
    sourceImagePathList = [os.path.join(
        imageDirectory, imageFile) for imageFile in os.listdir(imageDirectory)]
    print(sourceImagePathList)

    if not sourceImagePathList:
        print("画像が見つかりません。パスを確認してください。")
        return

    # シャッフル
    random.seed(42)
    random.shuffle(sourceImagePathList)

    # 分割
    split_idx = int(len(sourceImagePathList) * TRAIN_RATIO)
    train_files = sourceImagePathList[:split_idx]
    val_files = sourceImagePathList[split_idx:]

    print(f"Total images: {len(sourceImagePathList)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # ファイルのコピーと変換
    for split, files in [('train', train_files), ('val', val_files)]:
        print(f"Processing {split} data...")
        for img_path in tqdm(files):
            # パスの処理
            img_name = os.path.basename(img_path)
            # 対応するラベルファイルのパスを構築
            # dataset/images/front/00000.png -> dataset/labels/front/00000.txt
            # 親フォルダ名(frontなど)を取得
            parent_dir = os.path.basename(os.path.dirname(img_path))
            label_path = os.path.join(
                SOURCE_DATASET_DIR, "labels", parent_dir, img_name.replace('.png', '.txt'))

            if not os.path.exists(label_path):
                continue

            # 出力先パス
            dst_img_path = os.path.join(
                OUTPUT_DIR, 'images', split, f"{parent_dir}_{img_name}")
            dst_label_path = os.path.join(
                OUTPUT_DIR, 'labels', split, f"{parent_dir}_{img_name.replace('.png', '.txt')}")

            # ラベル変換と保存 (空のラベルファイルになった場合は画像もコピーしない)
            if convert_label(label_path, dst_label_path):
                shutil.copy2(img_path, dst_img_path)

    print("データセットの準備が完了しました。")
    print(f"保存先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
