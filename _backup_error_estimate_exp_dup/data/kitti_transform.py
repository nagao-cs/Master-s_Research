from pathlib import Path
from collections import defaultdict

cwd = Path(__file__).parent
d_drive = cwd.parent.parent.parent.parent.parent.parent / "d"
# 入力・出力ディレクトリ
input_dir = d_drive / "data_tracking_label_2" / "training" / "label_02" 
output_root = d_drive / "data_tracking_label_2" / "training" / "label_02" 
print(input_dir)

# クラス変換
type_to_id = {
    'Car': 2,
    'Van': 2,
    'Truck': 2,
    'Pedestrian': 0,
    'Person_sitting': 0,
    'Cyclist': 1,
    'Tram': 2,
}

# 画像サイズ
IMG_WIDTH = 1241
IMG_HEIGHT = 376

# 各KITTI txtを処理
for txt_file in input_dir.glob("*.txt"):

    # 例: 0000.txt → 0000/
    sequence_name = txt_file.stem
    sequence_dir = output_root / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)

    # frame_idごとに保持
    frame_annotations = defaultdict(list)

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            frame_id = int(parts[0])
            obj_type = parts[2]

            # 除外クラス
            if obj_type == "DontCare":
                continue

            # 未定義クラスをスキップ
            if obj_type not in type_to_id:
                continue

            xmin = float(parts[6])
            ymin = float(parts[7])
            xmax = float(parts[8])
            ymax = float(parts[9])

            # xywhへ変換
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            # 正規化
            xcenter /= IMG_WIDTH
            ycenter /= IMG_HEIGHT
            width /= IMG_WIDTH
            height /= IMG_HEIGHT

            annotation = (
                f"{type_to_id[obj_type]} "
                f"{xcenter:.6f} "
                f"{ycenter:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            frame_annotations[frame_id].append(annotation)

    # frameごとに保存
    for frame_id, annotations in frame_annotations.items():

        output_file = sequence_dir / f"{frame_id:06d}.txt"

        with open(output_file, "w") as f:
            f.write("\n".join(annotations))

print("変換完了")