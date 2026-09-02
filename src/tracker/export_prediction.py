"""
export_predictions.py
------------------------------
モデル検出と、SORTトラッカーの
    - predicted: 現在の検出で補正する前の線形予測状態
    - updated:   現在の検出で補正した後の状態
を、フレームごとにKITTI形式に近いラベルファイルとして書き出す。

検出とトラックの差分評価は別スクリプトで行う想定のため、
ここでは検出・トラッキングとラベル保存のみを行う。

出力構成:
    {output_dir}/{model_name}/labels/               モデル検出
    {output_dir}/{model_name}_tracker/labels/        トラッカー補正後 (updated)
    {output_dir}/{model_name}_tracker_pred/labels/   トラッカー補正前 (predicted)

ラベル形式（1行1box）:
    class_id x_center y_center width height confidence
    （正規化座標、center形式。tracker_evaluation.py の _save_labels と同じ形式）

使い方:
    python -m src.time_aware_exp.export_predictions --model yolov8n --dataset KITTI
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tqdm import tqdm

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.ObjectDetection.models.factory import build_model
from src.util.dataset import build_dataset
from src.config import DATASET_DIR
from .tracker import SortTracker


def write_boxes(boxes: list[DetectionBoundingBox], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for box in boxes:
            f.write(
                f"{box.classId} {box.xCenter:.3f} {box.yCenter:.3f} "
                f"{box.width:.3f} {box.height:.3f} {box.confidenceScore:.6f}\n"
            )


def run(model_name: str, dataset_name: str, map_name: str) -> None:
    dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
    model   = build_model(model_name=model_name, dataset=dataset_name, device="cuda")
    tracker = SortTracker()

    output_dir           = DATASET_DIR / f"tracker/{dataset_name}/{map_name}"
    model_label_dir       = output_dir / f"{model_name}" / "labels"
    updated_label_dir     = output_dir / f"{model_name}_tracker" / "labels"
    predicted_label_dir   = output_dir / f"{model_name}_tracker_pred" / "labels"

    for d in (model_label_dir, updated_label_dir, predicted_label_dir):
        d.mkdir(parents=True, exist_ok=True)

    for image_path, _label_path in tqdm(dataset, desc="detect + track"):
        detections = model.predict(image_path)
        result     = tracker.update(detections)

        filename = image_path.stem + ".txt"
        write_boxes(detections,           model_label_dir     / filename)
        write_boxes(result.tracked_boxes,   updated_label_dir   / filename)
        write_boxes(result.predicted_boxes, predicted_label_dir / filename)

    print(f"Saved labels under {output_dir}")
    print(f"  model:            {model_label_dir}")
    print(f"  tracker updated:  {updated_label_dir}")
    print(f"  tracker predicted:{predicted_label_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export model / tracker predicted & updated labels for diff evaluation"
    )
    parser.add_argument("--model",   type=str, default="yolov8n", help="Model name (default: yolov8n)")
    parser.add_argument("--dataset", type=str, default="CARLA", choices=["CARLA", "KITTI"],
                         help="Dataset type: CARLA or KITTI (default: CARLA)")
    parser.add_argument("--map",     type=str, default="Town02", help="CARLA map name (default: Town02)")
    args = parser.parse_args()

    run(model_name=args.model, dataset_name=args.dataset, map_name=args.map)