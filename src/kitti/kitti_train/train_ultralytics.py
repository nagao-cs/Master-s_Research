from ultralytics import YOLO, RTDETR
import argparse
from pathlib import Path

def build_model(model_name):
    if model_name == "yolov8n":
        return YOLO("yolov8n.pt")
    elif model_name == "yolov11n":
        return YOLO("yolo11n.pt")
    elif model_name == "yolov5n":
        return YOLO("yolov8n.pt")
    elif model_name == "rtdetr":
        return RTDETR("rtdetr-l.pt")
    else :
        raise ValueError(f"invalid model name: {model_name}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        required=True,
    )
    args = argparser.parse_args()
    model = build_model(model_name=args.model)

    cwd = Path(__file__).parent
    # 学習
    model.train(
        data=cwd/"data.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,           # GPU番号
        project=cwd/"runs/train",
        name=f"{args.model}_kitti",
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        patience=20,
        save=True,
    )