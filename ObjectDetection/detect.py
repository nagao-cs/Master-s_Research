import os
from logging import getLogger
from pathlib import Path
import time
import tqdm

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Object Detection on images")
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov8n", "yolov5n", "yolov11n",
                 "rtdetr", "ssd", "yolov8l", "fastrcnn"],
        help="Model to use: yolov8n, yolo11n, ssd, fastrcnn, yolov5n, mobilenet, detr",
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )
    argparser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )

    args = argparser.parse_args()
    model_name = args.model
    mapName = args.map
    conf_threshold = args.conf_threshold

    if model_name == 'yolov8n':
        from models.Yolov8n import Yolov8nDetector
        model = Yolov8nDetector()
    elif model_name == "yolov11n":
        from ObjectDetection.models.Yolov11n import Yolov11nDetector
        model = Yolov11nDetector()
    elif model_name == "yolov5n":
        from ObjectDetection.models.Yolov5n import Yolov5nDetector
        model = Yolov5nDetector()
    elif model_name == "rtdetr":
        from models.rtDETR import RTDETRDetector
        model = RTDETRDetector()
    elif model_name == 'yolov8l':
        from models.yolov8l import Yolov8lDetector
        model = Yolov8lDetector()
    elif model_name == "ssd":
        from models.SSD import SSDDetector
        model = SSDDetector()
    elif model_name == "fastrcnn":
        from models.FastRCNN import FastRCNNDetector
        model = FastRCNNDetector()
    else:
        supported_models = ["yolov8n", "yolov11n",
                            "yolov5n", "rtdetr", "yolov8l"]
        raise ValueError(
            f"モデル '{model_name}' はサポートされていません。\n"
            f"サポートされているモデル: {', '.join(supported_models)}"
        )

    cwd = Path(__file__).parent
    inputBaseDir = cwd.parent / "output" / "image"
    cameras = [
        "front",
        # "left_1",
        # "right_1",
        # "left_2",
        # "right_2"
    ]

    logger = getLogger('ultralytics')
    logger.disabled = True

    print("map: ", mapName)
    print("model: ", model_name)
    print(
        f"Input images directory: {inputBaseDir}/{mapName}/original/{cameras[0]}")

    start = time.time()
    for camera in cameras:
        inputImageDirectory = os.path.join(
            inputBaseDir, mapName, "original", camera)
        if not os.path.exists(inputImageDirectory):
            raise RuntimeError(
                f"Input directory does not exist: {inputImageDirectory}")
        imageFiles = os.listdir(inputImageDirectory)
        for imageFile in tqdm.tqdm(imageFiles):
            imagePath = os.path.join(inputImageDirectory, imageFile)
            if not os.path.exists(imagePath):
                raise FileExistsError(
                    f"Input image does not exist: {imagePath}")
            bboxes = model.predict(imagePath)
            index = imageFile.split('.')[0]
            model.save_result(
                imagePath, bboxes, mapName, camera, index, model_name
            )
    end = time.time()
    print(f"total object detection time: {end - start:.2f} seconds")
    print("All images processed.")
