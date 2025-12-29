import os
from logging import getLogger
from pathlib import Path

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
    map_name = args.map
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

    from pathlib import Path
    cwd = Path(__file__).parent
    input_base_dir = cwd.parent / "output"
    cameras = [
        "front",
        # "left_1",
        # "right_1",
        # "left_2",
        # "right_2"
    ]
    import time
    import tqdm
    logger = getLogger('ultralytics')
    logger.disabled = True
    print("map: ", map_name)
    print("model: ", model_name)
    print(
        f"Input images directory: {input_base_dir}/{map_name}/original/{cameras[0]}")
    start = time.time()
    for camera in cameras:
        input_images_directory = os.path.join(
            input_base_dir, map_name, "original", camera)
        if not os.path.exists(input_images_directory):
            print(
                f"Input directory does not exist: {input_images_directory}")
            continue
        image_files = os.listdir(input_images_directory)
        for image_file in tqdm.tqdm(image_files):
            image_path = os.path.join(input_images_directory, image_file)
            if image_path is None:
                print(f"Could not read image: {image_path}")
                continue
            bboxes = model.predict(image_path)
            index = image_file.split('.')[0]
            model.save_result(
                image_path, bboxes, map_name, camera, index, model_name
            )
            # print(f"Processed {image_file} for camera {camera}")
    end = time.time()
    print(f"total object detection time: {end - start:.2f} seconds")
    print("All images processed.")
