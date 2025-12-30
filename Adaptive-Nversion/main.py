from integrator import MajorityIntegrator
from VersionController import VersionController, VersionState
from NversionExecutor import NversionExecutor
from typing import List
import os
from pathlib import Path
import time
import os
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Adaptive Object Detection"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        choices=["yolov8n", "yolov5n", "yolov11n",
                 "rtdetr", "ssd", "yolov8l", "fastrcnn"],
        nargs='+',
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )

    args = argparser.parse_args()
    print(args)
    model_names = args.models
    n_model = len(model_names)
    map_name = args.map

    model_list = list()

    for model_name in model_names:
        if model_name == "yolov8n":
            from ObjectDetection.models.Yolov8n import Yolov8nDetector
            model = Yolov8nDetector()
        elif model_name == "yolov11n":
            from ObjectDetection.models.Yolov11n import Yolov11nDetector
            model = Yolov11nDetector()
        elif model_name == "yolov5n":
            from ObjectDetection.models.Yolov5n import Yolov5nDetector
            model = Yolov5nDetector()
        elif model_name == "rtdetr":
            from ObjectDetection.models.rtDETR import RTDETRDetector
            model = RTDETRDetector()
        elif model_name == 'yolov8l':
            from ObjectDetection.models.yolov8l import Yolov8lDetector
            model = Yolov8lDetector()
        elif model_name == "ssd":
            from ObjectDetection.models.SSD import SSDDetector
            model = SSDDetector()
        elif model_name == "fastrcnn":
            from ObjectDetection.models.FastRCNN import FastRCNNDetector
            model = FastRCNNDetector()
        else:
            raise ValueError(
                f"モデル '{model_name}' はサポートされていません。\n"
            )
        model_list.append(model)

    cwd = Path(__file__).parent

    input_image_dir = cwd.parent / "output" / "image" / \
        f"{map_name}" / "original" / "front"

    n_inference = 0
    output_label_list = list()

    print("map: ", map_name)
    print("models: ", model_list)

    from logging import getLogger
    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(input_image_dir):
        print(
            f"Input directory does not exist: {input_image_dir}")
        exit(1)

    file_list = os.listdir(input_image_dir)

    CONF_THRESHOLD = 0.5
    AGREEMENT_THRESHOLD = 0.8

    numVersion = len(model_list)

    integrator = MajorityIntegrator(iou_th=0.5, maxVersion=numVersion)
    executor = NversionExecutor(model_list, integrator)
    controller = VersionController(
        conf_threshold=CONF_THRESHOLD, agreement_threshold=AGREEMENT_THRESHOLD, maxVersion=numVersion)

    # 計測開始
    start = time.time()
    for image_file in tqdm(file_list):
        image_path = os.path.join(input_image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Could not read image: {image_path}")
            exit(1)

        state = controller.state
        if state == VersionState.ONE:
            base_detection = executor.execute_1version(image_path)
            controller.update_state(detections=base_detection)

            if state == VersionState.N:
                final_detections = executor.execute_N_1version(
                    image_path, base_detection)
                n_inference += len(model_list)
            else:
                final_detections = base_detection
                n_inference += 1
        else:
            detections, detection_by_model = executor.execute_Nversion(
                image_path)
            controller.update_state(detection_dict=detection_by_model)
            final_detections = detections
            n_inference += len(model_list)

        output_label_list.append(final_detections)

    # 計測終了
    end = time.time()
    print(f"total object detection time: {end - start:.2f} seconds")

    output_label_dir = cwd / "output" / "labels" / \
        f"{map_name}" / f"{'_'.join(model_names)}"
    os.makedirs(output_label_dir, exist_ok=True)

    index = 0
    for output_label in output_label_list:
        output_label_path = os.path.join(output_label_dir, f"{index:6f}.txt")
        with open(output_label_path, 'w') as f:
            for bbox in output_label:
                x_center = bbox['x_center']
                y_center = bbox['y_center']
                width = bbox['width']
                height = bbox['height']
                class_id = bbox['class_id']
                conf = bbox['confidence']
                f.write(
                    f"{class_id} {x_center} {y_center} {width} {height} {conf}\n")
        index += 1
    print(f"Total inferences made: {n_inference}")
    elapsed = end - start
    print("total time:", elapsed)
