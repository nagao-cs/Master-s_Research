from . import detection_logic
import sys
sys.path.append("..")


def save_stats(models, n_inference, time_elapsed, rule, threshold):
    import os
    import json
    output_stats_dir = rf"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\adaptive_Nversion_detector\output\stats"
    os.makedirs(output_stats_dir, exist_ok=True)
    model_key = '_'.join(models)
    stats = {
        "models": models,
        "rule": rule,
        "threshold": threshold,
        "n_inference": n_inference,
        "time_elapsed": time_elapsed,
        "throughput": 2000 / time_elapsed if time_elapsed > 0 else 0.0
    }
    output_stats_path = os.path.join(
        output_stats_dir, f"{model_key}_stats.json")
    with open(output_stats_path, 'w') as f:
        json.dump(stats, f, indent=4)


def draw_bbox(image, bboxes):
    import cv2
    im_width = image.shape[1]
    im_height = image.shape[0]
    for bbox in bboxes:
        x_center = bbox['x_center'] * im_width
        y_center = bbox['y_center'] * im_height
        width = bbox['width'] * im_width
        height = bbox['height'] * im_height
        xmin = int(x_center - width / 2)
        xmax = int(x_center + width / 2)
        ymin = int(y_center - height / 2)
        ymax = int(y_center + height / 2)
        label = bbox['label']
        conf = bbox['confidence']
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Adaptive Object Detection"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        choices=["yolov8n", "yolov5n", "yolov11n", "rtdetr", "ssd", "yolov8l"],
        nargs='+',
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )
    argparser.add_argument(
        "--rule",
        type=str,
        choices=["n_det", "min_conf"],
        default="min_conf",
        required=True
    )
    argparser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        required=True
    )
    args = argparser.parse_args()
    print(args)
    model_names = args.models
    n_model = len(model_names)
    map_name = args.map
    rule = args.rule
    threshold = args.threshold

    model_list = list()
    from models.Yolov8n import Yolov8nDetector
    from models.Yolov11 import Yolov11nDetector
    from models.Yolov5 import Yolov5nDetector
    from models.rtDETR import RTDETRDetector
    from models.yolov8l import Yolov8lDetector
    for model_name in model_names:
        match model_name:
            case "yolov8n":
                model = Yolov8nDetector()
            case "yolov11n":
                model = Yolov11nDetector()
            case "yolov5n":
                model = Yolov5nDetector()
            case "rtdetr":
                model = RTDETRDetector()
            case 'yolov8l':
                model = Yolov8lDetector()
            case _:
                supported_models = ["yolov8n", "yolov11n",
                                    "yolov5n", "rtdetr", "yolov8l"]
                raise ValueError(
                    f"モデル '{model_name}' はサポートされていません。\n"
                    f"サポートされているモデル: {', '.join(supported_models)}"
                )
        model_list.append(model)
    input_image_dir = rf"C:\CARLA_Latest\WindowsNoEditor\output\image\{map_name}\original\front"
    n_inference = 0
    output_label_list = list()
    import time
    import os
    print("map: ", map_name)
    print("model: ", model_name)
    from logging import getLogger
    logger = getLogger('ultralytics')
    logger.disabled = True
    if not os.path.exists(input_image_dir):
        print(
            f"Input directory does not exist: {input_image_dir}")
        exit(1)

    from tqdm import tqdm
    file_list = os.listdir(input_image_dir)
    # 計測開始
    start = time.time()

    for image_file in tqdm(file_list):
        image_path = os.path.join(input_image_dir, image_file)
        if image_path is None:
            print(f"Could not read image: {image_path}")
            continue
        base_bboxes = model_list[0].predict(image_path)
        if n_model > 1 and detection_logic.check_switch_to_Nversion(base_bboxes, rule, threshold):
            all_detections = [base_bboxes]
            for model in model_list[1:]:
                bboxes = model.predict(image_path)
                all_detections.append(bboxes)
            integrated_bboxes = detection_logic.integrate_N_detections(
                all_detections
            )
            final_bboxes = integrated_bboxes
            n_inference += len(model_list)
        else:
            final_bboxes = base_bboxes
            n_inference += 1
        output_label_list.append(final_bboxes)

    # 計測終了
    end = time.time()
    print(f"total object detection time: {end - start:.2f} seconds")

    import cv2
    output_image_dir = rf"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\adaptive_Nversion_detector\output\images\{map_name}\{'_'.join(model_names)}"
    output_label_dir = rf"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\adaptive_Nversion_detector\output\labels\{map_name}\{'_'.join(model_names)}"
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    for image_file, output_label in zip(os.listdir(input_image_dir), output_label_list):
        image_path = os.path.join(input_image_dir, image_file)
        if image_path is None:
            print(f"Could not read image: {image_path}")
            continue
        index = image_file.split('.')[0]
        image = cv2.imread(image_path)
        bbox_image = draw_bbox(image, output_label)
        output_image_path = os.path.join(output_image_dir, f"{index}.png")
        cv2.imwrite(output_image_path, bbox_image)

        output_label_path = os.path.join(output_label_dir, f"{index}.txt")
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
    print(f"Total inferences made: {n_inference}")
    print("All images processed.")
    save_stats(model_names, n_inference, end - start, rule, threshold)
