import os

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Evaluate object detection results")
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03",
                 "Town04", "Town05", "Town10HD_Opt"],
        help="Map name: Town01, Town02, Town04, Town05, Town10HD",
        required=True
    )
    argparser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr"],
    )
    argparser.add_argument(
        "--iou_th",
        type=float,
        default=0.5,
        help="IOU threshold for evaluation",
    )
    argparser.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Enable adaptive evaluation mode",
    )
    argparser.add_argument(
        "--instance_threshold",
        type=int,
        default=5,
        help="Instance threshold for adaptive mode",
    )
    argparser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for adaptive mode",
    )
    args = argparser.parse_args()
    models = args.models
    models.sort()
    gt_directory = f"C:/CARLA_Latest/WindowsNoEditor/output/label/{args.map}/front"
    det_directories = [
        f"C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{args.map}/labels/{model}/front" for model in models
    ]

    from dataset import dataset
    datasets = dataset.Dataset(
        gt_dir=gt_directory,
        det_dirs=det_directories,
        iou_th=args.iou_th,
        adaptive=args.adaptive,
        instance_threshold=args.instance_threshold,
        confidence_threshold=args.confidence_threshold
    )
    controller = datasets.controller
    stats = datasets.stats
    stats.add_metric("covod", func=stats.covod)
    stats.add_metric("cerod", func=stats.cerod)
    for frame_idx, gt, dets in datasets.loader.iter_frame():
        mode = controller.control_mode(gt, dets)
        analyzed = datasets.analyzer.analyze_frame(gt, dets, mode=mode)
        stats.update_counters(analyzed)
    results = stats.compute()
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.4f}")
