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
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr", "yolov8l"],
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
        default=0,
        help="Instance threshold for adaptive mode",
    )
    argparser.add_argument(
        "--confidence_threshold",
        type=float,
        default=1.0,
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
    for frame_idx, gt, dets in datasets.loader.iter_frame():
        # mode = controller.control_mode(gt, dets)
        mode = 'multi_version'
        analyzed = datasets.analyzer.analyze_frame(gt, dets, mode=mode)
        # comparison_results = datasets.analyzer.compare_detections(
        # gt, dets)
        # stats.analyze_model_specific_errors(comparison_results)

        stats.update(frame_idx, gt, analyzed)
    stats.plot_transition()
    stats.plot_detection_analysis()
    stats.analyze_class_statistics()
