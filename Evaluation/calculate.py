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
        choices=["yolov8n", "yolov11n", "yolov5n", "rtdetr", "yolov8l", "ssd"],
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
    gt_directory = f"C:/CARLA_Latest/WindowsNoEditor/output/label/{args.map}/front"
    det_directories = [
        f"C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{args.map}/labels/{model}/front" for model in models
    ]
    # det_directories = [
    #     rf"C:\CARLA_Latest\WindowsNoEditor\ObjectDetection\adaptive_Nversion_detector\output\labels\{args.map}\True_min_conf_{'_'.join(args.models)}"
    # ]

    from dataset import dataset
    datasets = dataset.Dataset(
        gt_dir=gt_directory,
        det_dirs=det_directories,
        iou_th=args.iou_th,
        adaptive=args.adaptive,
        instance_threshold=args.instance_threshold,
        confidence_threshold=args.confidence_threshold
    )
    print(args)
    controller = datasets.controller
    controller.select_rule(controller.confidence_rule)
    metrics = datasets.metrics
    # metrics.add_metric('covod', func=metrics.covod)
    # metrics.add_metric('cerod', func=metrics.cerod)
    metrics.add_metric("precision", func=metrics.precision)
    metrics.add_metric("recall", func=metrics.recall)
    metrics.add_metric("f1_score", func=metrics.f1_score)
    results = {
        # 'covod': 0,
        # 'cerod': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'num_inference': 0
    }
    for frame_idx, gt, dets in datasets.loader.iter_frame():
        mode = datasets.controller.control_mode(dets)
        integrated_det = datasets.integrator.integrate_detections(
            dets=dets, mode=mode)
        analyzed_integrated = datasets.analyzer._classify(
            gt, integrated_det)
        metrics.update_accuracy_counter(analyze_frame=analyzed_integrated)

        analyzed_frame = datasets.analyzer.analyze_frame(
            gt=gt, dets=dets, mode=mode)
        metrics.update_reliability_counters(
            analyzed_frame=analyzed_frame, mode=mode)
    res = metrics.compute()
    for metric_name, value in res.items():
        results[metric_name] = value
        print(metric_name, value)
    print(metrics.get_num_inference())
    # metrics.precision_recall_by_num_detection()
