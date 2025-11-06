import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import dataset
from typing import List, Dict
from utils import utils


def analyze_instance_threshold_sensitivity(gt_dir: str, det_dirs: List[str], thresholds: np.ndarray, iou_th: float) -> Dict[str, np.ndarray[float]]:
    # results = {
    #     'covod': list(),
    #     'cerod': list(),
    # }
    results = {
        'precision': list(),
        'recall': list(),
        'f1_score': list(),
        'num_inference': list()
    }

    for instance_th in thresholds:
        datasets = dataset.Dataset(
            gt_dir=gt_dir,
            det_dirs=det_dirs,
            iou_th=iou_th,
            adaptive=True,
            instance_threshold=instance_th,
            confidence_threshold=0
        )
        controller = datasets.controller
        controller.select_rule(controller.instance_rule)
        metrics = datasets.metrics
        # metrics.add_metric('covod', func=metrics.covod)
        # metrics.add_metric('cerod', func=metrics.cerod)
        metrics.add_metric("precision", func=metrics.precision)
        metrics.add_metric("recall", func=metrics.recall)
        metrics.add_metric("f1_score", func=metrics.f1_score)

        for frame_idx, gt, dets in datasets.loader.iter_frame():
            mode = datasets.controller.control_mode(dets)
            integrated_det = datasets.integrator.integrate_detections(
                dets=dets, mode=mode)
            analyzed = datasets.analyzer.analyze_frame(
                gt=gt, dets=integrated_det, mode=mode)
            # analyzed = datasets.analyzer.analyze_frame(
            # gt=gt, dets=dets, mode=mode)
            metrics.update_counters(analyzed_frame=analyzed, mode=mode)
        res = metrics.compute()
        for metric_name, value in res.items():
            results[metric_name].append(value)
        results["num_inference"].append(metrics.get_num_inference())

    for metric_name, values in results.items():
        results[metric_name] = np.array(values)

    return results


def plot_instance_threshold_sensitivity(thresholds: np.ndarray[int], results: Dict[str, np.ndarray[float]]) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), height_ratios=[1, 1])

    for metric_name, values in results.items():
        if metric_name != 'num_inference':
            ax1.plot(thresholds, values, label=metric_name, marker='o')

    ax1.set_xlabel('instance threshold')
    ax1.set_ylabel('score')
    ax1.set_title('Metrics by instance threshold')
    ax1.legend()

    ax2.plot(thresholds, results['num_inference'], label='number of inference')
    ax2.set_xlabel('instance threshold')
    ax2.set_ylabel('number of inferences')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def analyze_confidence_threshold_sensitivity(gt_dir: str, det_dirs: List[str], thresholds: np.ndarray, iou_th: float) -> Dict[str, np.ndarray[float]]:
    # results = {
    #     'covod': list(),
    #     'cerod': list(),
    # }
    results = {
        'precision': list(),
        'recall': list(),
        'f1_score': list(),
        'num_inference': list()
    }

    for conf_th in thresholds:
        datasets = dataset.Dataset(
            gt_dir=gt_dir,
            det_dirs=det_dirs,
            iou_th=iou_th,
            adaptive=True,
            instance_threshold=0,
            confidence_threshold=conf_th)
        controller = datasets.controller
        controller.select_rule(controller.confidence_rule)
        metrics = datasets.metrics
        # metrics.add_metric('covod', func=metrics.covod)
        # metrics.add_metric('cerod', func=metrics.cerod)
        metrics.add_metric("precision", func=metrics.precision)
        metrics.add_metric("recall", func=metrics.recall)
        metrics.add_metric("f1_score", func=metrics.f1_score)

        for frame_idx, gt, dets in datasets.loader.iter_frame():
            mode = datasets.controller.control_mode(dets)
            integrated_det = datasets.integrator.integrate_detections(
                dets=dets, mode=mode)
            analyzed = datasets.analyzer.analyze_frame(
                gt=gt, dets=integrated_det, mode=mode)
            # analyzed = datasets.analyzer.analyze_frame(
            # gt=gt, dets=dets, mode=mode)
            metrics.update_counters(analyzed_frame=analyzed, mode=mode)
        res = metrics.compute()
        for metric_name, value in res.items():
            results[metric_name].append(value)
        results["num_inference"].append(metrics.get_num_inference())

    for metric_name, values in results.items():
        results[metric_name] = np.array(values)

    return results


def polt_confidence_threshold_sensitivity(thresholds: np.ndarray, results: Dict[str, np.ndarray[float]]) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), height_ratios=[1, 1])

    for metric_name, values in results.items():
        if metric_name != 'num_inference':
            ax1.plot(thresholds, values, label=metric_name, marker='o')

    ax1.set_xlabel('confidence threshold')
    ax1.set_ylabel('score')
    ax1.set_title('Metrics by confidence threshold')
    ax1.legend()

    ax2.plot(thresholds, results['num_inference'], label='number of inference')
    ax2.set_xlabel('confidence threshold')
    ax2.set_ylabel('number of inferences')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
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
    args = argparser.parse_args()
    models = args.models
    gt_directory = f"C:/CARLA_Latest/WindowsNoEditor/output/label/{args.map}/front"
    det_directories = [
        f"C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/output/{args.map}/labels/{model}/front" for model in models
    ]

    # thresholds = np.arange(0, 11, 1)
    # results = analyze_instance_threshold_sensitivity(
    #     gt_dir=gt_directory, det_dirs=det_directories, thresholds=thresholds, iou_th=args.iou_th)
    # plot_instance_threshold_sensitivity(thresholds=thresholds, results=results)

    thresholds = np.arange(0.2, 1.1, 0.1)
    results = analyze_confidence_threshold_sensitivity(
        gt_dir=gt_directory, det_dirs=det_directories, thresholds=thresholds, iou_th=args.iou_th)
    polt_confidence_threshold_sensitivity(
        thresholds=thresholds, results=results)
