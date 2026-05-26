# src/time_aware_exp/factory/detection_factory.py

from typing import Callable, Dict
from pathlib import Path

# モデルを動的にロードするレジストリ
MODEL_REGISTRY: Dict[str, str] = {
    "yolov8n": "src.ObjectDetection.models.Yolov8n.Yolov8nDetector",
    "yolov8l": "src.ObjectDetection.models.yolov8l.Yolov8lDetector",
    "yolov11n": "src.ObjectDetection.models.Yolov11n.Yolov11nDetector",
    "yolov5n": "src.ObjectDetection.models.Yolov5n.Yolov5nDetector",
    "rtdetr": "src.ObjectDetection.models.rtDETR.RTDETRDetector",
    "yolo26x": "src.ObjectDetection.models.Yolo26x.Yolov26xDetector",
    "ssd": "src.ObjectDetection.models.SSD_torch.SSDDetector",
    "fastrcnn": "src.ObjectDetection.models.FastRCNN.FasterRCNNDetector",
    "fcos": "src.ObjectDetection.models.FCOS.FcosDetector",
    "retinanet": "src.ObjectDetection.models.retinanet.RetinanetDetector",
    "trainYolo": "src.ObjectDetection.models.trainedYolov8n.Yolov8nTrainedDetector",
}

def get_model_class(model_name: str):
    """モデル名からクラスを動的にロード"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"モデル '{model_name}' はサポートされていません。\n"
            f"利用可能: {list(MODEL_REGISTRY.keys())}"
        )
    
    class_path = MODEL_REGISTRY[model_name]
    module_path, class_name = class_path.rsplit(".", 1)
    
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def build_infer_fn(model_name: str) -> Callable:
    """モデル名から推論関数を生成"""
    detector_class = get_model_class(model_name)
    detector = detector_class()
    return detector.predict

def build_model_detection(cfg):
    """ConfigからModelDetectionStrategyを構築"""
    infer_fns = {
        cfg.model_1: build_infer_fn(cfg.model_1),
        cfg.model_2: build_infer_fn(cfg.model_2),
        cfg.model_3: build_infer_fn(cfg.model_3),
    }
    
    from ..strategy.detectionStrategy import ModelDetectionStrategy
    return ModelDetectionStrategy(infer_fns)

def build_single_model(model_name: str):
    """単一モデルを構築（StateContextで使用）"""
    detector_class = get_model_class(model_name)
    return detector_class()