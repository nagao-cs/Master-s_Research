from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    ssd300_vgg16,

)
import torch
from torchvision.models.detection import _utils as det_utils

from ...config import KITTI_WEIGHTS_DIR, KITTI_NUM_CLASS, KITTI_ULTRALYTICS_WEIGHTS_DIR

from torchvision.models.detection.ssd import SSDClassificationHead
from .SSD import SSDDetector
def build_kitti_ssd(device):
    model = ssd300_vgg16(weights=None)
    out_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        out_channels,
        num_anchors,
        num_classes=KITTI_NUM_CLASS,
    )
    state_dict = torch.load(
        KITTI_WEIGHTS_DIR + "ssd_best.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)

    return SSDDetector(model)
def build_carla_ssd(device):
    pass
def build_ssd(dataset, device):
    if dataset == "KITTI":
        detector = build_kitti_ssd(device)
    elif dataset == "CARLA":
        detector = build_carla_ssd(device)
    
    return detector

from torchvision.models.detection.fcos import FCOSClassificationHead
from .FCOS import FcosDetector
def build_kitti_fcos(device):
    model = fcos_resnet50_fpn(weights="DEFAULT")

    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = FCOSClassificationHead(
        in_channels,
        num_anchors,
        num_classes=KITTI_NUM_CLASS,
    )
    state_dict = torch.load(
        KITTI_WEIGHTS_DIR + "fcos_best.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)

    return FcosDetector(model)
def build_carla_fcos(device):
    pass
def build_fcos(dataset, device):
    if dataset == "KITTI":
        return build_kitti_fcos(device)
    elif dataset == "CARLA":
        return build_carla_fcos(device)

from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from .retinanet import RetinanetDetector
def build_kitti_retinanet(device):
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels,
        num_anchors,
        num_classes=KITTI_NUM_CLASS,
    )
    state_dict = torch.load(
        KITTI_WEIGHTS_DIR + "retinanet_best.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)

    return RetinanetDetector(model)
def build_carla_retinanet(device):
    pass
def build_retinanet(dataset, device):
    if dataset == "KITTI":
        return build_kitti_retinanet(device)
    elif dataset == "CARLA":
        return build_carla_retinanet(device)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .FastRCNN import FasterRCNNDetector
def build_kitti_fasterrcnn(device):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes=KITTI_NUM_CLASS,
    )

    state_dict = torch.load(
        KITTI_WEIGHTS_DIR + "fasterrcnn_best.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)

    return FasterRCNNDetector(model)
def build_carla_fasterrcnn(device):
    pass
def build_fasterrcnn(dataset, device):
    if dataset == "KITTI":
        return build_kitti_fasterrcnn(device)
    elif dataset == "CARLA":
        return build_carla_fasterrcnn(device)

from .Yolov8n import Yolov8nDetector
def build_kitti_yolov8n(device):
    model = KITTI_ULTRALYTICS_WEIGHTS_DIR / "yolov8n_kitti/weights/best.pt"

    return Yolov8nDetector(model)
def build_carla_yolov8n(device):
    pass
def build_yolov8n(dataset, device):
    if dataset == "KITTI":
        return build_kitti_yolov8n(device)
    elif dataset == "CARLA":
        return build_carla_yolov8n(device)

from .Yolov11n import Yolov11nDetector
def build_kitti_yolov11n(device):
    model = KITTI_ULTRALYTICS_WEIGHTS_DIR / "yolov11n_kitti/weights/best.pt"

    return Yolov11nDetector(model)
def build_carla_yolov11n(device):
    pass
def build_yolov11n(dataset, device):
    if dataset == "KITTI":
        return build_kitti_yolov11n(device)
    elif dataset == "CARLA":
        return build_carla_yolov11n(device)

from .Yolov5n import Yolov5nDetector
def build_kitti_yolov5n(device):
    model = KITTI_ULTRALYTICS_WEIGHTS_DIR / "yolov5n_kitti/weights/best.pt"

    return Yolov5nDetector(model)
def build_carla_yolov5n(device):
    pass
def build_yolov5n(dataset, device):
    if dataset == "KITTI":
        return build_kitti_yolov5n(device)
    elif dataset == "CARLA":
        return build_carla_yolov5n(device)
    
from .rtDETR import RTDETRDetector
def build_kitti_rtdetr(device):
    model = KITTI_ULTRALYTICS_WEIGHTS_DIR / "rtdetr_kitti/weights/best.pt"

    return RTDETRDetector(model)
def build_carla_rtdetr(device):
    pass
def build_rtdetr(dataset, device):
    if dataset == "KITTI":
        return build_kitti_rtdetr(device)
    elif dataset == "CARLA":
        return build_carla_rtdetr(device)
    

from .ObjectDetector import Detector
def build_model(model_name, dataset,device) -> Detector:
    if model_name == "fasterrcnn":
        model =  build_fasterrcnn(dataset, device=device)
    elif model_name == "retinanet":
        model =  build_retinanet(dataset, device=device)
    elif model_name == "fcos":
        model =  build_fcos(dataset, device=device)
    elif model_name == "ssd":
        model =  build_ssd(dataset, device=device)
    elif model_name == "yolov8n":
        model = build_yolov8n(dataset, device)
    elif model_name == "yolov11n":
        model = build_yolov11n(dataset, device)
    elif model_name == "rtdetr":
        model = build_rtdetr(dataset, device)
    elif model_name == "yolov5n":
        model = build_yolov5n(dataset, device)
    else:
        raise ValueError(model_name)

    # if not ("yolo" in model_name or "rtdetr" in model_name):
    #     model.detections_per_img = 50
    #     model.score_thresh = 0.25
    #     model.nms_thresh = 0.3
    if model is None:
        raise ValueError("something incorrect input")
    
    return model