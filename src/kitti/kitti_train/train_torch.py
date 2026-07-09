import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

import torchvision

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    ssd300_vgg16,

)

from torchvision.models.detection.ssd import SSDClassificationHead
def build_ssd(num_classes):

    model = ssd300_vgg16(weights="DEFAULT")

    in_channels = model.backbone.out_channels

    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
    )

    return model

from torchvision.models.detection.fcos import FCOSClassificationHead
def build_fcos(num_classes):

    model = fcos_resnet50_fpn(weights="DEFAULT")

    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = FCOSClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
    )

    return model

from torchvision.models.detection.retinanet import RetinaNetClassificationHead
def build_retinanet(num_classes):

    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")

    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
    )

    return model

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def build_fasterrcnn(num_classes):

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model

def build_model(model_name, num_classes):

    if model_name == "fasterrcnn":
        return build_fasterrcnn(num_classes)

    elif model_name == "retinanet":
        return build_retinanet(num_classes)

    elif model_name == "fcos":
        return build_fcos(num_classes)

    elif model_name == "ssd":
        return build_ssd(num_classes)

    else:
        raise ValueError(model_name)

from pathlib import Path
cwd = Path(__file__)
dataset_dir = cwd.parent.parent.parent.parent.parent.parent.parent / "d/kitti/detection"

from dataset import KITTIDataset
dataset = KITTIDataset(
    dataset_dir / "images/train",
    dataset_dir / "labels/torch",
)

from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)



import torch
import argparse
argparser = argparse.ArgumentParser(
    )
argparser.add_argument(
    "--model",
    type=str,
    required=True,
)
argparser.add_argument(
    "--epochs",
    type=int,
    required=True,
)
args = argparser.parse_args()

model = build_model(model_name=args.model ,num_classes=9)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

device = "cuda"
model.to(device)

epochs = args.epochs
best_loss = float("inf")
from tqdm import tqdm
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images,targets in tqdm(loader):

        images = [img.to(device) for img in images]
        targets = [
            {
                k:v.to(device)
                for k,v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images,targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}: {epoch_loss:.4f}")

    # last
    torch.save(
        model.state_dict(),
        f"torch_weights/{args.model}_last.pth"
    )

    # best
    if epoch_loss < best_loss:
        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            f"torch_weights/{args.model}_best.pth"
        )

        print(f"Best model updated ({best_loss:.4f})")
    