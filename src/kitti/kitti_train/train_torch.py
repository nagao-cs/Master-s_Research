import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    ssd300_vgg16,
)

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSDClassificationHead

def build_ssd(num_classes):
    model = ssd300_vgg16(weights="DEFAULT")
    out_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        out_channels,
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
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
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

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)
    model.train()
    return metric.compute()

if __name__ == '__main__':
    from pathlib import Path
    cwd = Path(__file__).parent
    from dataset import KITTIDataset, train_transforms, val_transforms

    dataset_dir = Path("/mnt/d/kitti/detection") 
    print(dataset_dir)

    train_dataset = KITTIDataset(
        dataset_dir / "images/train_split",
        dataset_dir / "labels/train_torch",
        transforms=train_transforms,
    )
    val_dataset = KITTIDataset(
        dataset_dir / "images/val",
        dataset_dir / "labels/val_torch",
        transforms=val_transforms,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0,
    )

    import argparse, csv
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--epochs", type=int, required=True)
    args = argparser.parse_args()

    model = build_model(model_name=args.model, num_classes=9)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,   
        momentum=0.9,
        weight_decay=0.0005,
    )
    device = "cuda"
    model.to(device)

    warmup_iters = min(1000, len(loader) - 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / 1000, total_iters=warmup_iters
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    epochs = args.epochs
    best_map = 0.0

    Path("torch_weights").mkdir(exist_ok=True)
    log_path = Path("logs") / f"{args.model}_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_map", "val_map_50", "lr"])

    
    from tqdm import tqdm
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        scaler = torch.amp.GradScaler("cuda")
        for images, targets in tqdm(loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item()}")
                continue
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch == 0:
                warmup_scheduler.step()
            total_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = total_loss / len(loader)

        val_result = evaluate(model, val_loader, device)
        val_map = val_result["map"].item()
        val_map_50 = val_result["map_50"].item()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f} mAP={val_map:.4f} mAP@50={val_map_50:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_loss, val_map, val_map_50, current_lr])

        torch.save(model.state_dict(), f"torch_weights/{args.model}_last.pth")

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), f"torch_weights/{args.model}_best.pth")
            print(f"Best model updated (mAP={best_map:.4f})")