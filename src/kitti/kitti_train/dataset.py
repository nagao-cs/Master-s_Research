from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# dataset.py に追加
from torchvision.transforms import v2
from torchvision import tv_tensors

train_transforms = v2.Compose([
    v2.RandomPhotometricDistort(p=0.5),
    v2.RandomZoomOut(fill=0, p=0.3),
    v2.RandomIoUCrop(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.SanitizeBoundingBoxes(),  # crop後に潰れたboxを除去
    v2.ToDtype(torch.float32, scale=True),
])
val_transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
])

class KITTIDataset(Dataset):

    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = sorted(self.image_dir.glob("*.png"))
        self.transforms = transforms
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        image = read_image(img_path) 

        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                cls, xmin, ymin, xmax, ymax = map(float, line.split())
                labels.append(int(cls))
                boxes.append([xmin, ymin, xmax, ymax])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_t, format="XYXY", canvas_size=image.shape[-2:]
            ),
            "labels": labels_t,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target