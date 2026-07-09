from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class KITTIDataset(Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        image = read_image(img_path).float()/255.

        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f:
                cls, xmin, ymin, xmax, ymax = map(float, line.split())
                labels.append(int(cls))

                boxes.append([
                    xmin,
                    ymin,
                    xmax,
                    ymax
                ])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image,target