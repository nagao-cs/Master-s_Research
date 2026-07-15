from src.config import KITTI_ROOT
from pathlib import Path
import os
from typing import Iterator


class Dataset:
    def __init__(self, dir_path: Path):
        self.dir_path = dir_path
        self.image_dir = self.dir_path / "images"
        self.label_dir = self.dir_path / "labels"

        self.image_paths = sorted(
            self.image_dir / name
            for name in os.listdir(self.image_dir)
        )
        self.label_paths = sorted(
            self.label_dir / name
            for name in os.listdir(self.label_dir)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[tuple[Path, Path]]:
        return zip(self.image_paths, self.label_paths)

def build_dataset(dataset_name: str, map_name: str):
    if dataset_name == "KITTI":
        dataset_dir = Path(KITTI_ROOT + f"/tracking/{map_name}")
    elif dataset_name == "CARLA":
        dataset_dir = Path("temp")
    print(f"dataset dir: {dataset_dir}")
    
    if not Path.exists(dataset_dir):
        raise ValueError(f"invalid datast: {dataset_name}, {map_name}")

    return Dataset(dir_path=dataset_dir)
    