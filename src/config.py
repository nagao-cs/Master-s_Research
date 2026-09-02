import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KITTI_ROOT = "/mnt/d/kitti"      # Dドライブのデータセット
DEFAULT_KITTI_WEIGHTS_DIR = PROJECT_ROOT / "src" / "kitti" / "kitti_train" / "torch_weights"
KITTI_WEIGHTS_DIR = os.environ.get("KITTI_WEIGHTS_DIR", str(DEFAULT_KITTI_WEIGHTS_DIR.resolve()) + "/")
KITTI_ULTRALYTICS_WEIGHTS_DIR = PROJECT_ROOT / "src/kitti/kitti_train/runs/train"
KITTI_NUM_CLASS = 9
KITTI_IMAGE_WIDTH = 1242
KITTI_IMAGE_HEIGHT = 375

DATASET_DIR = Path("/mnt/d/dataset")
RESULT_DIR = Path("/mnt/d/result")

IOU_THRESHOLD = 0.5