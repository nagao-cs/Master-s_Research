import shutil
import os
import glob
import random
imgs = sorted(glob.glob("./images/*.png"))
random.shuffle(imgs)
print(f"Total images: {len(imgs)}")
split_idx = int(len(imgs)*0.8)
train_imgs = imgs[:split_idx]
val_imgs = imgs[split_idx:]
os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)

for img_path in train_imgs:
    base = os.path.basename(img_path)
    name, _ = os.path.splitext(base)
    shutil.copy(img_path, f"dataset/images/train/{base}")
    shutil.copy(f"./labels/{name}.txt",
                f"dataset/labels/train/{name}.txt")

for img_path in val_imgs:
    base = os.path.basename(img_path)
    name, _ = os.path.splitext(base)
    shutil.copy(img_path, f"dataset/images/val/{base}")
    shutil.copy(f"./labels/{name}.txt",
                f"dataset/labels/val/{name}.txt")
