import os
import random
import shutil

# Source dataset path (all category folders are here)
SRC_DIR = 'data/Plant_leave_diseases_dataset_with_augmentation'
# Target paths
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
# Split ratio
SPLIT_RATIO = 0.8

random.seed(42)  # Ensure reproducibility

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_images:
        src_img = os.path.join(class_path, img)
        dst_img = os.path.join(train_class_dir, img)
        shutil.copy2(src_img, dst_img)
    for img in val_images:
        src_img = os.path.join(class_path, img)
        dst_img = os.path.join(val_class_dir, img)
        shutil.copy2(src_img, dst_img)

print('Dataset splitting completed!') 