import os
import shutil
import random

RAW_DATASET = "raw_dataset"  # Place your dataset here
OUTPUT_DIR = "dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% test

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data():
    categories = os.listdir(RAW_DATASET)

    for category in categories:
        input_path = os.path.join(RAW_DATASET, category)
        if not os.path.isdir(input_path):
            continue

        images = os.listdir(input_path)
        random.shuffle(images)

        train_size = int(len(images) * TRAIN_SPLIT)
        train_images = images[:train_size]
        test_images = images[train_size:]

        train_dir = os.path.join(OUTPUT_DIR, "train", category)
        test_dir = os.path.join(OUTPUT_DIR, "test", category)
        create_dir(train_dir)
        create_dir(test_dir)

        for img in train_images:
            shutil.copy(os.path.join(input_path, img), os.path.join(train_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(input_path, img), os.path.join(test_dir, img))

        print(f"[✅] {category}: {len(train_images)} train, {len(test_images)} test")

if __name__ == "__main__":
    split_data()
