import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# === Update these paths ===
images_dir = r"D:\newmodel\images"
labels_dir = r"D:\newmodel\annotations_yolo"

# Output folders
output_base = r"D:\newmodel"
train_images_dir = os.path.join(output_base, "train", "images")
train_labels_dir = os.path.join(output_base, "train", "labels")
val_images_dir = os.path.join(output_base, "val", "images")
val_labels_dir = os.path.join(output_base, "val", "labels")

# Make output dirs
for folder in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Train/Val split
train_ratio = 0.8
split_index = int(len(all_images) * train_ratio)
train_files = all_images[:split_index]
val_files = all_images[split_index:]

def copy_files(file_list, dest_img_dir, dest_lbl_dir):
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(dest_img_dir, img_file))
        src_label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, os.path.join(dest_lbl_dir, label_file))
        else:
            print(f"⚠️ Label not found for {img_file}")

copy_files(train_files, train_images_dir, train_labels_dir)
copy_files(val_files, val_images_dir, val_labels_dir)

print("✅ Dataset split completed!")
