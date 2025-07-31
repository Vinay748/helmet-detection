import os
import shutil

# Base paths
base_path = r"D:\newmodel"
images_train = os.path.join(base_path, "images/train")
images_val = os.path.join(base_path, "images/val")

labels_train = os.path.join(base_path, "labels/train")
labels_val = os.path.join(base_path, "labels/val")

annotations_path = os.path.join(base_path, "annotations_yolo")

# Make sure label folders exist
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

def copy_labels(images_folder, labels_folder):
    for img_file in os.listdir(images_folder):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(annotations_path, label_file)
            dst_label = os.path.join(labels_folder, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
                print(f"Copied {label_file} to {labels_folder}")
            else:
                print(f"Label for {img_file} not found in annotations_yolo")

print("Copying train labels...")
copy_labels(images_train, labels_train)

print("Copying val labels...")
copy_labels(images_val, labels_val)

print("Done.")
