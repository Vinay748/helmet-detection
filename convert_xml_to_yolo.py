import os
import xml.etree.ElementTree as ET

# CONFIGURATION
xml_folder = "annotations"               # Your original XML files
output_folder = "annotations_yolo"       # Where YOLO .txt files go
classes = ["helmet", "head"]             # Define your classes

os.makedirs(output_folder, exist_ok=True)

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text

        if name not in classes:
            continue

        class_id = classes.index(name)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write to .txt
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    txt_path = os.path.join(output_folder, txt_filename)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))
