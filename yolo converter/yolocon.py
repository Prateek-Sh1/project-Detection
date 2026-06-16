import os
import xml.etree.ElementTree as ET

# classes (EDIT THIS)
classes = ["helmet", "head", "person"]

image_dir = "archive/images"
xml_dir = "archive/annotations"
label_dir = "archive/labels"

os.makedirs(label_dir, exist_ok=True)

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    xmin = min(box[0], box[1])
    xmax = max(box[0], box[1])
    ymin = min(box[2], box[3])
    ymax = max(box[2], box[3])

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh

    return x_center, y_center, w, h


for file in os.listdir(xml_dir):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w_img = int(size.find("width").text)
    h_img = int(size.find("height").text)

    txt_path = os.path.join(label_dir, file.replace(".xml", ".txt"))
    f = open(txt_path, "w")

    for obj in root.iter("object"):
        name = obj.find("name").text

        if name not in classes:
            continue

        cls_id = classes.index(name)

        bndbox = obj.find("bndbox")

        box = (
            float(bndbox.find("xmin").text),
            float(bndbox.find("xmax").text),
            float(bndbox.find("ymin").text),
            float(bndbox.find("ymax").text),
        )

        yolo_box = convert((w_img, h_img), box)

        f.write(f"{cls_id} " + " ".join(map(str, yolo_box)) + "\n")

    f.close()

print("✅ YOLO conversion done!")