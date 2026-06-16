import os
import shutil
from sklearn.model_selection import train_test_split

base_path = "archive"

images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

train_img = os.path.join(images_path, "train")
val_img = os.path.join(images_path, "val")

train_lbl = os.path.join(labels_path, "train")
val_lbl = os.path.join(labels_path, "val")

for p in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(p, exist_ok=True)

images = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png", ".jpeg"))]

train_files, val_files = train_test_split(images, test_size=0.2, random_state=42)

def move(files, img_dest, lbl_dest):
    for f in files:
        label = f.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy(os.path.join(images_path, f), os.path.join(img_dest, f))
        shutil.copy(os.path.join(labels_path, label), os.path.join(lbl_dest, label))

move(train_files, train_img, train_lbl)
move(val_files, val_img, val_lbl)

print("✅ Dataset split completed")