import os
import random
import shutil
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN
# ============================================================

FRAMES_DIR = "Frames"

# Nombre de clases
CLASSES = ["objeto"]

# Porcentaje train/val
TRAIN_PERCENT = 0.8

# ============================================================
# CREAR ESTRUCTURA DEL DATASET
# ============================================================

dataset_dirs = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

for d in dataset_dirs:
    os.makedirs(d, exist_ok=True)

# ============================================================
# OBTENER IMÁGENES
# ============================================================

images = [f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")]
images.sort()

random.shuffle(images)

split_index = int(len(images) * TRAIN_PERCENT)

train_images = images[:split_index]
val_images = images[split_index:]

# ============================================================
# FUNCIÓN PARA COPIAR ARCHIVOS
# ============================================================

def copy_files(image_list, img_dest, label_dest):
    for img_name in image_list:

        img_path = os.path.join(FRAMES_DIR, img_name)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(FRAMES_DIR, label_name)

        shutil.copy(img_path, os.path.join(img_dest, img_name))

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(label_dest, label_name))

# Copiar train
copy_files(
    train_images,
    "dataset/images/train",
    "dataset/labels/train"
)

# Copiar val
copy_files(
    val_images,
    "dataset/images/val",
    "dataset/labels/val"
)

print("Dataset organizado correctamente.")

# ============================================================
# CREAR ARCHIVO data.yaml
# ============================================================

yaml_content = f"""
path: dataset

train: images/train
val: images/val

names:
"""

for i, cls in enumerate(CLASSES):
    yaml_content += f"  {i}: {cls}\n"

with open("dataset/data.yaml", "w") as f:
    f.write(yaml_content)

print("Archivo data.yaml creado.")

model = YOLO("yolov8n.pt")

# ============================================================
# ENTRENAMIENTO
# ============================================================

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,        # GPU NVIDIA
    workers=0,
    name="mi_entrenamiento"
)

# ============================================================
# VALIDACIÓN
# ============================================================

metrics = model.val()

print(metrics)

# ============================================================
# EXPORTAR MODELO
# ============================================================

model.export(format="onnx")

print("Entrenamiento finalizado.")