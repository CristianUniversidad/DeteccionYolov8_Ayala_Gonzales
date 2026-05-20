from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO("runs/detect/mi_entrenamiento/weights/best.pt")

# Probar una imagen
results = model("Frames/180.png", show=True)

# Guardar resultado
results[0].save(filename="resultado.png")