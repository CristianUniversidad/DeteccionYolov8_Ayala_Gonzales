import os

folders = [
    "dataset/labels/train",
    "dataset/labels/val"
]

for folder in folders:

    for file in os.listdir(folder):

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            nuevas_lineas = []

            with open(path, "r") as f:
                lineas = f.readlines()

            for linea in lineas:

                partes = linea.strip().split()

                if len(partes) >= 5:

                    # FORZAR clase 0
                    partes[0] = "0"

                    nuevas_lineas.append(" ".join(partes))

            with open(path, "w") as f:
                f.write("\n".join(nuevas_lineas))

print("Todos los labels corregidos.")