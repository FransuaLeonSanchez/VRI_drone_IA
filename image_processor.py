from ultralytics import YOLO
import cv2
import os

# Cargar el modelo YOLOv8
model = YOLO("yolov8x.pt")  # Puedes cambiar "yolov8n.pt" por otro modelo YOLOv8

# Ruta de la carpeta de imágenes de entrada
input_folder = "imagenes"

# Ruta de la carpeta de imágenes de salida
output_folder = "resultados"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Procesar todas las imágenes en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Ruta de la imagen de entrada
        image_path = os.path.join(input_folder, filename)

        # Leer la imagen
        image = cv2.imread(image_path)

        # Realizar la inferencia
        results = model(image)

        # Procesar la imagen con los resultados
        for result in results:
            boxes = result.boxes.xyxy  # Coordenadas de los cuadros delimitadores
            confs = result.boxes.conf  # Confianza de las detecciones
            labels = result.boxes.cls  # Clases de las detecciones

            for box, conf, label in zip(boxes, confs, labels):
                # Filtrar solo las detecciones de personas (clase 0)
                if int(label) == 0:
                    # Convertir las coordenadas a enteros
                    x1, y1, x2, y2 = map(int, box)

                    # Dibujar el cuadro delimitador
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ruta de la imagen de salida
        output_path = os.path.join(output_folder, filename)

        # Guardar la imagen de salida
        cv2.imwrite(output_path, image)