import cv2  # OpenCV para el procesamiento de imágenes
from ultralytics import YOLO  # Importa la clase YOLO de Ultralytics
import os  # Para manejar la creación de directorios

# Crea una instancia del modelo YOLO con los pesos preentrenados 'yolov8n.pt'
model = YOLO("yolov8s.pt")

# Verifica si la carpeta 'resultados' existe, si no, la crea
output_folder = "resultados"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Variable para contar los fotogramas
frame_count = 0

# Ejecuta el modelo YOLO en el video de la cámara en vivo
# - 'source=0' indica que la fuente de video es la cámara 0 (cámara predeterminada)
# - 'show=True' muestra la salida del modelo en una ventana de visualización
# - 'stream=True' mantiene la transmisión de video en tiempo real
# - 'conf=0.5' establece el umbral de confianza para las detecciones en 0.5
for results in model(source="videos/playa.mp4", show=True, stream=True, conf=0.5):
    # Incrementa el contador de fotogramas
    frame_count += 1

    # Obtiene el fotograma procesado
    frame = results.orig_img

    # Define la ruta y el nombre del archivo para guardar el fotograma
    output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

    # Guarda el fotograma en la carpeta 'resultados'
    cv2.imwrite(output_path, frame)

    if cv2.waitKey(30) == 27:
        # Configura la tecla 'Esc' para cerrar la ventana de visualización
        break

cv2.destroyAllWindows()
