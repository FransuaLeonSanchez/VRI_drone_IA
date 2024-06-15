import cv2  # OpenCV para el procesamiento de imágenes
from ultralytics import YOLO  # Importa la clase YOLO de Ultralytics

# Crea una instancia del modelo YOLO con los pesos preentrenados 'yolov8n.pt'
model = YOLO("yolov8s.pt")

# Ejecuta el modelo YOLO en el video de la cámara en vivo
# - 'source=0' indica que la fuente de video es la cámara 0 (cámara predeterminada)
# - 'show=True' muestra la salida del modelo en una ventana de visualización
# - 'stream=True' mantiene la transmisión de video en tiempo real
# - 'conf=0.5' establece el umbral de confianza para las detecciones en 0.5
for results in model(source=0, show=True, stream=True, conf=0.5):
    #Tambien se puede reemplazar con "model.track(source=0, show=True)"

    if cv2.waitKey(30) == 27:
        #Configura la tecla 'Esc' para cerrar la ventana de visualización
        break
