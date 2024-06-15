import cv2  # Importa la librería OpenCV para el procesamiento de imágenes
from ultralytics import YOLO  # Importa la clase YOLO de Ultralytics
import supervision as sv  # Importa el módulo de supervisión

def main():
    # Crea un objeto BoxAnnotator para añadir anotaciones a las detecciones
    box_annotator = sv.BoxAnnotator(
        thickness=2,  # Grosor del borde del cuadro de detección
        text_thickness=1,  # Grosor del texto de las etiquetas
        text_scale=0.5  # Escala del texto de las etiquetas
    )

    # Crea una instancia del modelo YOLO con los pesos preentrenados 'yolov8n.pt'
    model = YOLO("yolov8s.pt")

    # Bucle principal para procesar el video de la cámara en tiempo real
    for result in model.track(source=0, show=True, stream=True):
        # Obtiene el frame actual del video
        frame = result.orig_img

        # Convierte las detecciones del modelo YOLO a un objeto Detections
        detections = sv.Detections.from_yolov8(result)

        # Asigna los IDs de seguimiento de las detecciones si están disponibles
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Crea una lista de etiquetas para las detecciones
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        # Añade las anotaciones de las detecciones al frame
        frame = box_annotator.annotate(
            scene=frame,  # Escena (frame) donde se añadirán las anotaciones
            detections=detections,  # Detecciones de objetos
            labels=labels  # Etiquetas de las detecciones
        )

        # Muestra el frame con las detecciones en una ventana de OpenCV
        cv2.imshow("Roboflow", frame)

        # Espera a que se presione la tecla 'Esc' para salir del bucle y terminar la ejecución
        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
