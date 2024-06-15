# Test de Modelo YOLO con Roboflow

Este script utiliza el framework YOLO de Ultralytics para la detección de objetos en imágenes o video en tiempo real. Puedes cambiar el modelo utilizado por el script modificando la línea de código:

```python
model = YOLO("yolov8s.pt")
```

Puedes reemplazar `"yolov8s.pt"` con otros modelos disponibles en el repositorio de GitHub de Ultralytics, como:

- `yolov8n.pt`
- `yolov8m.pt`
- `yolov8l.pt`
- `yolov8x.pt`

Para obtener más detalles sobre los modelos disponibles y sus características, consulta la documentación en el [repositorio de Ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file#models).


## Descripción de Scripts

- `test_YOLOv8.py`

Este script es un test para YOLOv8 que permite probar su funcionalidad sin Roboflow, utilizando únicamente las librerías de Ultralytics.

- `Native_Tracking.py`

Este script utiliza el tracking nativo de YOLOv8 y Roboflow para encerrar los objetos dentro de cuadros. En cambio, `test_YOLOv8.py` realiza esta tarea de manera nativa, sin necesidad de Roboflow.
