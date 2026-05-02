from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from app.services.utils import Point, BoundingBox
from settings import settings

_yolo_model = None

def get_model():
  global _yolo_model
  if _yolo_model is None:
    _yolo_model = YOLO("models/yolov11l-face.pt") 
  return _yolo_model

def detect_faces_yolo11(image_path):
  model = get_model()
  result = model.predict(source=image_path, verbose=False, imgsz=settings.img_sz, conf=settings.face_min_confidence, show=False)
  detections = []
  for box in result[0].boxes:
    class_id = int(box.cls[0])
    class_name = _yolo_model.names[class_id]
    confidence = box.conf[0]

    if class_name.upper() == 'FACE':
      coords = box.xyxy.cpu().numpy().astype(int)
      x1, y1, x2, y2 = coords[0]
      start = Point(x=x1, y=y1)
      end = Point(x=x2, y=y2)
      box = BoundingBox(start_point=start, end_point=end)
      detections.append(box)
  return detections
