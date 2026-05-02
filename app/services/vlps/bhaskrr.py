from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import mediapipe as mp
import cv2
from app.services.utils import Point, BoundingBox 
from settings import settings
import time

_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('models/license_plate_detector.pt')
    return _yolo_model

def detect_vlps_bhaskrr(image_path):
  model = get_model()
  results = model.predict(image_path, imgsz=settings.img_sz, conf=settings.plate_min_confidence, verbose=False)
  detections = []
  for r in results:
    if r.boxes is None:
        continue
    for b in r.boxes:
        coords = b.xyxy.cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords[0]
        start = Point(x=x1, y=y1)
        end = Point(x=x2, y=y2)
        box = BoundingBox(start_point=start, end_point=end)
        detections.append(box)
  return detections
