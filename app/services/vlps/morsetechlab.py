from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import mediapipe as mp
import cv2
from app.services.utils import Point, BoundingBox
from settings import settings
import time

"""
In accordance with the AGPLv3 license:
- If you **use this model** in a service or project, you must **open source** the code that uses it.
- Please give proper attribution to Roboflow, Ultralytics, and MorseTechLab when using or deploying.
"""

_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        # https://huggingface.co/morsetechlab/yolov11-license-plate-detection/blob/main/README.md
        weights_path = hf_hub_download(
            repo_id="morsetechlab/yolov11-license-plate-detection",
            filename="license-plate-finetune-v1l.pt"
        )
        _yolo_model = YOLO(weights_path)
    return _yolo_model

def detect_vlps_morsetechlab(image_path):
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
