from enum import Enum
from .obscure import obscure_region, ObscureMethod, obscure_ellipse
from .faces.mediapipe import detect_faces_mediapipe
from .faces.retinaface import detect_faces_retinaface
from .faces.yolov8face import detect_faces_yolo
from .faces.yolov11face import detect_faces_yolo11
from .image_context import ImageContext
import time
from settings import settings
import cv2

class FaceModel(Enum):
  RETINA_FACE = 1
  MEDIA_PIPE = 2
  YOLO_V8_FACE = 3
  YOLO_V11_FACE = 4

def process_faces(image: ImageContext, model=FaceModel.YOLO_V11_FACE, method=ObscureMethod.GAUSSIAN_PLUS_PIXELATE):
  start_time = time.perf_counter_ns()
  if (model == FaceModel.RETINA_FACE):
    detections = detect_faces_retinaface(image.rgb)
  elif (model == FaceModel.YOLO_V8_FACE):
    detections = detect_faces_yolo(image.bgr)
  elif (model == FaceModel.YOLO_V11_FACE):
    detections = detect_faces_yolo11(image.bgr)
  elif (model == FaceModel.MEDIA_PIPE):
    detections = detect_faces_mediapipe(image.rgb)
  else:
    detections = []

  if len(detections) > 0:
    for d in detections:
      obscure_ellipse(image, d, method, expand_by=settings.face_obscure_padding)
      if settings.draw_rectangles:
        cv2.rectangle(image.bgr, (d.start_point.x, d.start_point.y), (d.end_point.x, d.end_point.y), settings.detection_colour)
  end_time = time.perf_counter_ns()
  elapsed_time = end_time - start_time
  return elapsed_time
