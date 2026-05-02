import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import torch
import cv2
import mediapipe as mp
import wget

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from app.services.utils import Point, BoundingBox

# this is the best bet from MP, but is still only decent for faces within 5 metres of the camera
# for faces further away you would need to implement tiling
mp_fd_model_file = 'models/face_detection_full_range.tflite'

if not os.path.isfile(mp_fd_model_file):
    os.makedirs('models/', exist_ok=True)
    wget.download('https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite', out='models/'),

# MediaPipe detector
# Create FaceDetector object.
face_base_options = python.BaseOptions(model_asset_path=mp_fd_model_file)
face_options = vision.FaceDetectorOptions(base_options=face_base_options, running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.2)
mp_face_detector = vision.FaceDetector.create_from_options(face_options)

def detect_faces_mediapipe(image_path):
  cv_mat = cv2.imread(image_path)
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
  detection_result = mp_face_detector.detect(image)

  detections = []
  for det in detection_result.detections:
    start = Point(x=det.bounding_box.origin_x, y=det.bounding_box.origin_y)
    end = Point(x=det.bounding_box.origin_x + det.bounding_box.width, y=det.bounding_box.origin_y + det.bounding_box.height)
    box = BoundingBox(start_point=start, end_point=end)
    detections.append(box)

  return detections
