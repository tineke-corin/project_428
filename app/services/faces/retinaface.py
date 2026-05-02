from retinaface import RetinaFace
from app.services.utils import Point, BoundingBox
from settings import settings

def detect_faces_retinaface(image_path):
  resp = RetinaFace.detect_faces(image_path, settings.face_min_confidence)
  """
  An example RetinaFace detection looks like:
  {
    "face_1": {
        "score": 0.9993440508842468,
        "facial_area": [155, 81, 434, 443],
        "landmarks": {
          "right_eye": [257.82974, 209.64787],
          "left_eye": [374.93427, 251.78687],
          "nose": [303.4773, 299.91144],
          "mouth_right": [228.37329, 338.73193],
          "mouth_left": [320.21982, 374.58798]
        }
    }
  }
  """
  detections = []
  for key, value in resp.items():
    box = value["facial_area"]
    start = Point(x=box[0], y=box[1])
    end = Point(x=box[2], y=box[3])
    box = BoundingBox(start_point=start, end_point=end)
    detections.append(box)

  return detections
