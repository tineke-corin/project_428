from datasets import load_dataset, concatenate_datasets
from app.services.faces.mediapipe import detect_faces_mediapipe
from app.services.faces.retinaface import detect_faces_retinaface
from app.services.faces.yolov8face import detect_faces_yolo
from app.services.faces.yolov11face import detect_faces_yolo11
from app.services.detect_faces import FaceModel
from app.services.image_context import ImageContext
from app.services.utils import Point, BoundingBox, is_within_box, intersection_over_union, point_from_normalised, box_from_normalised, expand_region
from PIL import Image
import cv2
import numpy as np
from dataset_utils import load_test_dataset, get_ground_truths
from settings import settings
from .test_utils import get_statistics, get_precision_and_recall
import argparse

def main():
  parser = argparse.ArgumentParser(description="Anonymisation speed test")
  parser.add_argument("--directory", default='dataset', help="Dataset directory", required=False)
  args = parser.parse_args()
  dataset = load_test_dataset(args.directory)

  for model in [ FaceModel.YOLO_V11_FACE, FaceModel.RETINA_FACE, FaceModel.YOLO_V8_FACE ]:
    statistics = []
    counter = 0
    for elt in dataset:
      # A PIL image
      image = elt['image']
      image_width, image_height = image.size

      img_np = np.array(image)
      cv_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
      ctx = ImageContext(cv_image)

      # Read locations of annotated faces
      face_truths = get_ground_truths(elt['faces'], image_width, image_height)

      # Get the face detections, and draw their bounding boxes on the image
      if (model == FaceModel.RETINA_FACE):
        # Note: RetinaFace does not accept a PIL image.
        face_detections = detect_faces_retinaface(ctx.rgb)
      elif (model == FaceModel.YOLO_V8_FACE):
        face_detections = detect_faces_yolo(ctx.bgr)
      elif (model == FaceModel.YOLO_V11_FACE):
        face_detections = detect_faces_yolo11(ctx.bgr)
      elif (model == FaceModel.MEDIA_PIPE):
        face_detections = detect_faces_mediapipe(ctx.rgb)

      # Expand the detected ROIs before comparing to ground truths, since the anonymisation
      # will be of an expanded area.
      face_detections = [ expand_region(d, image_width, image_height, settings.face_obscure_padding) for d in face_detections ]
      face_statistics = get_statistics(face_truths, face_detections)
      statistics.append(face_statistics)

      counter += 1
      if (counter % 10 == 0):
        print(f'Done {counter}')

    faces_p, faces_r, faces_f1 = get_precision_and_recall(statistics)

    print(f'Face model {model} results')
    print(f'Faces: Precision={faces_p}, Recall={faces_r}, F1={faces_f1}')

if __name__ == "__main__":
  main()
