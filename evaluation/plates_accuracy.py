from datasets import load_dataset, concatenate_datasets
from app.services.detect_vlps import VLPModel
from app.services.vlps.bhaskrr import detect_vlps_bhaskrr
from app.services.vlps.morsetechlab import detect_vlps_morsetechlab
from app.services.utils import Point, BoundingBox, is_within_box, intersection_over_union, point_from_normalised, box_from_normalised, expand_region, filter_aspect_ratio, filter_size, filter_contrast
from PIL import Image
import cv2
import numpy as np
from dataset_utils import load_test_dataset, get_ground_truths
from .test_utils import get_statistics, get_precision_and_recall
from settings import settings
import argparse

def main():
  parser = argparse.ArgumentParser(description="Anonymisation speed test")
  parser.add_argument("--directory", default='dataset', help="Dataset directory", required=False)
  args = parser.parse_args()
  dataset = load_test_dataset(args.directory, shuffle=Fale)

  for model in [ VLPModel.BHASKRR, VLPModel.MORSE_TECH_LAB ]:
    counter = 0
    statistics = []
    for elt in dataset:
      image = elt['image']
      img_np = np.array(image)
      image_width, image_height = image.size
      cv_image = None

      # Read locations of annotated license plates
      plate_truths = get_ground_truths(elt['plates'], image_width, image_height)

      # Get the face detections, and draw their bounding boxes on the image    
      if model == VLPModel.MORSE_TECH_LAB:
        plate_detections = detect_vlps_morsetechlab(img_np)
      elif model == VLPModel.BHASKRR:
        plate_detections = detect_vlps_bhaskrr(img_np)

      plate_detections = [ expand_region(d, image_width, image_height, scale=settings.plate_obscure_padding) for d in plate_detections ]
      plate_detections = [ d for d in plate_detections if filter_aspect_ratio(d) ]
      plate_detections = [ d for d in plate_detections if filter_size(d, np.shape(image)) ]

      plate_statistics = get_statistics(plate_truths, plate_detections)
      statistics.append(plate_statistics)

      counter += 1
      if (counter % 10 == 0):
        print(f'Done {counter}')
    
    plates_p, plates_r, plates_f1 = get_precision_and_recall(statistics)

    print(f'Mmodel {model} results')
    print(f'Plates: Precision={plates_p}, Recall={plates_r}, F1={plates_f1}')

if __name__ == "__main__":
  main()
