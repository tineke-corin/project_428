"""
  A command-line viewer for your dataset. Displays the locations of annotated license plates
  and faces.
"""

from app.services.utils import Point, BoundingBox, is_within_box, intersection_over_union, point_from_normalised, box_from_normalised
from PIL import Image
import cv2
import numpy as np
import os
import datasets
import argparse
from dataset_utils import load_test_dataset, get_ground_truths

FACE_SIZE_THRESHOLD = 12
FACE_COLOUR = (0, 0, 255)  # red (BGR)
PLATE_COLOUR = (255, 0, 0)  # blue (BGR)

def main():
  parser = argparse.ArgumentParser(description="Anonymisation demo")
  parser.add_argument("--directory", default='dataset', help="Image directory", required=True)
  args = parser.parse_args()
  dataset = load_test_dataset(args.directory)

  for data_point in dataset:
    # Read the image
    image = data_point['image']
    img_np = np.array(image)
    image_width, image_height = image.size
    cv_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Read locations of annotated faces, and mark them on the image
    face_truths = get_ground_truths(data_point['faces'], image_width, image_height)
    for bx in face_truths:
      cv2.rectangle(cv_image, (bx.start_point.x, bx.start_point.y), (bx.end_point.x, bx.end_point.y), FACE_COLOUR)

    # Read locations of annotated license plates, and mark them on the image
    plate_truths = get_ground_truths(data_point['plates'], image_width, image_height)
    for bx in plate_truths:
      cv2.rectangle(cv_image, (bx.start_point.x, bx.start_point.y), (bx.end_point.x, bx.end_point.y), PLATE_COLOUR)

    # Display the marked up image
    cv2.imshow(f'{data_point["file"]} {len(face_truths)} faces', cv_image)
    cv2.waitKey(0)
  cv2.destroyAllWindows()
 
if __name__ == "__main__":
  main()
