import cv2
import numpy as np
from pathlib import Path
from app.services.detect_faces import process_faces, FaceModel
from app.services.detect_vlps import process_plates, VLPModel
from app.services.image_context import ImageContext
from dataset_utils import load_test_dataset
import argparse

def main():
  parser = argparse.ArgumentParser(description="Anonymisation demo")
  parser.add_argument("--directory", default='dataset', help="Dataset directory", required=False)
  args = parser.parse_args()
  dataset = load_test_dataset(args.directory)
  for model in [ FaceModel.YOLO_V11_FACE, FaceModel.RETINA_FACE, FaceModel.YOLO_V8_FACE ]:
    counter = 0
    total_time = 0
    for elt in dataset:
      image = elt['image']
      img_np = np.array(image)
      cv_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
      ctx = ImageContext(cv_image)  
      time_ns = process_faces(ctx, model=model)
      total_time += time_ns
      counter += 1
      if (counter % 10 == 0):
        print(f'Done {counter}')

    avg_time = float(total_time) / counter
    print(f'Model {model} average detect + obscure time = {avg_time} nanoseconds ({ (avg_time / 1_000_000) } milliseconds)')

  for model in [ VLPModel.BHASKRR, VLPModel.MORSE_TECH_LAB ]:
    counter = 0
    total_time = 0
    for elt in dataset:
      image = elt['image']
      img_np = np.array(image)
      cv_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)    
      ctx = ImageContext(cv_image)  
      time_ns = process_plates(ctx, model=model)
      total_time += time_ns
      counter += 1

      if (counter % 10 == 0):
        print(f'Done {counter}')

    avg_time = float(total_time) / counter
    print(f'Model {model} license plate processing average detect + obscure time = {avg_time} nanoseconds ({ (avg_time / 1_000_000) } milliseconds)')

if __name__ == "__main__":
  main()
