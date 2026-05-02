import cv2
import numpy as np
from pathlib import Path
from app.services.detect_faces import process_faces, FaceModel
from app.services.detect_vlps import process_plates, VLPModel
from app.services.obscure import ObscureMethod
from dataset_utils import load_test_dataset, load_difficult_dataset
import argparse
import time
from app.services.image_context import ImageContext

faceModel = FaceModel.YOLO_V11_FACE
vlpModel = VLPModel.BHASKRR
method = ObscureMethod.GAUSSIAN_PLUS_NOISE

def main():
    counter = 0
    total_time = 0
    parser = argparse.ArgumentParser(description="Anonymisation speed test")
    parser.add_argument("--directory", default='dataset', help="Dataset directory", required=False)
    args = parser.parse_args()
    dataset = load_test_dataset(args.directory)

    for elt in dataset:
      start_time = time.perf_counter_ns()
      image = elt['image']
      img_np = np.array(image)
      cv_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
      ctx = ImageContext(cv_image)  
  
      _t = process_faces(ctx, model=faceModel, method=method)
      _t = process_plates(ctx, model=vlpModel, method=method)
      end_time = time.perf_counter_ns()
      elapsed_time = end_time - start_time
      counter += 1
      total_time += elapsed_time

    avg_time = (total_time / counter)
    print(f'Detect + obscure average time = {avg_time} nanoseconds ({ (avg_time / 1_000_000) } milliseconds)')

if __name__ == "__main__":
  main()
