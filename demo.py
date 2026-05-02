import cv2
import numpy as np
from pathlib import Path
from app.services.detect_faces import process_faces, FaceModel
from app.services.detect_vlps import process_plates, VLPModel
from app.services.obscure import ObscureMethod
from dataset_utils import load_test_dataset
import argparse
import os
from app.services.image_context import ImageContext

FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  # red (BGR)

faceModel = FaceModel.YOLO_V11_FACE
vlpModel = VLPModel.BHASKRR
method = ObscureMethod.GAUSSIAN_PLUS_PIXELATE

def main():
    parser = argparse.ArgumentParser(description="Anonymisation demo")
    parser.add_argument("--directory", default='dataset', help="Image directory", required=True)
    args = parser.parse_args()

    cv2.namedWindow('Anonymiser', cv2.WINDOW_AUTOSIZE)

    for filename in os.listdir(args.directory):
      if filename.endswith((".jpg", ".png", ".jpeg")):
        cv_image = cv2.imread(os.path.join(args.directory, filename))
        ctx = ImageContext(cv_image)  
  
        _time = process_faces(ctx, model=faceModel, method=method)
        _time = process_plates(ctx, model=vlpModel, method=method)

        cv2.putText(ctx.bgr, filename, (10, 20), cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        cv2.imshow('Anonymiser', ctx.bgr)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
  main()
