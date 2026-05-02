import cv2
import numpy as np
from pathlib import Path
from app.services.detect_faces import process_faces, FaceModel
from app.services.detect_vlps import process_plates, VLPModel
from app.services.image_context import ImageContext
import cv2
import os

FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  # red (BGR)

def main():
  cv2.namedWindow('Anonymiser', cv2.WINDOW_AUTOSIZE)
  folder = "images-for-eval"
  for filename in os.listdir(folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
      print(filename)
      cv_image = cv2.imread(os.path.join(folder, filename))
      ctx = ImageContext(cv_image)  
      time_ns = process_faces(ctx, model=FaceModel.YOLO_V11_FACE)
      time_ns = process_plates(ctx, model=VLPModel.BHASKRR)
      cv2.imwrite(f'{folder}/anon_{filename}', ctx.bgr)
      cv2.imshow('Anonymiser', ctx.bgr)
      cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
