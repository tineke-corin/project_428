import time
from .vlps.morsetechlab import detect_vlps_morsetechlab
from .vlps.bhaskrr import detect_vlps_bhaskrr
from .obscure import obscure_region, ObscureMethod
from enum import Enum
from settings import settings
from .utils import filter_aspect_ratio, filter_size
import cv2
import numpy
from .image_context import ImageContext

class VLPModel(Enum):
  MORSE_TECH_LAB = 1
  BHASKRR = 2

def process_plates(image: ImageContext, model=VLPModel.MORSE_TECH_LAB, method=ObscureMethod.GAUSSIAN_PLUS_PIXELATE):
  start_time = time.perf_counter_ns()

  if model == VLPModel.MORSE_TECH_LAB:
    detections = detect_vlps_morsetechlab(image.rgb)
  elif model == VLPModel.BHASKRR:
    detections = detect_vlps_bhaskrr(image.rgb)
  
  detections = [ d for d in detections if filter_aspect_ratio(d) ]
  detections = [ d for d in detections if filter_size(d, numpy.shape(image)) ]

  if len(detections) > 0:
    for d in detections:
      obscure_region(image, d, method, expand_by=settings.plate_obscure_padding)
      if settings.draw_rectangles:
        cv2.rectangle(image.bgr, (d.start_point.x, d.start_point.y), (d.end_point.x, d.end_point.y), settings.detection_colour)
  end_time = time.perf_counter_ns()
  elapsed_time = end_time - start_time
  return elapsed_time
