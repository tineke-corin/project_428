from collections import namedtuple
from enum import Enum
import cv2
import numpy
from app.services.utils import BoundingBox, Point, expand_region
from settings import settings
from .image_context import ImageContext

class ObscureMethod(Enum):
  GAUSSIAN_ONLY = 1
  GAUSSIAN_PLUS_PIXELATE = 2
  GAUSSIAN_PLUS_NOISE = 3

"""
  pixelate(roi)

  Applies pixelation to the roi, which is a portion of an image.
  Pixelation is achieved by resizing the roi to one quarter of its
  original size, then sizing it back up again.
"""
def pixelate(roi, shape):
  height, width = shape
  pixel_size = int(min(height, width) * settings.pixel_proportion)

  # Don't let pixel_size go below 1
  small_width = width // max(pixel_size, 1)
  small_height = height // max(pixel_size, 1)

  temp = cv2.resize(roi, (small_width, small_height), interpolation=cv2.INTER_AREA)
  out = cv2.resize(temp, (width, height), interpolation=cv2.INTER_AREA)
  return out

"""
  gaussian_blur(roi)

  Applies a gaussian blur to the roi, which is a portion of an image.
  The kernel size for the blur is calculated based on the image size
  and the KERNEL_PROPORTION setting. The bigger the kernel proportion,
  the more intense the blurring.
"""
def gaussian_blur(roi, shape):
  # find the length of the longest side of the box
  height, width = shape
  side = max(width, height)

  # the nearest odd number to the configured proportion of the side
  k = int(side * settings.kernel_proportion)
  if (k % 2 == 0):
    k += 1

  # Passing 0 for sigmaX lets cv2 calculate sigmaX and sigmaY based on kernel size
  out = cv2.GaussianBlur(roi, (k, k), 0)
  return out

"""
  gaussian_noise(roi)
"""
def gaussian_noise(roi, shape):
  height, width = shape
  noise = numpy.random.normal(0, settings.noise_std_dev, (height, width, 3)) 
  out = roi + noise
  out = numpy.clip(out, 0, 255)
  return out

"""
  obscure_region(image, region, method)

  Obscures the given region in the given image, using the given method.
  The method can be either Gaussian blur or pixelation. Returns the
  input image, with the specified region obscured.
"""
def obscure_region(ctx: ImageContext, region: BoundingBox, method: ObscureMethod, expand_by=settings.face_obscure_padding):
  # img_width and img_height are passed to ensure the expanded ROI does not go
  # outside of the bounds of the image.
  if expand_by:
    region = expand_region(region, ctx.width, ctx.height, expand_by)

  x1, y1 = region.start_point
  x2, y2 = region.end_point
  roi = ctx.bgr[y1:y2, x1:x2]
  roi_shape = (region.height, region.width)

  if (method == ObscureMethod.GAUSSIAN_ONLY):
    blurred = gaussian_blur(roi, roi_shape)
  elif (method == ObscureMethod.GAUSSIAN_PLUS_PIXELATE):
    blurred = gaussian_blur(roi, roi_shape)
    blurred = pixelate(blurred, roi_shape)
  elif (method == ObscureMethod.GAUSSIAN_PLUS_NOISE):
    blurred = gaussian_blur(roi, roi_shape)
    blurred = gaussian_noise(blurred, roi_shape)
  else:
    return image

  ctx.bgr[y1:y2, x1:x2] = blurred

"""
  obscure_oval(image, region, method)

  Creates an oval from the given bounding box and then obscures with
  the given method.
"""
def obscure_ellipse(ctx: ImageContext, region: BoundingBox, method: ObscureMethod, expand_by=settings.face_obscure_padding):
  # img_width and img_height are passed to ensure the expanded ROI does not go
  # outside of the bounds of the image.
  if expand_by:
    region = expand_region(region, ctx.width, ctx.height, expand_by)

  x1, y1 = region.start_point
  x2, y2 = region.end_point
  roi = ctx.bgr[y1:y2, x1:x2]
  roi_shape = (region.height, region.width)

  axes = (int(region.width/2), int(region.height/2))
  mask = numpy.zeros((region.height, region.width), dtype=numpy.uint8)

  # Fills the mask with white
  cv2.ellipse(mask, axes, axes, 0, 0, 360, (255,255,255), -1)
  if (method == ObscureMethod.GAUSSIAN_ONLY):
    blurred = gaussian_blur(roi, roi_shape)
  elif (method == ObscureMethod.GAUSSIAN_PLUS_PIXELATE):
    blurred = gaussian_blur(roi, roi_shape)
    blurred = pixelate(blurred, roi_shape)
  elif (method == ObscureMethod.GAUSSIAN_PLUS_NOISE):
    blurred = gaussian_blur(roi, roi_shape)
    blurred = gaussian_noise(blurred, roi_shape)
  else:
    return image

  ctx.bgr[y1:y2, x1:x2] = numpy.where(mask[..., None] == 255, blurred, ctx.bgr[y1:y2,x1:x2])
