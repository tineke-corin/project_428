from pydantic import BaseModel, computed_field
from enum import Enum
import cv2
import numpy
from settings import settings

class Point(BaseModel):
  x: int
  y: int

  def __iter__(self):
    yield self.x
    yield self.y

class BoundingBox(BaseModel):
  start_point: Point
  end_point: Point

  @property
  def width(self) -> int:
    return abs(self.end_point.x - self.start_point.x)

  @property
  def height(self) -> int:
    return abs(self.end_point.y - self.start_point.y)

  @property
  def centre(self) -> Point:
    xc = int(self.end_point.x - (self.width/2))
    yc = int(self.end_point.y - (self.height/2))
    return Point(x=xc, y=yc)

  @property
  def area(self) -> int:
    return self.width * self.height

  def contains_point(self, p: Point) -> bool:
    min_x = min(self.start_point.x, self.end_point.x)
    min_y = min(self.start_point.y, self.end_point.y)
    max_x = max(self.start_point.x, self.end_point.x)
    max_y = max(self.start_point.y, self.end_point.y)
    
    return min_x <= p.x <= max_x and min_y <= p.y <= max_y

def point_from_normalised(cx: float, cy: float, img_w: int, img_h: int) -> Point:
  x = int(cx * img_w)
  y = int(cy * img_h)
  return Point(x=x, y=y)

def box_from_normalised(top_left: Point, w: float, h: float, img_w: int, img_h: int) -> BoundingBox:
  wPx = w * img_w
  hPx = h * img_h
  return BoundingBox(
      start_point = top_left,
      end_point = Point(x=int(top_left.x + wPx), y=int(top_left.y + hPx))
  )

def is_within_box(p: Point, box: BoundingBox) -> bool:
  return box.contains_point(p)

def intersection_over_union(detection: BoundingBox, ground_truth: BoundingBox):
  # determine the (x, y)-coordinates of the intersection rectangle
  x_left = max(detection.start_point.x, ground_truth.start_point.x)
  y_top = max(detection.start_point.y, ground_truth.start_point.y)
  x_right = min(detection.end_point.x, ground_truth.end_point.x)
  y_bottom = min(detection.end_point.y, ground_truth.end_point.y)

  if x_right < x_left or y_bottom < y_top:
    # no intersection
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both the prediction and ground-truth
  # rectangles
  detection_area = (detection.end_point.x - detection.start_point.x) * (detection.end_point.y - detection.start_point.y)
  ground_truth_area = (ground_truth.end_point.x - ground_truth.start_point.x) * (ground_truth.end_point.y - ground_truth.start_point.y)

  iou = intersection_area / float(detection_area + ground_truth_area - intersection_area)
  assert iou >= 0.0
  assert iou <= 1.0
  return iou

"""
    filter_contrast(detection)

    A real license plate should have quite high contrast.
    But actually this made detections a LOT worse!
"""
def filter_contrast(detection, image, rms_threshold=40):
    cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    x1, y1 = detection.start_point
    x2, y2 = detection.end_point
    roi = cv_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rms_contrast = numpy.std(gray)
    return rms_contrast > rms_threshold

"""
   filter_size(detection, img_shape)
   
   An LP should rarely occupy more than 15% of the total frame
"""
def filter_size(detection: BoundingBox, img_shape):
  img_w, img_h, _c = img_shape
  img_area = img_w * img_h
  return detection.area < (0.15 * img_area)

"""
  filter_aspect_ratio(detection)

  Even a partly occluded LP is most likely to be wider than it is tall.
  Both LP models occasionally detect LPs on buildings, and these FPs are
     generally really large and either square or portrait.
  This should filter out the worst of them.
"""
def filter_aspect_ratio(detection: BoundingBox):
  return detection.height < detection.width

"""
    expand_region(region, max_x, max_y, scale)

    Expands a region by the given scale. The new bounding
    box is clipped to ensure it does not go off the edges of the original
    image (bounded by 0, 0, max_x, max_y).
"""
def expand_region(region: BoundingBox, max_x, max_y, scale = None):
  # default to no expansion
  if scale is None:
    return region

  extra_w = int((region.width * scale)/2)
  extra_h = int((region.height * scale)/2)

  new_x1 = region.start_point.x - extra_w
  new_x2 = region.end_point.x + extra_w
  new_y1 = region.start_point.y - extra_h
  new_y2 = region.end_point.y + extra_h

  # after expanding, make sure the coordinates of the ROI have not gone
  # outside the bounds of the actual image

  # x1 and y1 should not go below 0
  new_x1 = max(new_x1, 0)
  new_y1 = max(new_y1, 0)
  # x2 and y2 should not go above width/height
  new_x2 = min(new_x2, max_x)
  new_y2 = min(new_y2, max_y)

  return BoundingBox(
                     start_point=Point(x=new_x1, y=new_y1),
                     end_point=Point(x=new_x2, y=new_y2)
                   )
