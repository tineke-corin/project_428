import cv2
import numpy
from pathlib import Path
from typing import Union

class ImageContext:
  def __init__(self, bgr_image: numpy.ndarray):
    if bgr_image is None or not isinstance(bgr_image, numpy.ndarray):
      raise ValueError("Valid BGR numpy array is required")
    self._bgr = bgr_image
    self._rgb = None # Computed lazily
    self._shape = bgr_image.shape

  @classmethod
  def from_file(cls, file_path: Union[str, Path]) -> "ImageContext":
    """Initialize from a file path."""
    bgr = cv2.imread(str(file_path))
    if bgr is None:
      raise FileNotFoundError(f"Could not read image at {file_path}")
    return cls(bgr)

  @property
  def bgr(self) -> numpy.ndarray:
    """Returns the BGR image (used by OpenCV for drawing/obscuring)."""
    return self._bgr

  @property
  def rgb(self) -> numpy.ndarray:
    """Lazily computes and returns the RGB image (used by models like MediaPipe and RetinaFace)."""
    if self._rgb is None:
      self._rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
    return self._rgb

  @property
  def shape(self):
    return self._shape

  @property
  def width(self) -> int:
    return self._shape[1]

  @property
  def height(self) -> int:
    return self._shape[0]

  def to_bytes(self, extension: str = ".jpg") -> bytes:
    """Encodes the final processed image to bytes for the HTTP response."""
    success, encoded_image = cv2.imencode(extension, self._bgr)
    if not success:
      raise ValueError("Failed to encode image")
    return encoded_image.tobytes()
