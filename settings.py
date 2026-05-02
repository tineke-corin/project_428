from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Tuple

class Settings(BaseSettings):
  # Detection configuration
  face_min_confidence: float = 0.2
  plate_min_confidence: float = 0.2
  img_sz: int = 2048
  iou_threshold: float = 0.5
  face_size_threshold: int = 12

  # Drawing settings
  draw_rectangles: bool = False
  truth_colour: Tuple[int, int, int] = (0, 0, 255)  # red BGR
  detection_colour: Tuple[int, int, int] = (0, 255, 0)  # green BGR

  # Obscuration settings
  face_obscure_padding: float = 0.3   # Pads the ROI by 30%
  plate_obscure_padding: float = 0.15   # Pads the ROI by 15%
  kernel_proportion: float = 1.0 # For Guassian blur
  pixel_proportion: float = 0.25 # For pixelation
  noise_std_dev: int = 25        # Standard deviation value for Gaussian noise

  # Allow an .env file to override these defaults
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
 
settings = Settings() 
