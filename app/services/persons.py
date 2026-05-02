from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")
yolo_target_class = 'person'

def contains_people(image_path: str) -> bool:
  result = yolo_model(image_path)
  for box in result[0].boxes:
    class_id = int(box.cls[0])
    class_name = yolo_model.names[class_id]
    confidence = box.conf[0]

    if class_name == yolo_target_class and confidence > 0.5:
      return True
  return False

