from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('face.jpg', save=True, imgsz=320, conf=0.2)