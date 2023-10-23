from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='Face_Detector/data.yaml', resume=True, epochs=100, imgsz=640)