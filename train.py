from ultralytics import YOLO

dataset = 'https://universe.roboflow.com/ds/b3Pny8E8Ox?key=X3h3u1KTq1' #use dataset from generated roboflow using YOLOv8 format

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data=dataset, resume=True, epochs=100, imgsz=640)