from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Export the model
model.export(format='tflite')