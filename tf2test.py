import tensorflow as tf
import cv2
import numpy as np

# Load your custom YOLO model (yolo.pb)
model = tf.saved_model.load("yolov8n_saved_model")  # Replace with the path to your YOLO model directory

# Load and preprocess the input image
image = cv2.imread("bus.jpg")
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_image = tf.image.convert_image_dtype(input_image, tf.float32)
input_image = tf.image.resize(input_image, (640, 640))  # Adjust size as needed
input_image = tf.expand_dims(input_image, axis=0)

# Perform inference
detections = model(input_image)

# Process detection results and draw bounding boxes
for detection in detections:
    class_id = int(detection[0])
    score = detection[1]
    x, y, width, height = detection[2:]

    if score > 0.5:  # Adjust the confidence threshold as needed
        x1, y1, x2, y2 = int((x - width / 2) * image.shape[1]), int((y - height / 2) * image.shape[0]), int((x + width / 2) * image.shape[1]), int((y + height / 2) * image.shape[0])
        class_name = "YourClass"  # Replace with your class name
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the annotated image
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
