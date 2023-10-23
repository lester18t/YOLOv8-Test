import cv2
import numpy as np
import tensorflow as tf
import yaml


# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="yolov8n_saved_model/yolov8n_float32.tflite")
interpreter.allocate_tensors()

# Load and preprocess the input image
image = cv2.imread("bus.jpg")

class_names = []
with open("yolov8n_saved_model/metadata.yaml", "r") as yaml_file:
    class_names = yaml.safe_load(yaml_file)["names"]

# Perform preprocessing (resize, normalize, etc.) as required by your model
# Ensure the image is in the expected data type (e.g., FLOAT32)
input_details = interpreter.get_input_details()
input_image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
input_image = input_image.astype(np.float32)
input_image = input_image / 255.0  # Normalize to [0, 1]

# Perform inference
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
output_details = interpreter.get_output_details()
detections = interpreter.get_tensor(output_details[0]['index'])

# Process the detection results and draw bounding boxes
for detection in detections[0]:
    class_id = int(detection[1])
    confidence = detection[2]

    if confidence > 0.0:  # Adjust the confidence threshold as needed
        x, y, width, height = int(detection[3] * image.shape[1]), int(detection[4] * image.shape[0]), int(detection[5] * image.shape[1]), int(detection[6] * image.shape[0])

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        class_label = f"Class: {class_names[class_id]}, Confidence: {confidence:.2f}"
        cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the annotated image
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
