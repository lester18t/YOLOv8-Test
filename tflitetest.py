import tensorflow as tf
import cv2

model = tf.lite.Interpreter(model_path="yolov8n_saved_model/yolov8n_float32.tflite")
model.allocate_tensors()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.) to match the model's input requirements

    # Perform inference
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], frame)
    model.invoke()
    detections = model.get_tensor(output_details[0]['index'])

    # Process detections and draw bounding boxes on the frame

    # Display the result
    cv2.imshow("Real-time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()