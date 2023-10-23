import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "classroom.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():

    success, frame = cap.read()
    #frame = cv2.resize(frame, [640,640])

    if success:

        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        #get bounding boxes coordinates
        #for r in results:
        #    print(r.boxes)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()