import cv2
from ultralytics import YOLO

# Load the YOLO classification model
model = YOLO("./best.pt", task='classify')

# Open the video feed
video_path = "/dev/video0"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object

ret_val , cap_for_exposure = cap.read()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BRIGHTNESS, -21)
cap.set(cv2.CAP_PROP_CONTRAST, 5)
cap.set(cv2.CAP_PROP_SATURATION, 43)
cap.set(cv2.CAP_PROP_HUE, 0)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
cap.set(cv2.CAP_PROP_SHARPNESS, 7)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4444)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 1302)

frame_rate = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (320,320))


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:


        # Crop the frame to 320x320
        height, width, _ = frame.shape
        start_x = width // 2 - 160
        start_y = height // 2 - 120
        cropped_frame = frame[start_y:start_y + 320, start_x:start_x + 320]

        # Run YOLO inference on the frame
        results = model(cropped_frame, verbose=False, imgsz=320)

        # Get the top-1 class index and confidence
        probs = results[0].probs  # Probs object
        if probs is not None:
            top1_index = probs.top1  # Index of the top class
            top1_conf = probs.top1conf  # Confidence of the top class

            # Print the highest confidence label and its probability
            print(f"{model.names[top1_index]}, {top1_conf:.2f}")
            cv2.putText(
                cropped_frame,
                f"{model.names[top1_index]}, {top1_conf:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Display the original frame
        # cv2.imshow(window_name, cropped_frame)
        out.write(cropped_frame)
        cv2.waitKey(1)

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
