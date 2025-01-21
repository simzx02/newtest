import cv2
import numpy as np

#dectect camera
#cam = cv2.VideoCapture(0)
#if cam.isOpened():
 #   print("Camera is ready!")
#else:
  #  print("Camera not detected.")

# Function to detect a specific color
def detect_color(frame, lower_color, upper_color):
    """Detects objects of a specific color in the given frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Object Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# Define the color range for detection (e.g., red)
# Adjust HSV values as needed for your target object's color
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Initialize the webcam
#cam = cv2.VideoCapture(2)  # Use 0 for the default camera 
#cam = cv2.VideoCapture(1, cv2.CAP_V4L2)

video_path = "/dev/video"
cam = cv2.VideoCapture(1)
print("running1")
# Apply camera settings
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_BRIGHTNESS, -9)
cam.set(cv2.CAP_PROP_CONTRAST, 64)
cam.set(cv2.CAP_PROP_SATURATION, 25)
cam.set(cv2.CAP_PROP_HUE, 0)
cam.set(cv2.CAP_PROP_GAIN, 0)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_BACKLIGHT, 0)
cam.set(cv2.CAP_PROP_SHARPNESS, 2)
cam.set(cv2.CAP_PROP_AUTO_WB, 0)
cam.set(cv2.CAP_PROP_WB_TEMPERATURE, 3686)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv2.CAP_PROP_EXPOSURE, 356)
print("running2")
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'Ctrl+C' to quit.")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Detect the color and draw bounding boxes
        processed_frame = detect_color(frame, lower_red, upper_red)

        # Display the frame
        cv2.imshow("Object Detection", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    cam.release()
    cv2.destroyAllWindows()
