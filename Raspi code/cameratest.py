import cv2
print("123")
# Open the default camera (0) or change the index if multiple cameras are connected
cap = cv2.VideoCapture(1)
print("456")
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'q' to quit.")

# Loop to continuously capture frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame in a window
    cv2.imshow('Camera Feed', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()