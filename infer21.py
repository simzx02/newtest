import cv2
import multiprocessing
from ultralytics import YOLO
from time import time_ns

def capture_frames(frame_queue, stop_event):
    """
    Captures frames from the camera and puts them into the queue.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera (change 0 to camera index if needed)
    _ , _ = cap.read()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Remove the oldest frame if the queue is full
            height, width, _ = frame.shape
            start_x = width // 2 - 160
            start_y = height // 2 - 120
            cropped_frame = frame[start_y:start_y + 320, start_x:start_x + 320]
            frame_queue.put(cropped_frame)
    cap.release()

def run_yolo(frame_queue, stop_event):
    """
    Reads frames from the queue and runs YOLO inference on them.
    """
    model = YOLO("best.pt", task='classify')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 5, (320, 320))

    # Create a named window for displaying the frames
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        if not frame_queue.empty():
            start = time_ns()
            frame = frame_queue.get()
            # Run YOLO inference on the frame
            results = model(frame, verbose=False, imgsz=320)

            # Get the top-1 class index and confidence
            probs = results[0].probs  # Probs object
            if probs is not None:
                top1_index = probs.top1  # Index of the top class
                top1_conf = probs.top1conf  # Confidence of the top class

                # Print the highest confidence label and its probability
                print(f"{model.names[top1_index]}, {top1_conf:.2f}, {((time_ns() - start)/1e6):.2f} ms")
                cv2.putText(
                    frame,
                    f"{model.names[top1_index]}, {top1_conf:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Display the frame with detection results
            cv2.imshow("YOLO Detection", frame)

            # Write the frame to the output video file
            out.write(frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    # Release the video writer and close the OpenCV window
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = multiprocessing.Queue(maxsize=2)  # Queue to store frames
    stop_event = multiprocessing.Event()  # Event to signal processes to stop

    # Create processes for capturing frames and running YOLO
    capture_process = multiprocessing.Process(target=capture_frames, args=(frame_queue, stop_event))
    yolo_process = multiprocessing.Process(target=run_yolo, args=(frame_queue, stop_event))

    # Start the processes
    capture_process.start()
    yolo_process.start()

    try:
        # Wait for both processes to finish
        capture_process.join()
        yolo_process.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
        stop_event.set()
        capture_process.join()
        yolo_process.join()
    finally:
        stop_event.set()
        capture_process.join()
        yolo_process.join()
