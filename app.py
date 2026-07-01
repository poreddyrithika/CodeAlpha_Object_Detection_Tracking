from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

video_path = "input/traffic_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Fix if FPS is invalid
if fps == 0:
    fps = 30

print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps}")

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Video writer
out = cv2.VideoWriter(
    "output/output_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated = results[0].plot()

    out.write(annotated)

    cv2.imshow("Object Detection & Tracking", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Finished! Output saved in output/output_video.avi")