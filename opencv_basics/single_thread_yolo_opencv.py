import cv2 as cv
import time
from ultralytics import YOLO

cfr = 2
skip = False
show = True
do_detection  = True
model_plot = False
model_name = "yolo11n.pt"
source = 0

model = YOLO(model_name)
CONFIDENCE_THRESHOLD = 0.5

cap = cv.VideoCapture(source)
cam_fps = cap.get(cv.CAP_PROP_FPS)

fps, frameNo = 0, 0
prev_time = time.time()
start_time = prev_time

while cap.isOpened():
    ret, frame = cap.read()
    frameNo += 1
    
    if frameNo % int(cam_fps/cfr) != 0 and skip:
        continue
    
    if not ret:
        break
    
    if do_detection:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes = [0])
        if model_plot:
            frame = results[0].plot()
        else:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0].item()  # Confidence score
                    cls = int(box.cls[0].item())  # Class index
                    label = f"{result.names[cls]}: {conf:.2f}"  # Class name and confidence

                    # Draw bounding box
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label text
                    cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Calculate FPS
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    avg_fps = frameNo / elapsed_time if elapsed_time > 0 else 0

    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    print(f"FPS: {int(fps)}, Avg: {avg_fps:.2f}")

    if show:
        cv.putText(frame, f"FPS: {int(fps)}, Avg: {avg_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Frame', frame)
 

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


print(f"Final Average FPS: {avg_fps:.2f}")

cap.read
cv.destroyAllWindows()