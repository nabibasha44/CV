import cv2 as cv
from ultralytics import YOLO
model = YOLO('yolov8m', task='detect')

cap = cv.VideoCapture('./test_video/fall.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    model_result = model.predict(frame, classes=[0])
    for result in model_result:
        boxes = result.boxes  # result.boxes is a ultralytics.engine.results.Boxes object

        x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[0])
        width = int(x2 - x1)
        height = int(y2 - y1)
        conf = boxes.conf.cpu().numpy()  # shape: (N,) – confidence scores
        cls = int(boxes.cls.cpu().numpy()[0])  # shape: (N,) – class indices
        threshold = height - width

        label = f"{result.names[int(cls)]} {conf[0]}"

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if threshold < 0:
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    cv.imshow('Stream', frame)

    if cv.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()