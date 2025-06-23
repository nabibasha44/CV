import cv2 as cv
from ultralytics import YOLO
import numpy as np
model = YOLO('yolov8m-pose', task='detect')

cap = cv.VideoCapture('/home/nabi/Videos/Xinthe_entry_frames_01.mp4')
track_counts = {}
MAX_MISSING_FRAMES = 30
last_seen = {}
fps = int(cap.get(cv.CAP_PROP_FPS))


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    model_result = model.track(frame, classes=[0], conf=0.5, persist=True, verbose=False)
    current_frame_ids = []
    for person_id, person in enumerate(model_result[0].keypoints.data):
        keypoints = person.cpu().numpy().reshape(-1, 3)

        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]

        if keypoints[15][2]:
            ankle = keypoints[15][:2]
        elif keypoints[16][2]:
            ankle = keypoints[16][:2]
        else:
            ankle = None

        # Midpoints
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        mid_knee = (left_knee + right_knee) / 2
        # Vector from shoulder to hip
        delta = mid_hip - mid_shoulder
        body_angle = abs(np.degrees(np.arctan2(delta[1], delta[0])))

        # Use Euclidean distances
        body_height = np.linalg.norm(mid_shoulder - mid_hip)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        ratio = body_height / shoulder_width

        x1, y1, x2, y2 = map(int, model_result[0][person_id].boxes.xyxy.cpu().numpy()[0])
        width = int(x2 - x1)
        height = int(y2 - y1)
        conf = model_result[0][person_id].boxes.conf.cpu().numpy()  # shape: (N,) – confidence scores
        cls = int(model_result[0][person_id].boxes.cls.cpu().numpy()[0])
        track_id = int(model_result[0][person_id].boxes.id.cpu().numpy()[0])
        current_frame_ids.append(track_id)

        if track_id in track_counts:
            if len(track_counts[track_id]) > fps:
                track_counts[track_id].pop(0)
            track_counts[track_id].append([body_angle, ankle])

        else:
            track_counts[track_id] = [[body_angle, ankle]]

        posture = "Standing"
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Compare foot or knee positions between frames
        motion = np.linalg.norm(ankle - track_counts[track_id][0][1])
        if motion > 10:  # pixels moved
            posture = "Walking"

        # If body is horizontal (angle near 0° or 180°)
        if body_angle < 45 or body_angle > 135:
            if track_counts[track_id][0][0] > 40 or track_counts[track_id][0][0] < 135:
                frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                posture = "Falling"

        label = f"person_{track_id}__ {posture}"
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow('Stream', frame)
    for track_id in list(track_counts.keys()):
        if track_id in current_frame_ids:
            ...
        else:
            track_counts.pop(track_id)

    if cv.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()