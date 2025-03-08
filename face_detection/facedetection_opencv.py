import cv2
import face_recognition
import numpy as np
import os

known_faces_folder = 'path_to_known_faces_dataset'
known_faces_images = [(os.path.basename(os.path.dirname(os.path.join(root, file))), os.path.join(root, file)) 
              for root, dirs, files in os.walk(known_faces_folder) 
              for file in files]

# List of known faces and their corresponding names
known_face_encodings = []
known_face_names = []

# Function to load and encode known faces
def load_known_faces(known_faces_folder):

    known_faces_images = [(os.path.basename(os.path.dirname(os.path.join(root, file))), os.path.join(root, file)) 
              for root, dirs, files in os.walk(known_faces_folder) 
              for file in files]
    
    for person_name, image_of_person in known_faces_images:
        image_of_person = face_recognition.load_image_file(image_of_person)
        person_face_encoding = face_recognition.face_encodings(image_of_person)
        if person_face_encoding:
            person_face_encoding = person_face_encoding[0]
            known_face_encodings.append(person_face_encoding)
            known_face_names.append(person_name)
        del person_name, image_of_person

# Load known faces
load_known_faces(known_faces_folder)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)

    # Loop through each face in the frame and try to match it with known faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the current face matches any of the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        name = "Unknown"  # Default to "Unknown" if no match is found

        # If there is a match, get the name of the matched person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and put the name of the recognized person
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
