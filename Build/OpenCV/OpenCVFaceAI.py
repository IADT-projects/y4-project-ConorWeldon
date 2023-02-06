import cv2
import dlib
import imutils
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

def detect_landmarks(gray, rects):
    # initialize the list of facial landmarks
    landmarks = []

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = imutils.face_utils.shape_to_np(shape)

        # add the facial landmarks to the list
        landmarks.append(shape)

    # return the landmarks
    return landmarks

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    landmarks = detect_landmarks(gray, rects)
    display_emotions(frame, landmarks)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0x
