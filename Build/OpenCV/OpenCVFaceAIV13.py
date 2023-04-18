import cv2
import numpy as np

# Used for my Graph
import matplotlib.pyplot as plt

# Used for saving my data
import pickle
import pandas as pd

# Load the pre-trained face, eye and smile detection models from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize variables for smile and crows feet detection
smile_detected = False
crows_feet_detected = False

# Define the minimum size of the smile to be considered a "big" smile
min_big_smile_width = 50
min_big_smile_height = 35

# Define the minimum size of the smile to be considered a "normal" smile
min_normal_smile_width = 45
min_normal_smile_height = 25

# Initialize an empty list to store the data
data_list = []

def show_graph(label):
    """
    Display a graph with the given label in a new window
    """
    # plt.xlabel(label)
    # plt.show()

    data = label # The data that I want to plot
    data_list.append(data)
    print(data)

    # Plot the data using Matplotlib
    plt.plot(data_list)
    plt.draw()
    plt.pause(0.001)

def detect_smile(smile_roi_gray, smile_roi_color):
    """
    Detect the presence of a smile in a region of interest

    Args:
        smile_roi_gray (numpy.ndarray): the grayscale image of the region of interest
        smile_roi_color (numpy.ndarray): the color image of the region of interest

    Returns:
        bool: True if a smile is detected, False otherwise
        bool: True if a big smile is detected, False otherwise
    """
    # Detect smiles in the region of interest
    smiles = smile_cascade.detectMultiScale(smile_roi_gray, scaleFactor=1.1, minNeighbors=5)

    # Check if a smile is detected
    smile_detected = False
    big_smile_detected = False

    for (sx, sy, sw, sh) in smiles:
        # Draw a rectangle around the smile
        cv2.rectangle(smile_roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

        # Check if the smile is big (teeth are showing)
        if sw > min_big_smile_width and sh > min_big_smile_height:
            big_smile_detected = True
            # print("BIG")
        elif sw > min_normal_smile_width and sh > min_normal_smile_height:
            smile_detected = True
            # print("SMALL")
        else:
            smile_detected = False
            big_smile_detected = False
            # print("NO SMILE")

    return smile_detected, big_smile_detected

def detect_sadness(face_roi_gray, face_roi_color):
    """
    Detect the presence of sadness in a region of interest

    Args:
        face_roi_gray (numpy.ndarray): the grayscale image of the region of interest
        face_roi_color (numpy.ndarray): the color image of the region of interest

    Returns:
        bool: True if sadness is detected, False otherwise
    """

    #Testing
    # print("im getting here")

    # Define the threshold distance between the eyebrows and the eyes to be considered an "up" orientation
    eyebrow_eye_distance_threshold = -5

    # Define the minimum distance between the mouth corners and the bottom edge of the face to be considered a "down" orientation
    mouth_down_offset = 100

    # Set sadness detected as false
    sadness_detected = False
    eyebrow_detected = False

    # Detect the face region
    faces = face_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.2, minNeighbors=7)

    for (fx, fy, fw, fh) in faces:
        # Detect the eyebrows and mouth in the face region
        roi_gray = face_roi_gray[fy:fy + fh, fx:fx + fw]
        roi_color = face_roi_color[fy:fy + fh, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        mouth = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Find the topmost point of the eyes
        top_y = float('inf')
        for (ex, ey, ew, eh) in eyes:
            if ey < top_y:
                top_y = ey

        # Check if the distance between the eyebrows and the top of the eyes is above the threshold and the mouth is in a "down" orientation
        for (ex, ey, ew, eh) in eyes:
            for (exb, eyb, ewb, ehb) in eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5):
                if ex == exb and ey == eyb and ew == ewb and eh == ehb:
                    # Calculate the distance between the eyebrows and the top of the eyes
                    eyebrow_eye_distance = eyb - (fy + top_y)

                    if eyebrow_eye_distance > eyebrow_eye_distance_threshold:
                        # print("EYEBROWS DETECTED UP")
                        eyebrow_detected = True
                        for (mx, my, mw, mh) in mouth:
                            # print(my)
                            if my > (fy + fh - mouth_down_offset):
                                # print("I AM SAD")
                                sadness_detected = True

    return sadness_detected, eyebrow_detected

def recognize_emotion_and_face():
    """
    Recognize emotions and faces in a webcam video stream using OpenCV

    Returns:
        dict: a dictionary of emotions and faces information or None if an error occurs
    """
    # Open a connection to the default webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to open webcam")
    except Exception as e:
        print("Error: ", e)
        return None
    
    while True:
        # Capture a frame from the webcam video stream
        ret, frame = cap.read()

        # Convert the frame to JPEG format
        ret, image_data = cv2.imencode('.jpg', frame)

        # Display the captured frame
        cv2.imshow('Captured Frame', frame)

        # Check for key presses
        if cv2.waitKey(1) == ord('q'):
            # Saving as a CSV File
            df = pd.DataFrame(data_list, columns=["Prediction"])
            df.to_csv("results.csv", index=False)

            # Saving as a Excel File
            df = pd.DataFrame(data_list)
            df.to_excel("results.xlsx", index=False)

            # Saving as a PKL File
            with open("results.pkl", "wb") as f:
                pickle.dump(data_list, f)
            break

        # Process each detected face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:

            crows_feet_detected = False

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the face region from the frame
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

            # Detect smile in the face region
            smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

            smile_roi_gray = gray[y+h//2:y+h, x:x+w]
            smile_roi_color = frame[y+h//2:y+h, x:x+w]

            smile_detected, big_smile_detected = detect_smile(smile_roi_gray, smile_roi_color)

            sadness_detected, eyebrow_detected = detect_sadness(face_roi_gray, face_roi_color)

            # Check if both eyes and smile are detected
            if len(eyes) > 1 and len(smiles) > 0:
                # Draw rectangles around the eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

                # Draw a rectangle around the smile
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

                    # Search for crow's feet or wrinkles around the outer corners of the eyes
                    if ex > 0 and ey > 0:
                        # Define the search region around the eye
                        search_region = gray[ey:ey+eh, ex:ex+ew]

                        # Apply a median blur to remove noise
                        search_region = cv2.medianBlur(search_region, 5)

                        # Set the crows feet detection variable to True
                        crows_feet_detected = True

                    # Calculate the confidence percentage based on the number of detected eyes and smiles
                    num_eyes = len(eyes)
                    num_smiles = len(smiles)
                    smile_detected = False

                    # Check if my mouth is open and my teeth are visible
                    for (sx, sy, sw, sh) in smiles:
                        if sy < h/2:
                            smile_detected = True
                            break

                    # Set the smile detection variable to True if a smile is detected
                    if smile_detected:
                        smile_detected = True

                    # Calculate the confidence percentage based on the number of detected eyes and smile
                    confidence_percent = (len(eyes) + int(smile_detected)) / 3 * 1

                    # Add to the confidence rating if a big smile or crows feet are detected
                    if big_smile_detected:
                        confidence_percent += 80

                    # if smile_detected:
                    #     confidence_percent += 50

                    if crows_feet_detected:
                        confidence_percent += 30

                    if sadness_detected:
                        confidence_percent -= 200

                    if eyebrow_detected:
                        confidence_percent -=100


                    # Determine the emotion label based on the detected face region
                    if confidence_percent > 80:
                        label = 'Happy'
                    elif confidence_percent < 0:
                        label = 'Sad'
                    else:
                        label = 'Neutral'

                    print(label)

                    # Show the graph in a new window
                    show_graph(label)

                    # Add the confidence percentage to the text to display next to the recognized face
                    text = f'{label} ({confidence_percent}%)'
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            # Display the processed frame
            cv2.imshow('Processed Frame', frame)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()



def main():
    # Call the recognize_emotion_and_face function
    recognize_emotion_and_face()

if __name__ == '__main__':
    main()