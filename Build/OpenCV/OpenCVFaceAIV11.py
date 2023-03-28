import cv2
import numpy as np

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
            print("BIG")
        elif sw > min_normal_smile_width and sh > min_normal_smile_height:
            smile_detected = True
            print("SMALL")
        else:
            smile_detected = False
            big_smile_detected = False
            print("NO SMILE")

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
    # Define the minimum distance between the inner eyebrows and the top edge of the face to be considered an "up" orientation
    eyebrow_up_offset = 10

    # Define the minimum distance between the mouth corners and the bottom edge of the face to be considered a "down" orientation
    mouth_down_offset = 10

    # Detect the face region
    faces = face_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        # Detect the eyebrows and mouth in the face region
        eyebrows = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_eyepair_big.xml')
        mouth = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyebrows_rects = eyebrows.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
        mouth_rects = mouth.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the eyebrows and mouth
for (x, y, w, h) in eyebrows_rects:
    y = int(y - 0.15*h)
    cv2.rectangle(face, (x, y), (x+w, y+h), (0, 255, 0), 2)
for (x, y, w, h) in mouth_rects:
    cv2.rectangle(face, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Show the final image with facial features highlighted
cv2.imshow('Facial Features Detected', img)
cv2.waitKey(0)


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
            break

        # Process each detected face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:

            crow_feet_detected = False

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

                    # Calculate the confidence percentage
                    # if smile_detected and crow_feet_detected:
                    #     confidence_percent = round((num_eyes + num_smiles) / 3 * 1.3 * 5)
                    #     print("I got to the smile and crow feet")
                    # elif smile_detected:
                    #     confidence_percent = round((num_eyes + num_smiles) / 3 * 1.2 * 50)
                    #     print("I got to the smile")
                    # elif crow_feet_detected:
                    #     confidence_percent = round((num_eyes + num_smiles) / 3 * 1.1 * 50)
                    #     print("I got to the crow feet")
                    # else:
                    #     confidence_percent = round((num_eyes + num_smiles) / 3 * 100)
                    #     print("I got to nothing")

                    # # Determine the confidence percentage based on the detected features
                    # if big_smile_detected:
                    #     print("BIG SMILE")
                    #     if crow_feet_detected:
                    #         confidence_percent = round((num_eyes + num_smiles) / 3 * 1.5 * 50)
                    #     else:
                    #         confidence_percent = round((num_eyes + num_smiles) / 3 * 1.3 * 100)
                    #         print("I got to nothing")
                    # elif smile_detected:
                    #     print("NORMAL SMILE")
                    #     if crow_feet_detected:
                    #         confidence_percent = round((num_eyes + num_smiles) / 3 * 1.2 * 50)
                    #     else:
                    #         confidence_percent = round((num_eyes + num_smiles) / 3 * 100)
                    #         print("I got to nothing for small smile")
                    # elif crow_feet_detected:
                    #     confidence_percent = round((num_eyes + num_smiles) / 3 * 1.1 * 50)
                    #     print("ONLY CROW")
                    # else:
                    #     confidence_percent = round((num_eyes + num_smiles) / 3 * 100)
                    #     print("NO CROWS FEET OR SMALL / BIG SMILE")

                    # Calculate the confidence percentage based on the number of detected eyes and smile
                    confidence_percent = (len(eyes) + int(smile_detected)) / 3 * 1

                    # Add to the confidence rating if a big smile or crows feet are detected
                    if big_smile_detected:
                        confidence_percent += 80

                    if smile_detected:
                        confidence_percent += 50

                    if crows_feet_detected:
                        confidence_percent += 30


                    # Determine the emotion label based on the detected face region
                    if confidence_percent > 80:
                        label = 'Happy'
                    else:
                        label = 'Neutral'

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