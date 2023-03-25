import cv2
import numpy as np

# Load the pre-trained face, eye and smile detection models from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the face region from the frame
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

            # Detect smile in the face region
            smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

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

                        # Search for crow's feet or wrinkles around the outer corners of the eyes
                        crow_feet_present = False
                        for i in range(1, len(search_region)-1):
                            for j in range(1, len(search_region[0])-1):
                                # Check for a change in brightness in the search region
                                brightness_change = abs(int(search_region[i][j+1]) - int(search_region[i][j]))
                                if brightness_change > 10:
                                    crow_feet_present = True
                                    break
                                brightness_change = abs(int(search_region[i+1][j]) - int(search_region[i][j]))
                                if brightness_change > 10:
                                    crow_feet_present = True
                                    break
                            if crow_feet_present:
                                break

                        # If crow's feet or wrinkles are detected, draw a green circle around the eye
                        if crow_feet_present:
                            center = (int(x+ex+ew/2), int(y+ey+eh/2))
                            radius = int(ew/2)
                            cv2.circle(face_roi_color, center, radius, (0, 255, 0), 2)

                # Calculate the confidence percentage
                confidence_percent = round((len(eyes) + len(smiles)) / 3 * 100)

                # Determine the emotion label based on the detected face region
                if confidence_percent > 60:
                    emotion_label = "Happy"
                else:
                    emotion_label = "Neutral"

                # Display the emotion label and confidence percentage next to the recognized face
                text = f"{emotion_label} ({confidence_percent}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the processed frame
        cv2.imshow('Processed Frame', frame)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Call the recognize_emotion_and_face function
    recognize_emotion_and_face()

if __name__ == '__main__':
    main()
