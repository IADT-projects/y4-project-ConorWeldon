import cv2

# Load the pre-trained face and emotion detection models from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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

            # Detect emotions in the face region
            emotions = emotion_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

            # Process each detected emotion
            for (ex, ey, ew, eh) in emotions:
                # Draw a rectangle around the emotion
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

                # Determine the emotion label based on the detected face region
                if ew > 100 and eh > 100:
                    emotion_label = "Happy"
                else:
                    emotion_label = "Neutral"

                # Display the emotion label next to the recognized face
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources used by the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Call the main function to execute the script
    main()

def main():
    # Call the recognize_emotion_and_face function
    recognize_emotion_and_face()

if __name__ == '__main__':
    main()
