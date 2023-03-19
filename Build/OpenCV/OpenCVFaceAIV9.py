import cv2

# Load the pre-trained face and smile detection models from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def is_smiling(gray_face, face):
    """
    Determine if a face is smiling based on the presence of a smile in the face region

    Args:
        gray_face (numpy.ndarray): the grayscale face region
        face (numpy.ndarray): the color face region

    Returns:
        bool: True if the face is smiling, False otherwise
    """
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in smiles:
        # Draw a rectangle around the smile
        cv2.rectangle(face, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return True
    return False

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

    # Create a new window for displaying the emotion label
    cv2.namedWindow("Emotion")

    while True:
        # Capture a frame from the webcam video stream
        ret, frame = cap.read()

        # Process each detected face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the face region from the frame
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            # Determine if the face is smiling
            if is_smiling(face_roi_gray, face_roi_color):
                emotion_label = "Happy"
            else:
                emotion_label = "Neutral"

            # Display the emotion label in the separate window
            cv2.putText(cv2.namedWindow("Emotion"), emotion_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources used by the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the recognize_emotion_and_face function to execute the script
recognize_emotion_and_face()
