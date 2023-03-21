import cv2
import dlib
import openface

# Load the OpenFace models
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
net = openface.TorchNeuralNet("nn4.small2.v1.t7", imgDim=96, cuda=False)

# Load the pre-trained face, eye and smile detection models from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def recognize_emotion_and_face():
    """
    Recognize emotions and faces in a webcam video stream using OpenCV and OpenFace

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

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Get the facial landmarks for the detected face
            landmarks = align(frame, dlib.rectangle(x, y, x+w, y+h))

            # Generate an embedding for the detected face
            embedding = net.forward(frame, landmarks)

            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5)

            # Detect smile in the face region
            smiles = smile_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5)

            # Check if both eyes and smile are detected
            if len(eyes) > 1 and len(smiles) > 0:
                # Calculate the confidence percentage
                confidence_percent = round((len(eyes) + len(smiles)) / 10 * 100)

                # Determine the emotion label based on the detected face region and embedding
                if confidence_percent > 60:
                    if embedding is not None:
                        prediction = model.predict(embedding)[0]
                        emotion_label = prediction

                else:
                    emotion_label = "Neutral"

                # Display the emotion label and confidence percentage next to the recognized face
                text = f"{emotion_label} ({confidence_percent}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the captured frame
        cv2.imshow('Captured Frame', frame)

        # Check for key presses
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Call the recognize_emotion_and_face function
    recognize_emotion_and_face()

if __name__ == '__main__':
    main()

