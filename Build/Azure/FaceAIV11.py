import cv2
import numpy as np
import requests
import json

# Load the pre-trained model for face detection
face_model = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load the pre-trained model for emotion recognition
emotion_model = cv2.dnn.readNetFromTorch("emotion-ferplus-8.onnx")

# Define the emotions
EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Replace <subscription_key> with your Azure Face API subscription key
subscription_key = "bb67bd956c014f3ea8cf89621c75de21"

# Replace <endpoint> with the endpoint for your Azure Face API instance
endpoint = "https://smartemotionalmirror.cognitiveservices.azure.com"

# Define headers for the API request, including the subscription key
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

def recognize_emotion_and_face():
    """
    Recognize emotions and faces in a webcam video stream using Azure Face API

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

    # Define the parameters for the API request
    params = {
        'returnFaceAttributes': 'headpose,mask,qualityforrecognition',
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'true',
        'recognitionModel': 'recognition_04',
        'returnRecognitionModel': 'true',
        'detectionModel': 'detection_03',
        'faceIdTimeToLive': '86400'
    }

    results = {}
    
    while True:
        # Capture a frame from the webcam video stream
        ret, frame = cap.read()

        # Display the live camera feed
        cv2.imshow('Live Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Convert the frame to grayscale and detect faces using the face detection model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (h, w) = frame.shape[:2]
        face_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_model.setInput(face_blob)
        face_detections = face_model.forward()

        # Iterate over the face detections
        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            # If the detection confidence is high enough, estimate the emotions
            if confidence > 0.5:
                # Extract the face region and convert it to grayscale
                face_box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = face_box.astype("int")
                face = gray[y:y1, x:x1]
                                # Resize the face region to 48x48 pixels and preprocess it
                face = cv2.resize(face, (48, 48))
                face = face.astype("float") / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)

                # Feed the preprocessed face region to the emotion recognition model
                emotions_model.setInput(face)
                emotion_scores = emotions_model.forward()[0]

                # Map the predicted emotion scores to the corresponding emotions
                emotions = {}
                for i, emotion_score in enumerate(emotion_scores):
                    emotions[EMOTIONS[i]] = float(emotion_score)

                # Add the estimated emotions to the results dictionary
                results["emotions"] = emotions

                # Draw a rectangle around the face and label it with the estimated emotion
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, max(emotions, key=emotions.get), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Live Camera Feed', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    return results

               
