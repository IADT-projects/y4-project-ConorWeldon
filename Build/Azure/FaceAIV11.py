import cv2

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'C:\Users\conor\OneDrive\Desktop\y4-project-ConorWeldon\Build\Azure\haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
emotion_model = cv2.dnn.readNetFromCaffe("Build\Azure\deploy.prototxt.txt", "Build\Azure\emotion_detection_model.hdf5.xml")

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Loop through the frames captured from the camera
while True:
    # Read the current frame from the video capture stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale image
        face_gray = gray[y:y+h, x:x+w]
        
        # Resize the face region to match the input size required by the emotion detection model
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize the pixel values of the resized face region
        face_normalized = face_resized.astype("float32") / 255.0
        
        # Reshape the face region to match the input shape required by the emotion detection model
        face_reshaped = face_normalized.reshape((1, 1, 48, 48))
        
        # Pass the reshaped face region through the emotion detection model and get the predictions
        emotion_model.setInput(face_reshaped)
        predictions = emotion_model.forward()
        
        # Find the index of the predicted emotion with the highest probability
        emotion_index = np.argmax(predictions)
        
        # Map the index to the corresponding emotion label
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        emotion_label = emotion_labels[emotion_index]
        
        # Draw a rectangle around the detected face and display the predicted emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
