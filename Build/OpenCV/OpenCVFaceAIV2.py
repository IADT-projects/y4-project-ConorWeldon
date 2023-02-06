import cv2
import dlib
print(dlib.__version__)

# Load the pre-trained model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Loop over the faces
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # ...

    # Display the frame
    cv2.imshow("Emotion Recognition", frame)

    # Break the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
