import requests  # Library to make HTTP requests
import json  # Library to work with JSON data
import cv2  # Library to work with computer vision, including video capturing
import asyncio  # Library for asynchronous programming
import io  # Library for input and output operations
import os  # Library to interact with the operating system
import sys  # Library to access system-specific parameters and functions
import time  # Library to measure time intervals
import uuid  # Library to generate universally unique identifiers
from urllib.parse import urlparse  # Library to work with URLs
from io import BytesIO  # Library for input and output operations with bytes
from PIL import Image, ImageDraw  # Library for image processing
from azure.cognitiveservices.vision.face import FaceClient  # Library for working with Azure Face API
from msrest.authentication import CognitiveServicesCredentials  # Library for authentication with Azure Face API
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition  # Models for Azure Face API

# This key will serve all examples in this document.
KEY = "006ac883c7664a7d854fa47fd1d6aa3e"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://smartemotionalmirror.cognitiveservices.azure.com/"

# Headers for HTTP requests to Azure Face API
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

# Create a client object for Azure Face API
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Create a new person group with a random UUID
testgroup = str(uuid.uuid4())

# Create persons within the group and assign names to them
woman = face_client.person_group_person.create(testgroup, name="Woman")
man = face_client.person_group_person.create(testgroup, name="Man")
child = face_client.person_group_person.create(testgroup, name="Child")

# Specify the parameters for the face detection API
params = {
    'returnFaceAttributes': 'emotion,age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
    'returnFaceId': 'true'
}

# Define a function to recognize emotions and faces in real time video stream
def recognize_emotion_and_face():
    try:
        # Initialize a connection to the default camera (webcam)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to open webcam")
    except Exception as e:
        print("Error: ", e)
        return None

    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()
        # Convert the frame to JPEG format
        ret, image_data = cv2.imencode('.jpg', frame)

        if not ret:
            print("Error: An error occurred while encoding the image.")
            return None

        try:
            # Call the face detection API with the image data
            response = requests.post(
                endpoint + "/detect",
                headers=headers,
                params=params,
                data=image_data.tobytes()
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(f"Error: An error occurred while calling the API: {err}")
            return None

        try:
            # Parse the JSON response from the API
            response_json = json.loads(response.text)
        except json.decoder.JSONDecodeError as err:
            # Handle JSON parsing errors
            print(f"Error: An error occurred while parsing the response: {err}")
            return None


        if len(response_json) > 0:
            # Extract face IDs from the JSON response
            face_ids = [face['faceId'] for face in response_json]
            try:
                # Call the "group" API to group faces by similarity
                group_response = requests.post(
                    endpoint + "/group",
                    headers=headers,
                    json={"faceIds": face_ids}
                )
                group_response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                # Handle API call errors
                print(f"Error: An error occurred while calling the API: {err}")
                return None
            try:
                # Parse the JSON response from the "group" API
                group_response_json = json.loads(group_response.text)
            except json.decoder.JSONDecodeError as err:
                # Handle JSON parsing errors
                print(f"Error: An error occurred while parsing the response: {err}")
                return None
            # Extract the groups from the JSON response
            groups = group_response_json["groups"]
            return groups
        else:
            # Handle case where no faces were detected in the image
            print("No faces were detected in the image.")
        break

    # Release the resources used by the webcam (cap.release()) and close the window used by OpenCV (destroyAllWindows)
    cap.release()
    cv2.destroyAllWindows()
