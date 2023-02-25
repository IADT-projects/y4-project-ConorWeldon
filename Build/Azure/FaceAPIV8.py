import requests
import json
import cv2
import asyncio
import io
import os
import sys
import time
import uuid
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition


subscription_key = "006ac883c7664a7d854fa47fd1d6aa3e"
endpoint = "https://smartemotionalmirror.cognitiveservices.azure.com/"

headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

testgroup = str(uuid.uuid4())
TARGET_testgroup = str(uuid.uuid4())

print('Person group:', testgroup)
face_client.person_group.create(person_group_id=testgroup, name=testgroup, recognition_model='recognition_04')

woman = face_client.person_group_person.create(testgroup, name="Woman")
man = face_client.person_group_person.create(testgroup, name="Man")
child = face_client.person_group_person.create(testgroup, name="Child")

params = {
    'returnFaceAttributes': 'emotion,age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
    'returnFaceId': 'true'
}

def recognize_emotion_and_face():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to open webcam")
    except Exception as e:
        print("Error: ", e)
        return None

    while True:
        ret, frame = cap.read()
        ret, image_data = cv2.imencode('.jpg', frame)

        if not ret:
            print("Error: An error occurred while encoding the image.")
            return None

        try:
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
            response_json = json.loads(response.text)
        except json.decoder.JSONDecodeError as err:
            print(f"Error: An error occurred while parsing the response: {err}")
            return None

        if len(response_json) > 0:
            face_ids = [face['faceId'] for face in response_json]
            try:
                group_response = requests.post(
                    endpoint + "/group",
                    headers=headers,
                    json={"faceIds": face_ids}
                )
                group_response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                print(f"Error: An error occurred while calling the API: {err}")
                return None
            try:
                group_response_json = json.loads(group_response.text)
            except json.decoder.JSONDecodeError as err:
                print(f"Error: An error occurred while parsing the response: {err}")
                return None
            groups = group_response_json["groups"]
            return groups
        else:
            print("No faces were detected in the image.")
        break

    cap.release()
    cv2.destroyAllWindows()
