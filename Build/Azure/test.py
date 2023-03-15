import asyncio
import io
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition


# This key will serve all examples in this document.
KEY = "006ac883c7664a7d854fa47fd1d6aa3e"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://smartemotionalmirror.cognitiveservices.azure.com/"

# Base url for the Verify and Facelist/Large Facelist operations
IMAGE_BASE_URL = 'https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/'

# Used in the Person Group Operations and Delete Person Group examples.
# You can call list_person_groups to print a list of preexisting PersonGroups.
# SOURCE_PERSON_GROUP_ID should be all lowercase and alphanumeric. For example, 'mygroupname' (dashes are OK).
testgroup = str(uuid.uuid4()) # assign a random ID (or name it anything)

# Used for the Delete Person Group example.
TARGET_testgroup = str(uuid.uuid4()) # assign a random ID (or name it anything)

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

'''
Create the PersonGroup
'''
# Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
print('Person group:', testgroup)
face_client.person_group.create(person_group_id=testgroup, name=testgroup, recognition_model='recognition_04')

'''
Train PersonGroup
'''
# Train the person group
print("pg resource is {}".format(testgroup))
face_client.person_group.train(testgroup)

while (True):
    training_status = face_client.person_group.get_training_status(testgroup)
    print("Training status: {}.".format(training_status.status))
    print()
    if (training_status.status is TrainingStatusType.succeeded):
        break

'''
Detect faces and register them to each person
'''
# Find all jpeg images of friends in working directory (TBD pull from web instead)
woman_images = ["https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Mom1.jpg", "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Mom2.jpg"]
man_images = ["https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad1.jpg", "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad2.jpg"]
child_images = ["https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Son1.jpg", "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Son2.jpg"]

# Add to woman person
for image in woman_images:
    # Check if the image is of sufficient quality for recognition.
    sufficientQuality = True
    detected_faces = face_client.face.detect_with_url(url=image, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
    for face in detected_faces:
        if face.face_attributes.quality_for_recognition != QualityForRecognition.high:
            sufficientQuality = False
            break
        face_client.person_group_person.add_face_from_url(testgroup, woman.person_id, image)
        print("face {} added to person {}".format(face.face_id, woman.person_id))

    if not sufficientQuality: continue

# Add to man person
for image in man_images:
    # Check if the image is of sufficient quality for recognition.
    sufficientQuality = True
    detected_faces = face_client.face.detect_with_url(url=image, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
    for face in detected_faces:
        if face.face_attributes.quality_for_recognition != QualityForRecognition.high:
            sufficientQuality = False
            break
        face_client.person_group_person.add_face_from_url(testgroup, man.person_id, image)
        print("face {} added to person {}".format(face.face_id, man.person_id))

    if not sufficientQuality: continue

# Add to child person
for image in child_images:
    # Check if the image is of sufficient quality for recognition.
    sufficientQuality = True
    detected_faces = face_client.face.detect_with_url(url=image, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
    for face in detected_faces:
        if face.face_attributes.quality_for_recognition.value < 0.5:
            print(f"Skipping face with ID {face.face_id} due to low recognition quality score.")
            continue
        # rest of your code for processing high-quality faces


'''
Identify a face against a defined PersonGroup
'''
# Group image for testing against
test_image = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/identification1.jpg"

print('Pausing for 10 seconds to avoid triggering rate limit on free account...')
time.sleep (10)

# Detect faces
face_ids = []
# We use detection model 3 to get better performance, recognition model 4 to support quality for recognition attribute.
faces = face_client.face.detect_with_url(test_image, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
for face in faces:
    # Only take the face if it is of sufficient quality.
    if face.face_attributes.quality_for_recognition == QualityForRecognition.high or face.face_attributes.quality_for_recognition == QualityForRecognition.medium:
        face_ids.append(face.face_id)

# Identify faces
results = face_client.face.identify(face_ids, testgroup)
print('Identifying faces in image')
if not results:
    print('No person identified in the person group')
for identifiedFace in results:
    if len(identifiedFace.candidates) > 0:
        print('Person is identified for face ID {} in image, with a confidence of {}.'.format(identifiedFace.face_id, identifiedFace.candidates[0].confidence)) # Get topmost confidence score

        # Verify faces
        verify_result = face_client.face.verify_face_to_person(identifiedFace.face_id, identifiedFace.candidates[0].person_id, testgroup)
        print('verification result: {}. confidence: {}'.format(verify_result.is_identical, verify_result.confidence))
    else:
        print('No person identified for face ID {} in image.'.format(identifiedFace.face_id))
 
'''
    # Train PersonGroup
# Train the person group
print("pg resource is {}".format(testgroup))
face_client.person_group.train(testgroup)

# Wait for training to complete
while (True):
    training_status = face_client.person_group.get_training_status(testgroup)
    print("Training status: {}.".format(training_status.status))
    if (training_status.status is TrainingStatusType.succeeded):
        break
    time.sleep(5)

# Identify faces in images
# Define a test image URL
test_image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/test-image-person-group.jpg"

# Detect faces in the test image
detected_faces = face_client.face.detect_with_url(url=test_image_url)

# Identify each face in the test image
for face in detected_faces:
    identified_faces = face_client.face.identify([face.face_id], person_group_id=testgroup)

    # Print the ID of each identified person
    for identified_face in identified_faces:
        if len(identified_face.candidates) > 0:
            person_id = identified_face.candidates[0].person_id
            person = face_client.person_group_person.get(testgroup, person_id)
            print("Person {} is identified in the test image.".format(person.name))
'''

print()
print('End of quickstart.')