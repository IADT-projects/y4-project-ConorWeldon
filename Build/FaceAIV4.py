# Fourth attempt at creating a facial recognition app

import requests
import json

# Replace <subscription_key> with your Azure Face API subscription key
subscription_key = "<subscription_key>"

# Replace <endpoint> with the endpoint for your Azure Face API instance
endpoint = "<endpoint>"

# Define headers for the API request, including the subscription key
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

def recognize_emotion_and_face(image_path):
    """
    Recognize emotions and faces in an image using Azure Face API

    Parameters:
        image_path (str): the path of the image to be processed

    Returns:
        dict: a dictionary of emotions and faces information or None if an error occurs
    """
    if not image_path:
        print("Error: Image path is not provided.")
        return None
    # Try to open the image and read its binary data
    try:
        with open(image_path, "rb") as image:
            image_data = image.read()
    # Handle FileNotFoundError
    except FileNotFoundError:
        print("Error: Image not found. Please check the path and try again.")
        return None
    # Handle any other error that may occur while reading the image
    except:
        print("Error: An error occurred while reading the image.")
        return None

    # Define the parameters for the API request
    params = {
        'returnFaceAttributes': 'emotion,age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'returnFaceId': 'true'
    }

    # Send a POST request to the Azure Face API endpoint
    try:
        response = requests.post(
            endpoint + "/detect",
            headers=headers,
            params=params,
            data=image_data
        )
        response.raise_for_status()
    # handle http errors
    except requests.exceptions.HTTPError as err:
        print(f"Error: An error occurred while calling the API: {err}")
        return None

    # Parse the JSON response
    try:
        response_json = json.loads(response.text)
    except json.decoder.JSONDecodeError as err:
        print(f"Error: An error occurred while parsing the response: {err}")
        return None

    # check if faces are detected
    if len(response_json) > 0:
        return response_json
    else:
        print("Error: No faces were detected in the image.")
        return None

# Test the function with an image
emotions_and_face = recognize_emotion_and_face("image.jpg")
print(emotions_and_face)


# Create a face group
group_id = "my_group_id"
group_name = "My Group"

data = {
    "name": group_name
}

response = requests.put(
    endpoint + f"/largepersongroups/{group_id}",
    headers=headers,
    json=data
)

response.raise_for_status()

# Add a known face to the group
person_id = "my_person_id"
person_name = "My Person"
image_path = "my_person.jpg"

with open(image_path, "rb") as image:
    image_data = image.read()

data = {
    "name": person_name,
    "userData": "My Person"
}

response = requests.post(
    endpoint + f"/largepersongroups/{group_id}/persons",
    headers=headers,
    json=data,
    data=image_data
)

response.raise_for_status()

# Identify a face in an image
test_image_path = "test_image.jpg"

with open(test_image_path, "rb") as image:
    image_data = image.read()

params = {
    "largePersonGroupId": group_id
}

response = requests.post(
    endpoint + "/identify",
    headers=headers,
    params=params,
    data=image_data
)

response.raise_for_status()
