# First attempt at creating a facial recognition app

import requests # used to make HTTP requests
import json # used to parse the JSON response from the API
import ssl # This line imports the 'ssl' module which will allow your code to connect to HTTPS URLs

# from json import loads

# Replace <subscription_key> with your Azure Cognitive Services subscription key
subscription_key = "006ac883c7664a7d854fa47fd1d6aa3e"

# Replace <endpoint> with the endpoint for your Azure Cognitive Services instance
endpoint = "https://smartemotionalmirror.cognitiveservices.azure.com"

headers = {
    'Content-Type': 'application/octet-stream', # binary image data
    'Ocp-Apim-Subscription-Key': subscription_key # Azure Cognitive Services subscription key
}

# function to recognize emotions in an image
def recognize_emotion(image_path):
    # open image and read it as binary data
    with open(image_path, "rb") as image:
        image_data = image.read()

    params = {
        'returnFaceAttributes': 'emotion' # request the emotion attribute
    }

    # Make a POST request to the Azure Cognitive Services Face API
    response = requests.post(
        endpoint + "/face/v1.0/detect", # API endpoint for detecting faces
        headers=headers, # headers including the subscription key
        params=params, # request parameters
        data=image_data # image data
    )

    # parse the response as JSON
    response_json = json.loads(response.text)
    if len(response_json) > 0:
        # extract emotions from the response
        emotions = response_json[0]['faceAttributes']['emotion']
        return emotions
    else:
        return None

# Test the function with an image
emotions = recognize_emotion("Build\Images\DSC_0478_0624.jpg")
print(emotions)
