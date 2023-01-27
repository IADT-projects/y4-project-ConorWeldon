# Second attempt at creating a facial recognition app

import requests # used to make HTTP requests
import json # used to parse the JSON response from the API

# Replace <subscription_key> with your Azure Face API subscription key
subscription_key = "006ac883c7664a7d854fa47fd1d6aa3e"

# Replace <endpoint> with the endpoint for your Azure Face API instance
endpoint = "https://smartemotionalmirror.cognitiveservices.azure.com"

# Define headers for the API request, including the subscription key
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

# function to recognize emotions in an image
def recognize_emotion(image_path):
    """
    Recognize emotions in an image using Azure Face API

    Parameters:
        image_path (str): the path of the image to be processed

    Returns:
        dict: a dictionary of emotions and their scores or None if an error occurs
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
        'returnFaceAttributes': 'emotion'
    }

    # Send a POST request to the Azure Face API endpoint
    try:
        response = requests.post(
            endpoint + "/detect", # API endpoint for detecting faces
            headers=headers, # headers including the subscription key
            params=params, # request parameters
            data=image_data # image data
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
        # extract emotions from the response
        emotions = response_json[0]['faceAttributes']['emotion']
        return emotions
    else:
        print("Error: No faces were detected in the image.")
        return None

# Test the function with an image
emotions = recognize_emotion("Build\DSC_0478_0624.jpg")
print(emotions)
