# Sixth attempt at creating a facial recognition app

import requests
import json
import cv2

# Replace <subscription_key> with your Azure Face API subscription key
subscription_key = "006ac883c7664a7d854fa47fd1d6aa3e"

# Replace <endpoint> with the endpoint for your Azure Face API instance
endpoint = "https://smartemotionalmirror.cognitiveservices.azure.com/"

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
        'returnFaceAttributes': 'emotion,age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'returnFaceId': 'true'
    }

    while True:
        # Capture a frame from the webcam video stream
        ret, frame = cap.read()

        # Convert the frame to JPEG format
        ret, image_data = cv2.imencode('.jpg', frame)

        if not ret:
            print("Error: An error occurred while encoding the image.")
            return None

        # Send a POST request to the Azure Face API endpoint
        try:
            response = requests.post(
                endpoint + "/detect",
                headers=headers,
                params=params,
                data=image_data.tobytes()
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

    # Release the resources used by the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


