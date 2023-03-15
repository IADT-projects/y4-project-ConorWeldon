import requests
import json
import cv2
import numpy as np

# Replace <subscription_key> with your Azure Face API subscription key
subscription_key = "006ac883c7664a7d854fa47fd1d6aa3e"

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
        'returnFaceAttributes': 'emotion,age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'returnFaceId': 'true'
    }

    while True:
        # Capture a frame from the webcam video stream
        ret, frame = cap.read()

        # Display the live camera feed
        cv2.imshow('Live Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        image_data = np.array(buffer).tobytes()

        if not ret:
            print("Error: An error occurred while encoding the image.")
            return None

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
            # Convert the first face found to JPEG format and send it to Azure Face API
            try:
                face_data = image_data[response_json[0]['faceRectangle']['y']:response_json[0]['faceRectangle']['y']+response_json[0]['faceRectangle']['height'],
                            response_json[0]['faceRectangle']['x']:response_json[0]['faceRectangle']['x']+response_json[0]['faceRectangle']['width']]
                _, face_jpeg_data = cv2.imencode('.jpg', face_data)
                # Convert the first face found to JPEG format and send it to Azure Face API
                try:
                    face_data = image_data[response_json[0]['faceRectangle']['y']:response_json[0]['faceRectangle']['y'] + response_json[0]['faceRectangle']['height'],
                                response_json[0]['faceRectangle']['x']:response_json[0]['faceRectangle']['x'] + response_json[0]['faceRectangle']['width']]
                    _, face_jpeg_data = cv2.imencode('.jpg', face_data)
                except Exception as e:
                    print(f"Error: An error occurred while cropping the face: {e}")
                    return None

                try:
                    response = requests.post(
                        endpoint + "/detect",
                        headers=headers,
                        params=params,
                        data=face_jpeg_data.tobytes()
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

                # check if emotions are detected
                if len(response_json) > 0 and 'faceAttributes' in response_json[0]:
                    emotions = response_json[0]['faceAttributes']['emotion']
                    return emotions
                else:
                    print("No emotions were detected in the face.")
            except Exception as e:
                    print(f"Error: An error occurred while cropping the face: {e}")
                    return None
        else:
            print("No faces were detected in the image.")

    # Stop the webcam capture and close the window after processing the first face
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    result = recognize_emotion_and_face()
    print(result)
